import numpy as np
import pandas as pd
import argparse
import math
from types import SimpleNamespace
    
class SolarPositionTemporal():
    ## from https://gml.noaa.gov/grad/solcalc/calcdetails.html
    def __init__(self, timezone=0):
        self.timezone = timezone
        
    def calc_solar_angles(self, timestamps, lat, lon):
        # time in hours past midnight, e.g. 2.3/24
        time = timestamps.dt.hour + timestamps.dt.minute/60 + timestamps.dt.second/60/60        
        base_date = pd.to_datetime('01/01/1900', format='%d/%m/%Y')
        # add 2 from indexing/counting shift?
        jul_day = (2 + (pd.to_datetime(timestamps.dt.date) - base_date).dt.total_seconds()/60/60/24 + 
            2415018.5 + time/24 - self.timezone/24)
        jul_century = (jul_day - 2451545)/36525
        
        # calculate location independent quantities
        geom_mean_anom_sun = 357.52911 + jul_century*(35999.05029 - 0.0001537*jul_century)    
        sun_eq_cent = (np.sin(np.deg2rad(geom_mean_anom_sun)) * 
            (1.914602 - jul_century*(0.004817 + 0.000014*jul_century)) + 
            np.sin(np.deg2rad(2*geom_mean_anom_sun)) * 
            (0.019993 - 0.000101*jul_century) + 
            np.sin(np.deg2rad(3*geom_mean_anom_sun)) * 0.000289
        )
        geom_mean_long_sun = np.mod(
            280.46646 + jul_century*(36000.76983 + jul_century*0.0003032), 360)
        sun_true_long = geom_mean_long_sun + sun_eq_cent
        sun_app_long = (sun_true_long - 0.00569 - 
            0.00478*np.sin(np.deg2rad(125.04 - 1934.136*jul_century)))

        mean_obliq_eliptic = (23 + (26 + (21.448 - jul_century*(46.815 + 
            jul_century*(0.00059 - jul_century*0.001813)))/60)/60)
        obliq_corr = (mean_obliq_eliptic + 
            0.00256*np.cos(np.deg2rad(125.04 - 1934.136*jul_century)))
        sun_declin = np.rad2deg(np.arcsin((
            np.sin(np.deg2rad(obliq_corr)) * 
            np.sin(np.deg2rad(sun_app_long))).clip(-1,1))
        ) 

        ecc_earth_orbit = (0.016708634 - 
            jul_century*(0.000042037 + 0.0000001267*jul_century))
        var_y = np.tan(np.deg2rad(obliq_corr/2)) * np.tan(np.deg2rad(obliq_corr/2))
        eq_of_time = (4*np.rad2deg(var_y * np.sin(2*np.deg2rad(geom_mean_long_sun)) - 
            2*ecc_earth_orbit * np.sin(np.deg2rad(geom_mean_anom_sun)) + 
            4*ecc_earth_orbit * var_y * 
                np.sin(np.deg2rad(geom_mean_anom_sun))*np.cos(2*np.deg2rad(geom_mean_long_sun)) - 
            0.5*var_y*var_y * np.sin(4*np.deg2rad(geom_mean_long_sun)) - 
            1.25*ecc_earth_orbit*ecc_earth_orbit * np.sin(2*np.deg2rad(geom_mean_anom_sun)))
        )
        
        true_solar_time = np.mod((time/24) * 1440 + eq_of_time + 
            4*lon - 60*self.timezone, 1440)        
        
        hour_angle = true_solar_time.copy()
        mask = (true_solar_time/4) < 0
        hour_angle[mask] = true_solar_time[mask]/4 + 180
        hour_angle[~mask] = true_solar_time[~mask]/4 - 180
        
        # solar zenith angle in degrees from vertical
        rlat = math.radians(lat)
        rsundeclin = np.deg2rad(sun_declin)
        solar_zenith_angle = np.rad2deg(
            np.arccos((np.sin(rlat) * np.sin(rsundeclin) + 
                    np.cos(rlat) * np.cos(rsundeclin) * 
                        np.cos(np.deg2rad(hour_angle))).clip(-1,1))
        )
        
        # solar azimuth angle in degrees clockwise from north
        solar_azimuth_angle = solar_zenith_angle.copy()
        rzenith = np.deg2rad(solar_zenith_angle)        
        mask = hour_angle>0
        solar_azimuth_angle[mask] = np.mod(
            np.rad2deg(np.arccos((
                ((np.sin(rlat) * np.cos(rzenith[mask])) - 
                    np.sin(rsundeclin)) / 
                (np.cos(rlat) * np.sin(rzenith[mask]))).clip(-1,1))) + 180, 360
        )
        solar_azimuth_angle[~mask] = np.mod(
            540 - np.rad2deg(np.arccos((
                ((np.sin(rlat) * np.cos(rzenith[~mask])) - 
                    np.sin(rsundeclin)) / 
                (np.cos(rlat) * np.sin(rzenith[~mask]))).clip(-1,1))), 360
        )
        # fix nans at 360
        mask = np.isnan(solar_azimuth_angle)
        solar_azimuth_angle[mask] = 0.
        
        # solar elevation in degrees from horizontal
        solar_elevation = 90 - solar_zenith_angle
        atmos_refraction = solar_elevation.copy()
        rsolelev = np.deg2rad(solar_elevation)
        mask = np.bitwise_and(solar_elevation > 5, solar_elevation <= 85)
        atmos_refraction[mask] = (
            58.1 / np.tan(rsolelev[mask]) - 
            0.07 / np.power(np.tan(rsolelev[mask]), 3) +
            0.000086 / np.power(np.tan(rsolelev[mask]), 5)
        )
        mask2 = np.bitwise_and(solar_elevation <= 5, solar_elevation > -0.575)
        atmos_refraction[mask2] = (
            1735 + solar_elevation[mask2]*(-518.2 + 
                solar_elevation[mask2]*(103.4 + 
                    solar_elevation[mask2]*(-12.79 + 
                        solar_elevation[mask2]*0.711)))
        )
        mask3 = np.bitwise_or(mask, mask2)
        atmos_refraction[~mask3] = -20.772 / np.tan(rsolelev[~mask3])

        atmos_refraction /= 3600
        solar_elevation = solar_elevation + atmos_refraction
        return pd.DataFrame({'solar_elevation':solar_elevation.values,
                             'solar_azimuth_angle':solar_azimuth_angle.values},
                             index=timestamps)



