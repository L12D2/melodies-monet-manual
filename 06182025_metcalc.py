# will need to make this an optional dependency if we proceed in using this. 
import metpy

# import specific metpy libraries needed for each calculation
from metpy.calc import dewpoint_from_specific_humidity
from metpy.calc import relative_humidity_from_specific_humidity
# if we want to add a standard hydrometeorological comparison (e.g. Dewpoint), we need 
# to append the observations. E.g. in the airnow we only have Relative humidity. 
# metpy has an option to do dewpoint from RELH iirc. 
from metpy.calc import wind_speed

# addl libraries to make the world go round
from metpy.units import units
import numpy as np
import pandas as pd

# calc dewpoint 
def dewpoint(obj, varmap = None, output_key = "dewpoint"):
    # grab variable names from the yaml 
    pressure_key = varmap['pressure'] if varmap and 'pressure' in varmap else 'surfpres_pa'
    specific_hum_key = varmap['specific_hum'] if varmap and 'specific_hum' in varmap else 'specific_hum'

    pressure = obj[pressure_key]
    specific_hum = obj[specific_hum_key]

    dpt = (metpy.calc.dewpoint_from_specific_humidity(
        pressure * units.Pa,
        specific_hum * units("kg/kg")
    )).metpy.convert_units("K")

    dpt_np = dpt.astype("float64").values

    # add this to obj
    if hasattr(obj, "coords") and hasattr(obj, "dims"):  
        obj[output_key] = (obj[pressure_key].dims, dpt_np)
        obj[output_key].attrs["units"] = "K"
        return obj
    elif isinstance(obj, dict):
        obj[output_key] = dpt_np
        return obj
    else:
        return dpt_np

# calc relative humidity
def relh(obj, varmap = None, output_key = "rel_hum"):
    # grab variable names from the yaml 
    pressure_key = varmap["pressure"] if varmap and "pressure" in varmap else "surfpres_pa"
    specific_hum_key = varmap["specific_hum"] if varmap and "specific_hum" in varmap else "specific_hum"
    temperature_key = varmap["temperature"] if varmap and "pressure" in varmap else "temperature_k"

    pressure = obj[pressure_key]
    specific_hum = obj[specific_hum_key]
    temperature = obj[temperature_key]

    rlh = (metpy.calc.relative_humidity_from_specific_humidity(
        pressure * units.Pa,
        temperature * units.kelvin, 
        specific_hum * units("kg/kg")
    )).metpy.convert_units("%")

    rlh_np = rlh.astype("float64").values

    # add this to obj
    if hasattr(obj, "coords") and hasattr(obj, "dims"): 
        obj[output_key] = (obj[pressure_key].dims, rlh_np)
        obj[output_key].attrs["units"] = "K"
        return obj
    elif isinstance(obj, dict):
        obj[output_key] = rlh_np
        return obj
    else:
        return rlh_np

def wspd(obj, varmap = None, output_key = "windspeed"):
    # grab variable names from the yaml 
    u_key = varmap["u_comp"] 
    v_key = varmap["v_comp"] 

    u = obj[u_key]
    v = obj[v_key]

    wspd = (metpy.calc.wind_speed(
        u * units("m/s"),
        v * units("m/s")
        )).metpy.convert_units("m/s")

    wspd_np = wspd.astype("float64").values
    
    # add this to obj
    if hasattr(obj, "coords") and hasattr(obj, "dims"):  
        obj[output_key] = (u.dims, wspd_np)
        obj[output_key].attrs["units"] = "m/s"
        return obj
    elif isinstance(obj, dict):
        obj[output_key] = wspd_np
        return obj
    else:
        return wspd_np
    
def wdir(obj, varmap = None, output_key = "winddir"):
    # grab variable names from the yaml 
    u_key = varmap["u_comp"] 
    v_key = varmap["v_comp"] 
    
    u = obj[u_key]
    v = obj[v_key]

    # metpy version of this is throwing in weird dimensions so calc by hand

    wdr_rad = 90 * units.degrees - np.arctan2(-v, -u)
    wdr_deg = np.degrees(wdr_rad)
    winddir = wdr_deg % 360
    print(winddir)
    
    # Add this to obj
    if hasattr(obj, "coords") and hasattr(obj, "dims"):  
        obj[output_key] = (u.dims, winddir.values)
        obj[output_key].attrs["units"] = "degrees"
        return obj
    elif isinstance(obj, dict):
        obj[output_key] = winddir
        return obj
    else:
        return winddir

