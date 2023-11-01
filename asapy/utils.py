import pandas as pd
from scipy.stats import ttest_ind
import math
import json
import ast
import asaclient

def prepare_simulation_batch(sim: asaclient.Simulation) -> asaclient.Simulation:
    """
    Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run in batch mode.

    Args:
        sim (Simulation): The simulation instance for which the simulation setup needs to be prepared.

    Returns:
        Simulation: The updated Simulation instance with the added recorder configurations in its subcomponents.
    """
    batch_recorder = {
            "identifier": "AsaWsDBRecorder@AsaModels",
            "attributes": {},
            "subcomponents": {
                "asaBasicMsgs": [
                ],
                "asaCustomMsgs": [
                    {
                        "identifier": "AsaCustomMsg11@AsaModels",
                        "attributes": {}
                    }
                ]
            }
        }
    recorders=[batch_recorder, ]
    sim = dict(sim)
    sim['station']['subcomponents']['recorders'] = recorders
    return asaclient.Simulation(**sim)

def prepare_simulation_tacview(sim: asaclient.Simulation) -> asaclient.Simulation:
    """
    Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run on Tacview.

    Args:
        sim (Simulation): The simulation instance for which the simulation setup needs to be prepared.

    Returns:
        Simulation: The updated Simulation instance with the added recorder configurations in its subcomponents.
    """
    tac_recorder = {
        "identifier": "AsaWsTacviewRecorder@AsaModels",
        "attributes": {},
        "subcomponents": {}
    }
    recorders=[tac_recorder, ]
    sim = dict(sim)
    sim['station']['subcomponents']['recorders'] = recorders
    return asaclient.Simulation(**sim)

def load_simulation(path: str) -> asaclient.Simulation:
    """
    Loads a Simulation object from a JSON file.

    This method accepts a path to a JSON file, reads the content of the file and 
    creates a Simulation object using the data parsed from the file. 

    Args:
        path (str): The absolute or relative path to the JSON file to be loaded.

    Returns:
        Simulation: The Simulation object created from the loaded JSON data.
    """
    with open(path, "r") as f:
        sim_data = json.load(f)
    simulation = asaclient.Simulation(**sim_data)
    return simulation
    
def json_to_df(self, json, id='id') -> pd.DataFrame:
    """
    Convert a JSON object to a pandas DataFrame and set the index to the given id column.

    Args:
        json (dict): A JSON object.
        id (str): The name of the column to set as the index. Default is 'id'.

    Returns:
        pandas.DataFrame: A DataFrame representation of the JSON object.
    """
    return pd.DataFrame(json).set_index(id)


def list_to_df(arr, id='id'):
    """
    Convert a list of dictionaries to a pandas DataFrame and set the index to the given id column.

    Args:
        arr (list): A list of dictionaries.
        id (str): The name of the column to set as the index. Default is 'id'.

    Returns:
        pandas.DataFrame: A DataFrame representation of the list of dictionaries.
    """
    return pd.DataFrame(arr).set_index(id).sort_index()


def unique_list(list1):
    """
    Return a list of unique values in the given list.

    Args:
        list1 (list): A list of values.

    Returns:
        list: A list of unique values in the input list.
    """
    # initialize a null list
    unique_arr = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_arr:
            unique_arr.append(x)
    return unique_arr


def get_parents_dict(dic, value):
    """
    Return a list of keys that lead to the given value in the given dictionary.

    Args:
        dic (dict): A dictionary to search.
        value: The value to search for in the dictionary.

    Returns:
        list: A list of keys that lead to the given value in the dictionary.
    """
    for k, v in dic.items():
        if isinstance(v, dict):
            p = get_parents_dict(v, value)
            if p:
                return [k] + p
        elif v == value:
            return [k]


def check_samples_similar(new_sample, last_sample, threshold):
    """
    Checks if two samples are similar based on a given threshold.

    Args:
        new_sample (np.ndarray): The new sample to compare.
        last_sample (np.ndarray): The last sample to compare.
        threshold (float): The threshold to use for comparison.

    Returns:
        bool: True if the samples are similar, False otherwise.
    """
    if (last_sample.mean() - new_sample.mean()) / last_sample.mean() < threshold and (
            1 - threshold < (new_sample.std() / last_sample.std()) < 1 + threshold):
        return True
    else:
        return False


def test_t(sample1, sample2, alpha=0.05):
    """
    Performs a t-test and compares the p-value with a given alpha value.

    Args:
        sample1 (np.ndarray): The first sample.
        sample2 (np.ndarray): The second sample.
        alpha (float, optional): The alpha value to use for comparison. Defaults to 0.05.

    Returns:
        bool: True if the samples are similar, False otherwise.
    """
    # perform t-test
    t_stat, p_value = ttest_ind(sample1, sample2)

    # compare p-value with alpha value
    if p_value > alpha:
        print("The two samples are similar.")
        return True
    else:
        print("The two samples are different.")
        return False

def convert_nested_string_to_dict(s):
    """
    Converts a string that contains a dictionary and JSON-formatted strings into a nested dictionary.
    
    Args:
        s (str): The input string containing a dictionary and JSON-formatted strings.
        
    Returns:
        dict: The output dictionary after conversion of JSON-formatted strings.
    """
    d = ast.literal_eval(s)
    for k, v in d.items():
        if isinstance(v, str):
            try:
                d[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return d

def find_key(nested_dict, target_key):
    """Find a key in a nested dictionary.

    Args:
        nested_dict (dict): The dictionary to search.
        target_key (str): The key to find.

    Returns:
        value: The value of the found key, or None if the key was not found.
    """
    for key, value in nested_dict.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_key(value, target_key)
            if result is not None:
                return result
    return None

def gen_dict_extract(key, var):
    """
    A generator function to iterate and yield values from a dictionary or list nested inside the dictionary, given a key.

    Args:
        key (str): The key to search for in the dictionary.
        var (dict or list): The dictionary or list to search.

    Yields:
        value: The value from the dictionary or list that corresponds to the given key.
    """
    if hasattr(var,'items'): 
        for k, v in var.items(): 
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result

def transform_stringified_dict(data):
    """
    Recursively converts stringified JSON parts of a dictionary or list into actual dictionaries or lists.
    
    This function checks if an item is a string and attempts to convert it to a dictionary or list 
    using `json.loads()`. If the conversion is successful, the function recursively processes the new 
    dictionary or list. If a string is not a valid JSON representation, it remains unchanged.
    
    Args:
        data (Union[dict, list, str]): Input data that might contain stringified JSON parts.

    Returns:
        Union[dict, list, str]: The transformed data with all stringified JSON parts converted 
        to dictionaries or lists.
        
    Raises:
        json.JSONDecodeError: If there's an issue decoding a JSON string. This is caught internally 
        and the original string is returned.
    """
    
    # If data is a string, try to parse it as JSON
    if isinstance(data, str):
        try:
            data = json.loads(data)
            return transform_stringified_dict(data)  # Recursive call
        except json.JSONDecodeError:  # String is not JSON
            return data
    
    # If data is a dictionary, process its values
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = transform_stringified_dict(value)
    
    # If data is a list, process its items
    if isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = transform_stringified_dict(value)
            
    return data

"""
Constants and Methods for Converting Distance Measurement Units
These constants and methods are equivalent to those of the file distance_utils.cpp of MIXR.

"""
# Constants for conversion between distances
FT2M = 0.30480  # Feet => Meters
"""
Conversion factor from feet to meters.
"""
M2FT = 1 / FT2M  # Meters => Feet
"""
Conversion factor from meters to feet.
"""
IN2M = 0.02540  # Inches => Meters
"""
Conversion factor from inches to meters.
"""
M2IN = 1 / IN2M  # Meters => Inches
"""
Conversion factor from meters to inches.
"""
NM2M = 1852.0  # Nautical Miles => Meters
"""
Conversion factor from nautical miles to meters.
"""
M2NM = 1 / NM2M  # Meters => Nautical Miles
"""
Conversion factor from meters to nautical miles.
"""
NM2FT = NM2M * M2FT  # Nautical Miles => Feet
"""
Conversion factor from nautical miles to feet.
"""
FT2NM = 1 / NM2FT  # Feet => Nautical Miles
"""
Conversion factor from feet to nautical miles.
"""
SM2M = 1609.344  # Statue Miles => Meters
"""
Conversion factor from statute miles to meters.
"""
M2SM = 1 / SM2M  # Meters => Statue Miles
"""
Conversion factor from meters to statute miles.
"""
SM2FT = 5280.0  # Statue Miles => Feet
"""
Conversion factor from statute miles to feet.
"""
FT2SM = 1 / SM2FT  # Feet => Statue Miles
"""
Conversion factor from feet to statute miles.
"""
KM2M = 1000.0  # Kilometers => Meters
"""
Conversion factor from kilometers to meters.
"""
M2KM = 1 / KM2M  # Meters => Kilometers
"""
Conversion factor from meters to kilometers.
"""
CM2M = 0.01  # Centimeters => Meters
"""
 Conversion factor from centimeters to meters.
"""
M2CM = 1 / CM2M  # Meters => Centimeters
"""
Conversion factor from meters to centimeters.
"""
UM2M = 0.000001  # Micrometer (Micron) => Meters
"""
 Conversion factor from micrometers to meters.
"""
M2UM = 1 / UM2M  # Meters => Micrometer (Micron)
"""
 Conversion factor from meters to micrometers.

"""

# Methods for conversion between distances
def meters_to_micrometers(v):
    """
    Convert meters to micrometers.

    Args:
        v (float): Value in meters.

    Returns:
        float: Value in micrometers.
    """
    return v * M2UM


def micrometers_to_meters(v):
    """
    Convert micrometers to meters.

    Args:
        v (float): Value in micrometers.

    Returns:
        float: Value in meters.
    """
    return v * UM2M


def meters_to_centimeters(v):
    """
    Convert meters to centimeters.

    Args:
        v (float): Value in meters.

    Returns:
        float: Value in centimeters.
    """
    return v * M2CM


def centimeters_to_meters(v):
    """
    Convert centimeters to meters.

    Args:
        v (float): Value in centimeters.

    Returns:
        float: Value in meters.
    """
    return v * CM2M


def meters_to_kilometers(v):
    """
    Convert meters to kilometers.

    Args:
        v (float): Value in meters.

    Returns:
        float: Value in kilometers.
    """
    return v * M2KM


def kilometers_to_meters(v):
    """
    Convert kilometers to meters.

    Args:
        v (float): Value in kilometers.

    Returns:
        float: Value in meters.
    """
    return v * KM2M


def meters_to_inches(v):
    """
    Convert meters to inches.

    Args:
        v (float): Value in meters.

    Returns:
        float: Value in inches.
    """
    return v * M2IN


def inches_to_meters(v):
    """
    Convert inches to meters.

    Args:
        v (float): Value in inches.

    Returns:
        float: Value
    """
    return v * IN2M


def meters_to_feet(v):
    """
    Converts a distance in meters to feet.

    Args:
        v (float): distance in meters

    Returns:
        float: distance in feet
    """
    return v * M2FT


def feet_to_meters(v):
    """
    Converts a distance in feet to meters.

    Args:
        v (float): distance in feet

    Returns:
        float: distance in meters
    """
    return v * FT2M


def kilometers_to_nautical_miles(v):
    """
    Converts a distance in kilometers to nautical miles.

    Args:
        v (float): distance in kilometers

    Returns:
        float: distance in nautical miles
    """
    return (v * KM2M) * M2NM


def nautical_miles_to_kilometers(v):
    """
    Converts a distance in nautical miles to kilometers.

    Args:
        v (float): distance in nautical miles

    Returns:
        float: distance in kilometers
    """
    return (v * NM2M) * M2KM


def kilometers_to_statute_miles(v):
    """
    Converts a distance in kilometers to statute miles.

    Args:
        v (float): distance in kilometers

    Returns:
        float: distance in statute miles
    """
    return (v * KM2M) * M2SM


def statute_miles_to_kilometers(v):
    """
    Converts a distance in statute miles to kilometers.

    Args:
        v (float): distance in statute miles

    Returns:
        float: distance in kilometers
    """
    return (v * SM2M) * M2KM


def nautical_miles_to_statute_miles(v):
    """
    Converts a distance in nautical miles to statute miles.

    Args:
        v (float): distance in nautical miles

    Returns:
        float: distance in statute miles
    """
    return (v * NM2M) * M2SM


def statute_miles_to_nautical_miles(v):
    """
    Converts a distance in statute miles to nautical miles.

    Args:
        v (float): distance in statute miles

    Returns:
        float: distance in nautical miles
    """
    return (v * SM2M) * M2NM


""""
Constants and Methods for Converting Angle Measurement Units
These constants and methods are equivalent to those of the file angle_utils.cpp of MIXR.
"""

# Constants for conversion between angles
D2SC = 0.0055555555555556  # Degrees => Semicircles
"""
Conversion factor for converting degrees to semicircles.
"""
SC2D = 180.0  # Semicircles => Degrees
"""
Conversion factor for converting semicircles to degrees.
"""
R2SC = 0.3183098861837906  # Radians => Semicircles
"""
Conversion factor for converting radians to semicircles.
"""
SC2R = math.pi  # Semicircles => Radians
"""
Conversion factor for converting semicircles to radians.
"""
R2DCC = 180.0 / math.pi  # Radians => Degrees
"""
Conversion factor for converting radians to degrees.
"""
D2RCC = math.pi / 180.0  # Degrees => Radians
"""
Conversion factor for converting degrees to radians.
"""

# Methods for conversion between angles
def degrees_to_radians(v):
    """
    Converts a value from degrees to radians.

    Args:
        v (float): The value in degrees to convert to radians.

    Returns:
        float: The value in radians.
    """
    return (v * D2SC) * SC2R


def degrees_to_semicircles(v):
    """
    Converts a value from degrees to semicircles.

    Args:
        v (float): The value in degrees to convert to semicircles.

    Returns:
        float: The value in semicircles.
    """
    return v * D2SC


def radians_to_degrees(v):
    """
    Converts a value from radians to degrees.

    Args:
        v (float): The value in radians to convert to degrees.

    Returns:
        float: The value in degrees.
    """
    return (R2SC * v) * SC2D


def radians_to_semicircles(v):
    """
    Converts a value from radians to semicircles.

    Args:
        v (float): The value in radians to convert to semicircles.

    Returns:
        float: The value in semicircles.
    """
    return v * R2SC


def semicircles_to_radians(v):
    """
    Converts a value from semicircles to radians.

    Args:
        v (float): The value in semicircles to convert to radians.

    Returns:
        float: The value in radians.
    """
    return v * SC2R


def semicircles_to_degrees(v):
    """
    Converts a value from semicircles to degrees.

    Args:
        v (float): The value in semicircles to convert to degrees.

    Returns:
        float: The value in degrees.
    """
    return v * SC2D


# Methods for operation between angles
# Angle end-point check, in degrees (keeps angles within the range: -180 <= x <= 180)
def aepcd_deg(x):
    """
    The method aepcd_deg keeps an angle within the range -180.0 to 180.0 as presented in the figure bellow.
    In the example of this figure, the angle of 225 degrees is converted to -135 degrees through aepcd_deg.
    
    Args:
        x (float): Angle in degrees.

    Returns:
        float: The angle in degrees adjusted to lie within the range -180.0 to 180.0.
    """
    y = 0.0

    if (x < -180.0) or (x > 180.0):
        y = math.fmod(x, 360.0)
        if y > 180.0:
            y = y - 360.0
        if y < -180.0:
            y = y + 360.0
        return y
    else:
        if x == -180.0:
            return 180.0
        else:
            return x


# Angle end-point check, in radians (keeps angles within the range: -180 <= x <= 180)
def aepcd_rad(x):
    """
    Keeps an angle within the range -pi to pi, equivalent to the function aepcd_deg but in radians.

    Args:
        x: float, the angle to be checked in radians.

    Returns:
        float, the angle within the range -pi to pi, with the same orientation as the original angle.
    """
    y = 0.0

    if (x < -math.pi) or (x > math.pi):
        y = math.fmod(x, (2.0 * math.pi))
        if y > math.pi:
            y = y - (2.0 * math.pi)
        if y < -math.pi:
            y = y + (2.0 * math.pi)
        return y
    else:
        if x == -math.pi:
            return math.pi
        else:
            return x


"""
Numerical Manipulation Methods
These methods are equivalent to those of the file math_utils.cpp of MIXR.
"""
# alimd -- limits the value of x to +/-limit.
def alimd(x, limit):
    """
    Limits the value of `x` to +/- `limit`.

    Args:
        x (float): The value to be limited.
        limit (float): The maximum absolute value allowed for `x`.

    Returns:
        float: The limited value of `x`. If `x` is greater than `limit`, returns `limit`.
               If `x` is less than negative `limit`, returns `-limit`. Otherwise, returns `x`.
    """

    if x > limit:
        return limit
    else:
        if x < -limit:
            return -limit
        else:
            return x


"""
Data from Different Models of the Earth
Data from 22 Earth surface models are stored in the array earthModelData. 
Each cell of this array corresponds to a type of model and has the following information: semi major axis (a), in meters; and flattening (f). 
"""

earth_model_data = [
    # wgs84 -> indexEarthModel = 0
    [6378137.0, 1.0 / 298.257223563],

    # airy -> indexEarthModel = 1
    [6377563.396, 1.0 / 299.3249646],

    # australianNational -> indexEarthModel = 2
    [6378160.0, 1.0 / 298.25],

    # bessel1841 -> indexEarthModel = 3
    [6377397.155, 1.0 / 299.1528128],

    # clark1866 -> indexEarthModel = 4
    [678206.4, 1.0 / 294.9786982],

    # clark1880 -> indexEarthModel = 5
    [6378249.145, 1.0 / 293.465],

    # everest -> indexEarthModel = 6
    [6377276.345, 1.0 / 300.8017],

    # fischer1960 -> indexEarthModel = 7
    [6378166.0, 1.0 / 298.3],

    # fischer1968 -> indexEarthModel = 8
    [6378150.0, 1.0 / 298.3],

    # grs1967 -> indexEarthModel = 9
    [6378160.0, 1.0 / 298.247167427],

    # grs1980 -> indexEarthModel = 10
    [6378137.0, 1.0 / 298.257222101],

    # helmert1906 -> indexEarthModel = 11
    [6378200.0, 1.0 / 298.3],

    # hough -> indexEarthModel = 12
    [6378270.0, 1.0 / 297.0],

    # international -> indexEarthModel = 13
    [6378388.0, 1.0 / 297.0],

    # kravosky -> indexEarthModel = 14
    [6378245.0, 1.0 / 298.3],

    # modAiry -> indexEarthModel = 15
    [6377340.189, 1.0 / 299.3249646],

    # modEverest -> indexEarthModel = 16
    [6377304.063, 1.0 / 300.8017],

    # modFischer -> indexEarthModel = 17
    [6378155.0, 1.0 / 298.3],

    # southAmerican1969 -> indexEarthModel = 18
    [6378160.0, 1.0 / 298.25],

    # wgs60 -> indexEarthModel = 19
    [6378165.0, 1.0 / 298.3],

    # wgs66 -> indexEarthModel = 20
    [6378145.0, 1.0 / 298.25],

    # wgs72 -> indexEarthModel = 21
    [6378135.0, 1.0 / 298.26]
]
"""
Data from Different Models of the Earth
Data from 22 Earth surface models are stored in the array earthModelData. 
Each cell of this array corresponds to a type of model and has the following information: semi major axis (a), in meters; and flattening (f). 
"""


def gbd2ll(slat, slon, brg, dist, index_earth_model):
    """
    This function computes the destination (target) point from starting (ref) point given distance and initial bearing. 

    This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

    Args:

        - latitude (slat) and longitude (slon) of the starting point; 
        
        - bearing (brg), in degrees, between the starting and destination points;
        
        - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points; and
        
        - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

    Returns:

        - latitude (dlat) and longitude (dlon) of the destination point. 
        

    .. note:: 
    
        possible values for indexEarthModel.

        wgs84 -> indexEarthModel = 0

        airy -> indexEarthModel = 1

        australianNational -> indexEarthModel = 2

        bessel1841 -> indexEarthModel = 3

        clark1866 -> indexEarthModel = 4

        clark1880 -> indexEarthModel = 5

        everest -> indexEarthModel = 6

        fischer1960 -> indexEarthModel = 7

        fischer1968 -> indexEarthModel = 8

        grs1967 -> indexEarthModel = 9

        grs1980 -> indexEarthModel = 10

        helmert1906 -> indexEarthModel = 11

        hough -> indexEarthModel = 12

        international -> indexEarthModel = 13

        kravosky -> indexEarthModel = 14

        modAiry -> indexEarthModel = 15

        modEverest -> indexEarthModel = 16

        modFischer -> indexEarthModel = 17

        southAmerican1969 -> indexEarthModel = 18

        wgs60 -> indexEarthModel = 19

        wgs66 -> indexEarthModel = 20
        
        wgs72 -> indexEarthModel = 21
    """

    # --------------------------------------------------------------------------------------------------------------
    # initialize earth model parameters
    # --------------------------------------------------------------------------------------------------------------
    if (index_earth_model < 0) or (index_earth_model > 21):
        # consider the default WGS-84
        index_earth_model = 0

    temp_data = earth_model_data[index_earth_model]

    # semi major axis (a), in meters
    a = temp_data[0]

    # flattening (f)
    f = temp_data[1]

    # semi minor axis (b), in meters
    b = a * (1.0 - f)

    # eccentricity squared (e2)
    e2 = f * (2.0 - f)

    eem_a = a * M2NM

    eem_e2 = e2

    # --------------------------------------------------------------------------------------------------------------
    # convert slat, slon and brg to radians
    # --------------------------------------------------------------------------------------------------------------
    slatr = slat * D2RCC
    slonr = slon * D2RCC
    psi = aepcd_deg(brg) * D2RCC

    # --------------------------------------------------------------------------------------------------------------
    # transform source point about zero longitude
    # --------------------------------------------------------------------------------------------------------------
    tslatr = slatr

    # --------------------------------------------------------------------------------------------------------------
    # calculate Gaussian radius of curvature at source lat
    # --------------------------------------------------------------------------------------------------------------
    grad = eem_a * (1.0 - ((eem_e2 / 2.0) * math.cos(2.0 * tslatr)))  # Gaussian radius

    # --------------------------------------------------------------------------------------------------------------
    # compute transformed destination lat/lon
    # --------------------------------------------------------------------------------------------------------------
    tdlatr = 0.0
    tdlonr = -slonr
    if dist <= 10000.0:
        x = math.cos(dist / grad) * math.sin(tslatr)
        y = math.sin(dist / grad) * math.cos(tslatr) * math.cos(psi)
        tdlatr = math.asin(x + y)

        x = math.cos(dist / grad) - math.sin(tslatr) * math.sin(tdlatr)
        y = math.cos(tslatr) * math.cos(tdlatr)

        z = 0.0
        if y != 0.0:
            z = x / y
        else:
            if x >= 0:
                z = 1.0
            else:
                z = -1.0
        z = alimd(z, 1.0)

        tdlonr = math.acos(z)

        if psi < 0.0:
            tdlonr = -tdlonr

    # --------------------------------------------------------------------------------------------------------------
    # retransform destination point
    # --------------------------------------------------------------------------------------------------------------
    dlatr = tdlatr
    dlonr = tdlonr + slonr

    # --------------------------------------------------------------------------------------------------------------
    # convert to degrees
    # --------------------------------------------------------------------------------------------------------------
    dlat0 = dlatr * R2DCC
    dlon0 = dlonr * R2DCC

    # --------------------------------------------------------------------------------------------------------------
    # apply ellipsoidal correction
    # --------------------------------------------------------------------------------------------------------------
    ellip = 0.00334 * math.pow(math.cos(tslatr), 2)
    dlat0 = dlat0 - ellip * (dlat0 - slat)
    dlon0 = dlon0 + ellip * (dlon0 - slon)

    # --------------------------------------------------------------------------------------------------------------
    # limit check for destination longitude
    # --------------------------------------------------------------------------------------------------------------
    if dlon0 > 180.0:
        dlon0 = dlon0 - 360.0
    else:
        if dlon0 < -180.0:
            dlon0 = dlon0 + 360.0

    # --------------------------------------------------------------------------------------------------------------
    # return to caller
    # --------------------------------------------------------------------------------------------------------------
    dlat = dlat0
    dlon = dlon0

    return dlat, dlon


def fbd2ll(slat, slon, brg, dist):
    """
    This function computes the destination (target) point from starting (ref) point given distance and initial bearing. 

    This method considers the flat-earth projection and a spherical earth radius of 'ERAD60'. This method is similar to the method of the file nav_utils.inl of MIXR.

    Args:

        - latitude (slat) and longitude (slon) of the starting point; 
        
        - bearing (brg), in degrees, between the starting and destination points; and
        
        - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.

    Returns:

        - latitude (dlat) and longitude (dlon) of the destination point.
    """
    ang = brg * D2RCC
    ew = math.sin(ang) * dist
    ns = math.cos(ang) * dist

    dlat = slat + (ns / 60.0)

    tlat = slat
    if (tlat > 89.0) or (tlat < -89.0):
        tlat = 89.0

    dlon = aepcd_deg(slon + (ew / (60.0 * math.cos(tlat * D2RCC))))

    return dlat, dlon


def gll2bd(slat, slon, dlat, dlon, index_earth_model):
    """
    This function computes the initial bearing and the distance from the starting point to the destination point.

    This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

    Args:

        - latitude (slat) and longitude (slon) of the starting point;
        
        - latitude (dlat) and longitude (dlon) of the destination point; and
        
        - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

    Returns:

        - bearing (brg), in degrees, between the starting and destination points; and
        
        - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points. 
        

    .. note:: 
    
        possible values for indexEarthModel.

        wgs84 -> indexEarthModel = 0

        airy -> indexEarthModel = 1

        australianNational -> indexEarthModel = 2

        bessel1841 -> indexEarthModel = 3

        clark1866 -> indexEarthModel = 4

        clark1880 -> indexEarthModel = 5

        everest -> indexEarthModel = 6

        fischer1960 -> indexEarthModel = 7

        fischer1968 -> indexEarthModel = 8

        grs1967 -> indexEarthModel = 9

        grs1980 -> indexEarthModel = 10

        helmert1906 -> indexEarthModel = 11

        hough -> indexEarthModel = 12

        international -> indexEarthModel = 13

        kravosky -> indexEarthModel = 14

        modAiry -> indexEarthModel = 15

        modEverest -> indexEarthModel = 16

        modFischer -> indexEarthModel = 17

        southAmerican1969 -> indexEarthModel = 18

        wgs60 -> indexEarthModel = 19

        wgs66 -> indexEarthModel = 20

        wgs72 -> indexEarthModel = 21
    """
    # --------------------------------------------------------------------------------------------------------------
    # initialize earth model parameters
    # --------------------------------------------------------------------------------------------------------------
    if (index_earth_model < 0) or (index_earth_model > 21):
        # consider the default WGS-84
        index_earth_model = 0

    temp_data = earth_model_data[index_earth_model]

    # semi major axis (a), in meters
    a = temp_data[0]

    # flattening (f)
    f = temp_data[1]

    # semi minor axis (b), in meters
    b = a * (1.0 - f)

    # eccentricity squared (e2)
    e2 = f * (2.0 - f)

    eem_a = a * M2NM

    eem_e2 = e2

    # --------------------------------------------------------------------------------------------------------------
    # check for source and destination at same point
    # --------------------------------------------------------------------------------------------------------------
    if (dlat == slat) and (dlon == slon):
        dist = 0.0
        brg = 0.0
    else:
        # --------------------------------------------------------------------------------------------------------------
        # transform destination lat/lon into the equivalent spherical lat/lon
        # --------------------------------------------------------------------------------------------------------------
        # Ellipsoidal correction factor
        ellip = 0.00334 * math.pow(math.cos(slat * D2RCC), 2)

        dlat0 = aepcd_deg(dlat + ellip * aepcd_deg(dlat - slat))
        dlon0 = aepcd_deg(dlon - ellip * aepcd_deg(dlon - slon))

        # --------------------------------------------------------------------------------------------------------------
        # transform lat/lon about zero longitude
        # --------------------------------------------------------------------------------------------------------------
        tslat = slat  # Transformed source lat (deg)
        tdlat = dlat0  # Transformed destination lat (deg)
        tdlon = dlon0 - slon  # Transformed destination lon (deg)
        if tdlon < -180.0:
            tdlon = tdlon + 360.0
        else:
            if tdlon > 180.0:
                tdlon = tdlon - 360.0

        # --------------------------------------------------------------------------------------------------------------
        # convert lat/lon to radians
        # --------------------------------------------------------------------------------------------------------------
        tslatr = tslat * D2RCC  # Transformed source lat (rad)
        tdlatr = tdlat * D2RCC  # Transformed destination lat (rad)
        tdlonr = tdlon * D2RCC  # Transformed destination lon (rad)

        # --------------------------------------------------------------------------------------------------------------
        # calculate Gaussian radius of curvature at source lat
        # --------------------------------------------------------------------------------------------------------------
        grad = eem_a * (1.0 - ((eem_e2 / 2.0) * math.cos(2.0 * tslatr)))  # Gaussian radius

        # --------------------------------------------------------------------------------------------------------------
        # compute great circle distance
        # --------------------------------------------------------------------------------------------------------------
        tzlonr = tdlonr  # Lon deviation(rad)
        x = math.sin(tslatr) * math.sin(tdlatr)
        y = math.cos(tslatr) * math.cos(tdlatr) * math.cos(tzlonr)
        z = x + y
        z = alimd(z, 1.0)

        dist = grad * math.fabs(math.acos(z))
        if dist == 0.0:
            brg = 0.0
        else:
            # --------------------------------------------------------------------------------------------------------------
            # compute great circle bearing
            # --------------------------------------------------------------------------------------------------------------
            x = math.sin(tdlatr) - math.sin(tslatr) * math.cos(dist / grad)
            y = math.sin(dist / grad) * math.cos(tslatr)
            if y != 0.0:
                z = x / y
            else:
                if x >= 0:
                    z = 1.0
                else:
                    z = -1.0
            z = alimd(z, 1.0)

            x = math.acos(z) * R2DCC
            if tzlonr < 0.0:
                x = 360.0 - x
            brg = aepcd_deg(x)

    return brg, dist


def fll2bd(slat, slon, dlat, dlon):
    """
    This function computes the initial bearing and the distance from the starting point to the destination point.

    This method considers a flat earth projection and a spherical earth radius of 'ERAD60'.

    Args:

        - latitude (slat) and longitude (slon) of the starting point; and
        
        - latitude (dlat) and longitude (dlon) of the destination point.

    Returns:

        - bearing (brg), in degrees, between the starting and destination points; and
        
        - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points. 
    """
    ns = aepcd_deg(dlat - slat) * 60.0
    ew = aepcd_deg(dlon - slon) * 60.0 * math.cos(slat * D2RCC)
    brg = math.atan2(ew, ns) * R2DCC
    dist = math.sqrt(ns * ns + ew * ew)

    return brg, dist


def convert_ecef_to_geod(x, y, z, index_earth_model):
    """
    This function converts Earth Centered, Earth Fixed (ECEF) coordinates (x,y,z) to geodetic coordinates (latitude,longitude,altitude).

    Args:

        - ECEF coordinates (x,y,z), in meters; and
        
        - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

    Returns:

        - geodetic coordinates (lat, lon, alt), considering lat and lon in degrees, and alt in meters.


    .. note::  
        
        possible values for indexEarthModel.

        wgs84 -> indexEarthModel = 0

        airy -> indexEarthModel = 1

        australianNational -> indexEarthModel = 2

        bessel1841 -> indexEarthModel = 3

        clark1866 -> indexEarthModel = 4

        clark1880 -> indexEarthModel = 5

        everest -> indexEarthModel = 6

        fischer1960 -> indexEarthModel = 7

        fischer1968 -> indexEarthModel = 8

        grs1967 -> indexEarthModel = 9

        grs1980 -> indexEarthModel = 10

        helmert1906 -> indexEarthModel = 11

        hough -> indexEarthModel = 12

        international -> indexEarthModel = 13

        kravosky -> indexEarthModel = 14

        modAiry -> indexEarthModel = 15

        modEverest -> indexEarthModel = 16

        modFischer -> indexEarthModel = 17

        southAmerican1969 -> indexEarthModel = 18

        wgs60 -> indexEarthModel = 19

        wgs66 -> indexEarthModel = 20

        wgs72 -> indexEarthModel = 21

    """
    # --------------------------------------------------------------------------------------------------------------
    # initialize earth model parameters
    # --------------------------------------------------------------------------------------------------------------
    if (index_earth_model < 0) or (index_earth_model > 21):
        # consider the default WGS-84
        index_earth_model = 0

    temp_data = earth_model_data[index_earth_model]

    # semi major axis (a), in meters
    a = temp_data[0]

    # flattening (f)
    f = temp_data[1]

    # semi minor axis (b), in meters
    b = a * (1.0 - f)

    # eccentricity squared (e2)
    e2 = f * (2.0 - f)

    # --------------------------------------------------------------------------------------------------------------
    # initialize pLat, pLon and pAlt
    # --------------------------------------------------------------------------------------------------------------
    lat = 0.0
    lon = 0.0
    alt = 0.0

    # --------------------------------------------------------------------------------------------------------------
    # define local constants
    # --------------------------------------------------------------------------------------------------------------
    p = math.sqrt(x * x + y * y)
    ACCURACY = 0.1  # iterate to accuracy of 0.1 meter
    EPS = 1.0E-10
    MAX_LOOPS = 10

    # --------------------------------------------------------------------------------------------------------------
    # initialize local variables
    # --------------------------------------------------------------------------------------------------------------
    status = "NORMAL"
    rn = a
    phi = 0.0
    old_h = 0.0
    new_h = 100.0 * ACCURACY  # (new_h - old_h) significantly different
    idx = 0

    # --------------------------------------------------------------------------------------------------------------
    # check status
    # --------------------------------------------------------------------------------------------------------------
    polar_xy = math.fabs(x) + math.fabs(y)
    if polar_xy < EPS:
        status = "POLAR_POINT"

    # --------------------------------------------------------------------------------------------------------------
    # iterate for accurate latitude and altitude
    # --------------------------------------------------------------------------------------------------------------
    if status == "NORMAL":
        idx = 1
        while (idx <= MAX_LOOPS) and (math.fabs(new_h - old_h) > ACCURACY):
            sin_phi = z / (new_h + rn * (1.0 - e2))
            q = z + e2 * rn * sin_phi
            phi = math.atan2(q, p)
            cos_phi = math.cos(phi)
            w = math.sqrt(1.0 - e2 * sin_phi * sin_phi)
            rn = a / w
            old_h = new_h
            new_h = p / cos_phi - rn
            idx = idx + 1

    # --------------------------------------------------------------------------------------------------------------
    # re-check status after iteration
    # --------------------------------------------------------------------------------------------------------------
    if idx > MAX_LOOPS:
        status = "TOO_MANY_LOOPS"

    # --------------------------------------------------------------------------------------------------------------
    # process based on status
    # --------------------------------------------------------------------------------------------------------------
    if status == "NORMAL":
        # begin iteration loop

        # calculate outputs
        lat = R2DCC * phi
        lon = R2DCC * math.atan2(y, x)
        alt = new_h
    else:
        if status == "POLAR_POINT":
            if z < 0.0:
                lat = -90.0
                lon = 0.0
                alt = -b - z
            else:
                lat = 90.0
                lon = 0.0
                alt = -b + z

    return lat, lon, alt


def convert_geod_to_ecef(lat, lon, alt, index_earth_model):
    """
    This function converts Geodetic ((Latitude,Longitude,Altitude) coordinates) to ECEF ((X,Y,Z) coordinates).

    Args:

        - geodetic coordinates (lat, lon, alt), considering lat and lon in degrees, and alt in meters; and
        
        - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

    Returns:

        - ECEF coordinates (x,y,z), in meters.

    
    .. note::
     
        possible values for indexEarthModel.

        wgs84 -> indexEarthModel = 0

        airy -> indexEarthModel = 1

        australianNational -> indexEarthModel = 2

        bessel1841 -> indexEarthModel = 3

        clark1866 -> indexEarthModel = 4

        clark1880 -> indexEarthModel = 5

        everest -> indexEarthModel = 6

        fischer1960 -> indexEarthModel = 7

        fischer1968 -> indexEarthModel = 8

        grs1967 -> indexEarthModel = 9

        grs1980 -> indexEarthModel = 10

        helmert1906 -> indexEarthModel = 11

        hough -> indexEarthModel = 12

        international -> indexEarthModel = 13

        kravosky -> indexEarthModel = 14

        modAiry -> indexEarthModel = 15

        modEverest -> indexEarthModel = 16

        modFischer -> indexEarthModel = 17

        southAmerican1969 -> indexEarthModel = 18

        wgs60 -> indexEarthModel = 19

        wgs66 -> indexEarthModel = 20
        
        wgs72 -> indexEarthModel = 21
    """
    # --------------------------------------------------------------------------------------------------------------
    # initialize earth model parameters
    # --------------------------------------------------------------------------------------------------------------
    if (index_earth_model < 0) or (index_earth_model > 21):
        # consider the default WGS-84
        index_earth_model = 0

    temp_data = earth_model_data[index_earth_model]

    # semi major axis (a), in meters
    a = temp_data[0]

    # flattening (f)
    f = temp_data[1]

    # semi minor axis (b), in meters
    b = a * (1.0 - f)

    # eccentricity squared (e2)
    e2 = f * (2.0 - f)

    # --------------------------------------------------------------------------------------------------------------
    # define local constants
    # --------------------------------------------------------------------------------------------------------------
    EPS = 0.5  # degrees
    sin_lat = math.sin(D2RCC * lat)
    cos_lat = math.cos(D2RCC * lat)
    sin_lon = math.sin(D2RCC * lon)
    cos_lon = math.cos(D2RCC * lon)
    w = math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    rn = a / w

    # --------------------------------------------------------------------------------------------------------------
    # initialize local variables
    # --------------------------------------------------------------------------------------------------------------
    status = "NORMAL"

    # --------------------------------------------------------------------------------------------------------------
    # check status
    # --------------------------------------------------------------------------------------------------------------
    b1 = (lat < -90.0) or (lat > +90.0)
    b2 = (lon < -180.0) or (lon > +180.0)
    b3 = ((90.0 - lat) < EPS)
    b4 = ((90.0 + lat) < EPS)

    if b1 or b2:
        status = "BAD_INPUT"
    else:
        if b3 or b4:
            status = "POLAR_POINT"
        else:
            status = "NORMAL"

    # --------------------------------------------------------------------------------------------------------------
    # process according to status
    # --------------------------------------------------------------------------------------------------------------
    if status == "NORMAL":
        x = (alt + rn) * cos_lat * cos_lon
        y = (alt + rn) * cos_lat * sin_lon
        z = (alt + rn * (1.0 - e2)) * sin_lat
    else:
        if status == "BAD_INPUT":
            x = 0.0
            y = 0.0
            z = 0.0
        else:
            if status == "POLAR_POINT":
                x = 0.0
                y = 0.0
                if lat > 0.0:
                    z = (b + alt)
                else:
                    z = -1.0 * (b + alt)

    return x, y, z
