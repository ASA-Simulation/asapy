:py:mod:`asapy.utils`
=====================

.. py:module:: asapy.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   asapy.utils.prepare_simulation_batch
   asapy.utils.prepare_simulation_tacview
   asapy.utils.load_simulation
   asapy.utils.json_to_df
   asapy.utils.list_to_df
   asapy.utils.unique_list
   asapy.utils.get_parents_dict
   asapy.utils.check_samples_similar
   asapy.utils.test_t
   asapy.utils.convert_nested_string_to_dict
   asapy.utils.find_key
   asapy.utils.gen_dict_extract
   asapy.utils.transform_stringified_dict
   asapy.utils.meters_to_micrometers
   asapy.utils.micrometers_to_meters
   asapy.utils.meters_to_centimeters
   asapy.utils.centimeters_to_meters
   asapy.utils.meters_to_kilometers
   asapy.utils.kilometers_to_meters
   asapy.utils.meters_to_inches
   asapy.utils.inches_to_meters
   asapy.utils.meters_to_feet
   asapy.utils.feet_to_meters
   asapy.utils.kilometers_to_nautical_miles
   asapy.utils.nautical_miles_to_kilometers
   asapy.utils.kilometers_to_statute_miles
   asapy.utils.statute_miles_to_kilometers
   asapy.utils.nautical_miles_to_statute_miles
   asapy.utils.statute_miles_to_nautical_miles
   asapy.utils.degrees_to_radians
   asapy.utils.degrees_to_semicircles
   asapy.utils.radians_to_degrees
   asapy.utils.radians_to_semicircles
   asapy.utils.semicircles_to_radians
   asapy.utils.semicircles_to_degrees
   asapy.utils.aepcd_deg
   asapy.utils.aepcd_rad
   asapy.utils.alimd
   asapy.utils.gbd2ll
   asapy.utils.fbd2ll
   asapy.utils.gll2bd
   asapy.utils.fll2bd
   asapy.utils.convert_ecef_to_geod
   asapy.utils.convert_geod_to_ecef



Attributes
~~~~~~~~~~

.. autoapisummary::

   asapy.utils.FT2M
   asapy.utils.M2FT
   asapy.utils.IN2M
   asapy.utils.M2IN
   asapy.utils.NM2M
   asapy.utils.M2NM
   asapy.utils.NM2FT
   asapy.utils.FT2NM
   asapy.utils.SM2M
   asapy.utils.M2SM
   asapy.utils.SM2FT
   asapy.utils.FT2SM
   asapy.utils.KM2M
   asapy.utils.M2KM
   asapy.utils.CM2M
   asapy.utils.M2CM
   asapy.utils.UM2M
   asapy.utils.M2UM
   asapy.utils.D2SC
   asapy.utils.SC2D
   asapy.utils.R2SC
   asapy.utils.SC2R
   asapy.utils.R2DCC
   asapy.utils.D2RCC
   asapy.utils.earth_model_data


.. py:function:: prepare_simulation_batch(sim: asaclient.Simulation) -> asaclient.Simulation

   Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run in batch mode.

   :param sim: The simulation instance for which the simulation setup needs to be prepared.
   :type sim: Simulation

   :returns: The updated Simulation instance with the added recorder configurations in its subcomponents.
   :rtype: Simulation


.. py:function:: prepare_simulation_tacview(sim: asaclient.Simulation) -> asaclient.Simulation

   Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run on Tacview.

   :param sim: The simulation instance for which the simulation setup needs to be prepared.
   :type sim: Simulation

   :returns: The updated Simulation instance with the added recorder configurations in its subcomponents.
   :rtype: Simulation


.. py:function:: load_simulation(path: str) -> asaclient.Simulation

   Loads a Simulation object from a JSON file.

   This method accepts a path to a JSON file, reads the content of the file and
   creates a Simulation object using the data parsed from the file.

   :param path: The absolute or relative path to the JSON file to be loaded.
   :type path: str

   :returns: The Simulation object created from the loaded JSON data.
   :rtype: Simulation


.. py:function:: json_to_df(self, json, id='id') -> pandas.DataFrame

   Convert a JSON object to a pandas DataFrame and set the index to the given id column.

   :param json: A JSON object.
   :type json: dict
   :param id: The name of the column to set as the index. Default is 'id'.
   :type id: str

   :returns: A DataFrame representation of the JSON object.
   :rtype: pandas.DataFrame


.. py:function:: list_to_df(arr, id='id')

   Convert a list of dictionaries to a pandas DataFrame and set the index to the given id column.

   :param arr: A list of dictionaries.
   :type arr: list
   :param id: The name of the column to set as the index. Default is 'id'.
   :type id: str

   :returns: A DataFrame representation of the list of dictionaries.
   :rtype: pandas.DataFrame


.. py:function:: unique_list(list1)

   Return a list of unique values in the given list.

   :param list1: A list of values.
   :type list1: list

   :returns: A list of unique values in the input list.
   :rtype: list


.. py:function:: get_parents_dict(dic, value)

   Return a list of keys that lead to the given value in the given dictionary.

   :param dic: A dictionary to search.
   :type dic: dict
   :param value: The value to search for in the dictionary.

   :returns: A list of keys that lead to the given value in the dictionary.
   :rtype: list


.. py:function:: check_samples_similar(new_sample, last_sample, threshold)

   Checks if two samples are similar based on a given threshold.

   :param new_sample: The new sample to compare.
   :type new_sample: np.ndarray
   :param last_sample: The last sample to compare.
   :type last_sample: np.ndarray
   :param threshold: The threshold to use for comparison.
   :type threshold: float

   :returns: True if the samples are similar, False otherwise.
   :rtype: bool


.. py:function:: test_t(sample1, sample2, alpha=0.05)

   Performs a t-test and compares the p-value with a given alpha value.

   :param sample1: The first sample.
   :type sample1: np.ndarray
   :param sample2: The second sample.
   :type sample2: np.ndarray
   :param alpha: The alpha value to use for comparison. Defaults to 0.05.
   :type alpha: float, optional

   :returns: True if the samples are similar, False otherwise.
   :rtype: bool


.. py:function:: convert_nested_string_to_dict(s)

   Converts a string that contains a dictionary and JSON-formatted strings into a nested dictionary.

   :param s: The input string containing a dictionary and JSON-formatted strings.
   :type s: str

   :returns: The output dictionary after conversion of JSON-formatted strings.
   :rtype: dict


.. py:function:: find_key(nested_dict, target_key)

   Find a key in a nested dictionary.

   :param nested_dict: The dictionary to search.
   :type nested_dict: dict
   :param target_key: The key to find.
   :type target_key: str

   :returns: The value of the found key, or None if the key was not found.
   :rtype: value


.. py:function:: gen_dict_extract(key, var)

   A generator function to iterate and yield values from a dictionary or list nested inside the dictionary, given a key.

   :param key: The key to search for in the dictionary.
   :type key: str
   :param var: The dictionary or list to search.
   :type var: dict or list

   :Yields: *value* -- The value from the dictionary or list that corresponds to the given key.


.. py:function:: transform_stringified_dict(data)

   Recursively converts stringified JSON parts of a dictionary or list into actual dictionaries or lists.

   This function checks if an item is a string and attempts to convert it to a dictionary or list
   using `json.loads()`. If the conversion is successful, the function recursively processes the new
   dictionary or list. If a string is not a valid JSON representation, it remains unchanged.

   :param data: Input data that might contain stringified JSON parts.
   :type data: Union[dict, list, str]

   :returns: The transformed data with all stringified JSON parts converted
             to dictionaries or lists.
   :rtype: Union[dict, list, str]

   :raises json.JSONDecodeError: If there's an issue decoding a JSON string. This is caught internally
   :raises and the original string is returned.:


.. py:data:: FT2M
   :value: 0.3048

   Conversion factor from feet to meters.

.. py:data:: M2FT

   Conversion factor from meters to feet.

.. py:data:: IN2M
   :value: 0.0254

   Conversion factor from inches to meters.

.. py:data:: M2IN

   Conversion factor from meters to inches.

.. py:data:: NM2M
   :value: 1852.0

   Conversion factor from nautical miles to meters.

.. py:data:: M2NM

   Conversion factor from meters to nautical miles.

.. py:data:: NM2FT

   Conversion factor from nautical miles to feet.

.. py:data:: FT2NM

   Conversion factor from feet to nautical miles.

.. py:data:: SM2M
   :value: 1609.344

   Conversion factor from statute miles to meters.

.. py:data:: M2SM

   Conversion factor from meters to statute miles.

.. py:data:: SM2FT
   :value: 5280.0

   Conversion factor from statute miles to feet.

.. py:data:: FT2SM

   Conversion factor from feet to statute miles.

.. py:data:: KM2M
   :value: 1000.0

   Conversion factor from kilometers to meters.

.. py:data:: M2KM

   Conversion factor from meters to kilometers.

.. py:data:: CM2M
   :value: 0.01

   Conversion factor from centimeters to meters.

.. py:data:: M2CM

   Conversion factor from meters to centimeters.

.. py:data:: UM2M
   :value: 1e-06

   Conversion factor from micrometers to meters.

.. py:data:: M2UM

   Conversion factor from meters to micrometers.

.. py:function:: meters_to_micrometers(v)

   Convert meters to micrometers.

   :param v: Value in meters.
   :type v: float

   :returns: Value in micrometers.
   :rtype: float


.. py:function:: micrometers_to_meters(v)

   Convert micrometers to meters.

   :param v: Value in micrometers.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_centimeters(v)

   Convert meters to centimeters.

   :param v: Value in meters.
   :type v: float

   :returns: Value in centimeters.
   :rtype: float


.. py:function:: centimeters_to_meters(v)

   Convert centimeters to meters.

   :param v: Value in centimeters.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_kilometers(v)

   Convert meters to kilometers.

   :param v: Value in meters.
   :type v: float

   :returns: Value in kilometers.
   :rtype: float


.. py:function:: kilometers_to_meters(v)

   Convert kilometers to meters.

   :param v: Value in kilometers.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_inches(v)

   Convert meters to inches.

   :param v: Value in meters.
   :type v: float

   :returns: Value in inches.
   :rtype: float


.. py:function:: inches_to_meters(v)

   Convert inches to meters.

   :param v: Value in inches.
   :type v: float

   :returns: Value
   :rtype: float


.. py:function:: meters_to_feet(v)

   Converts a distance in meters to feet.

   :param v: distance in meters
   :type v: float

   :returns: distance in feet
   :rtype: float


.. py:function:: feet_to_meters(v)

   Converts a distance in feet to meters.

   :param v: distance in feet
   :type v: float

   :returns: distance in meters
   :rtype: float


.. py:function:: kilometers_to_nautical_miles(v)

   Converts a distance in kilometers to nautical miles.

   :param v: distance in kilometers
   :type v: float

   :returns: distance in nautical miles
   :rtype: float


.. py:function:: nautical_miles_to_kilometers(v)

   Converts a distance in nautical miles to kilometers.

   :param v: distance in nautical miles
   :type v: float

   :returns: distance in kilometers
   :rtype: float


.. py:function:: kilometers_to_statute_miles(v)

   Converts a distance in kilometers to statute miles.

   :param v: distance in kilometers
   :type v: float

   :returns: distance in statute miles
   :rtype: float


.. py:function:: statute_miles_to_kilometers(v)

   Converts a distance in statute miles to kilometers.

   :param v: distance in statute miles
   :type v: float

   :returns: distance in kilometers
   :rtype: float


.. py:function:: nautical_miles_to_statute_miles(v)

   Converts a distance in nautical miles to statute miles.

   :param v: distance in nautical miles
   :type v: float

   :returns: distance in statute miles
   :rtype: float


.. py:function:: statute_miles_to_nautical_miles(v)

   Converts a distance in statute miles to nautical miles.

   :param v: distance in statute miles
   :type v: float

   :returns: distance in nautical miles
   :rtype: float


.. py:data:: D2SC
   :value: 0.0055555555555556

   Conversion factor for converting degrees to semicircles.

.. py:data:: SC2D
   :value: 180.0

   Conversion factor for converting semicircles to degrees.

.. py:data:: R2SC
   :value: 0.3183098861837906

   Conversion factor for converting radians to semicircles.

.. py:data:: SC2R

   Conversion factor for converting semicircles to radians.

.. py:data:: R2DCC

   Conversion factor for converting radians to degrees.

.. py:data:: D2RCC

   Conversion factor for converting degrees to radians.

.. py:function:: degrees_to_radians(v)

   Converts a value from degrees to radians.

   :param v: The value in degrees to convert to radians.
   :type v: float

   :returns: The value in radians.
   :rtype: float


.. py:function:: degrees_to_semicircles(v)

   Converts a value from degrees to semicircles.

   :param v: The value in degrees to convert to semicircles.
   :type v: float

   :returns: The value in semicircles.
   :rtype: float


.. py:function:: radians_to_degrees(v)

   Converts a value from radians to degrees.

   :param v: The value in radians to convert to degrees.
   :type v: float

   :returns: The value in degrees.
   :rtype: float


.. py:function:: radians_to_semicircles(v)

   Converts a value from radians to semicircles.

   :param v: The value in radians to convert to semicircles.
   :type v: float

   :returns: The value in semicircles.
   :rtype: float


.. py:function:: semicircles_to_radians(v)

   Converts a value from semicircles to radians.

   :param v: The value in semicircles to convert to radians.
   :type v: float

   :returns: The value in radians.
   :rtype: float


.. py:function:: semicircles_to_degrees(v)

   Converts a value from semicircles to degrees.

   :param v: The value in semicircles to convert to degrees.
   :type v: float

   :returns: The value in degrees.
   :rtype: float


.. py:function:: aepcd_deg(x)

   The method aepcd_deg keeps an angle within the range -180.0 to 180.0 as presented in the figure bellow.
   In the example of this figure, the angle of 225 degrees is converted to -135 degrees through aepcd_deg.

   :param x: Angle in degrees.
   :type x: float

   :returns: The angle in degrees adjusted to lie within the range -180.0 to 180.0.
   :rtype: float


.. py:function:: aepcd_rad(x)

   Keeps an angle within the range -pi to pi, equivalent to the function aepcd_deg but in radians.

   :param x: float, the angle to be checked in radians.

   :returns: float, the angle within the range -pi to pi, with the same orientation as the original angle.


.. py:function:: alimd(x, limit)

   Limits the value of `x` to +/- `limit`.

   :param x: The value to be limited.
   :type x: float
   :param limit: The maximum absolute value allowed for `x`.
   :type limit: float

   :returns:

             The limited value of `x`. If `x` is greater than `limit`, returns `limit`.
                    If `x` is less than negative `limit`, returns `-limit`. Otherwise, returns `x`.
   :rtype: float


.. py:data:: earth_model_data
   :value: [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,...

   Data from Different Models of the Earth
   Data from 22 Earth surface models are stored in the array earthModelData.
   Each cell of this array corresponds to a type of model and has the following information: semi major axis (a), in meters; and flattening (f).

.. py:function:: gbd2ll(slat, slon, brg, dist, index_earth_model)

   This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

   This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - bearing:
   :type - bearing: brg
   :param - distance:
   :type - distance: dist) or ground range, in nautical miles (nm
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

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


.. py:function:: fbd2ll(slat, slon, brg, dist)

   This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

   This method considers the flat-earth projection and a spherical earth radius of 'ERAD60'. This method is similar to the method of the file nav_utils.inl of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - bearing:
   :type - bearing: brg
   :param - distance:
   :type - distance: dist) or ground range, in nautical miles (nm

   :returns:

             - latitude (dlat) and longitude (dlon) of the destination point.


.. py:function:: gll2bd(slat, slon, dlat, dlon, index_earth_model)

   This function computes the initial bearing and the distance from the starting point to the destination point.

   This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - latitude:
   :type - latitude: dlat) and longitude (dlon
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

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


.. py:function:: fll2bd(slat, slon, dlat, dlon)

   This function computes the initial bearing and the distance from the starting point to the destination point.

   This method considers a flat earth projection and a spherical earth radius of 'ERAD60'.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - latitude:
   :type - latitude: dlat) and longitude (dlon

   :returns:

             - bearing (brg), in degrees, between the starting and destination points; and

             - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.


.. py:function:: convert_ecef_to_geod(x, y, z, index_earth_model)

   This function converts Earth Centered, Earth Fixed (ECEF) coordinates (x,y,z) to geodetic coordinates (latitude,longitude,altitude).

   :param - ECEF coordinates:
   :type - ECEF coordinates: x,y,z
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

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



.. py:function:: convert_geod_to_ecef(lat, lon, alt, index_earth_model)

   This function converts Geodetic ((Latitude,Longitude,Altitude) coordinates) to ECEF ((X,Y,Z) coordinates).

   :param - geodetic coordinates:
   :type - geodetic coordinates: lat, lon, alt
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

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


