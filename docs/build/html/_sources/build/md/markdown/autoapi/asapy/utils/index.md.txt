# `asapy.utils`

## Module Contents

### Functions

| `json_to_df`(→ pandas.DataFrame)

 | Convert a JSON object to a pandas DataFrame and set the index to the given id column.

 |
| `list_to_df`(arr[, id])

          | Convert a list of dictionaries to a pandas DataFrame and set the index to the given id column.

                                                                                                                        |
| `unique_list`(list1)

             | Return a list of unique values in the given list.

                                                                                                                                                                     |
| `get_parents_dict`(dic, value)

   | Return a list of keys that lead to the given value in the given dictionary.

                                                                                                                                           |
| `check_samples_similar`(new_sample, last_sample, threshold)

 | Checks if two samples are similar based on a given threshold.

                                                                                                                                                         |
| `test_t`(sample1, sample2[, alpha])

                         | Performs a t-test and compares the p-value with a given alpha value.

                                                                                                                                                  |
| `meters_to_micrometers`(v)

                                  | Convert meters to micrometers.

                                                                                                                                                                                        |
| `micrometers_to_meters`(v)

                                  | Convert micrometers to meters.

                                                                                                                                                                                        |
| `meters_to_centimeters`(v)

                                  | Convert meters to centimeters.

                                                                                                                                                                                        |
| `centimeters_to_meters`(v)

                                  | Convert centimeters to meters.

                                                                                                                                                                                        |
| `meters_to_kilometers`(v)

                                   | Convert meters to kilometers.

                                                                                                                                                                                         |
| `kilometers_to_meters`(v)

                                   | Convert kilometers to meters.

                                                                                                                                                                                         |
| `meters_to_inches`(v)

                                       | Convert meters to inches.

                                                                                                                                                                                             |
| `inches_to_meters`(v)

                                       | Convert inches to meters.

                                                                                                                                                                                             |
| `meters_to_feet`(v)

                                         | Converts a distance in meters to feet.

                                                                                                                                                                                |
| `feet_to_meters`(v)

                                         | Converts a distance in feet to meters.

                                                                                                                                                                                |
| `kilometers_to_nautical_miles`(v)

                           | Converts a distance in kilometers to nautical miles.

                                                                                                                                                                  |
| `nautical_miles_to_kilometers`(v)

                           | Converts a distance in nautical miles to kilometers.

                                                                                                                                                                  |
| `kilometers_to_statute_miles`(v)

                            | Converts a distance in kilometers to statute miles.

                                                                                                                                                                   |
| `statute_miles_to_kilometers`(v)

                            | Converts a distance in statute miles to kilometers.

                                                                                                                                                                   |
| `nautical_miles_to_statute_miles`(v)

                        | Converts a distance in nautical miles to statute miles.

                                                                                                                                                               |
| `statute_miles_to_nautical_miles`(v)

                        | Converts a distance in statute miles to nautical miles.

                                                                                                                                                               |
| `degrees_to_radians`(v)

                                     | Converts a value from degrees to radians.

                                                                                                                                                                             |
| `degrees_to_semicircles`(v)

                                 | Converts a value from degrees to semicircles.

                                                                                                                                                                         |
| `radians_to_degrees`(v)

                                     | Converts a value from radians to degrees.

                                                                                                                                                                             |
| `radians_to_semicircles`(v)

                                 | Converts a value from radians to semicircles.

                                                                                                                                                                         |
| `semicircles_to_radians`(v)

                                 | Converts a value from semicircles to radians.

                                                                                                                                                                         |
| `semicircles_to_degrees`(v)

                                 | Converts a value from semicircles to degrees.

                                                                                                                                                                         |
| `aepcd_deg`(x)

                                              | The method aepcd_deg keeps an angle within the range -180.0 to 180.0 as presented in the figure bellow.

                                                                                                               |
| `aepcd_rad`(x)

                                              | Keeps an angle within the range -pi to pi, equivalent to the function aepcd_deg but in radians.

                                                                                                                       |
| `alimd`(x, limit)

                                           | Limits the value of x to +/- limit.

                                                                                                                                                                                   |
| `gbd2ll`(slat, slon, brg, dist, index_earth_model)

          | This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

                                                                                                   |
| `fbd2ll`(slat, slon, brg, dist)

                             | This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

                                                                                                   |
| `gll2bd`(slat, slon, dlat, dlon, index_earth_model)

         | This function computes the initial bearing and the distance from the starting point to the destination point.

                                                                                                         |
| `fll2bd`(slat, slon, dlat, dlon)

                            | This function computes the initial bearing and the distance from the starting point to the destination point.

                                                                                                         |
| `convert_ecef_to_geod`(x, y, z, index_earth_model)

          | This function converts Earth Centered, Earth Fixed (ECEF) coordinates (x,y,z) to geodetic coordinates (latitude,longitude,altitude).

                                                                                  |
| `convert_geod_to_ecef`(lat, lon, alt, index_earth_model)

    | This function converts Geodetic ((Latitude,Longitude,Altitude) coordinates) to ECEF ((X,Y,Z) coordinates).

                                                                                                            |
### Attributes

| `FT2M`
                                                      | Conversion factor from feet to meters.

                                                                                                                                                                                |
| `M2FT`
                                                      | Conversion factor from meters to feet.

                                                                                                                                                                                |
| `IN2M`
                                                      | Conversion factor from inches to meters.

                                                                                                                                                                              |
| `M2IN`
                                                      | Conversion factor from meters to inches.

                                                                                                                                                                              |
| `NM2M`
                                                      | Conversion factor from nautical miles to meters.

                                                                                                                                                                      |
| `M2NM`
                                                      | Conversion factor from meters to nautical miles.

                                                                                                                                                                      |
| `NM2FT`
                                                     | Conversion factor from nautical miles to feet.

                                                                                                                                                                        |
| `FT2NM`
                                                     | Conversion factor from feet to nautical miles.

                                                                                                                                                                        |
| `SM2M`
                                                      | Conversion factor from statute miles to meters.

                                                                                                                                                                       |
| `M2SM`
                                                      | Conversion factor from meters to statute miles.

                                                                                                                                                                       |
| `SM2FT`
                                                     | Conversion factor from statute miles to feet.

                                                                                                                                                                         |
| `FT2SM`
                                                     | Conversion factor from feet to statute miles.

                                                                                                                                                                         |
| `KM2M`
                                                      | Conversion factor from kilometers to meters.

                                                                                                                                                                          |
| `M2KM`
                                                      | Conversion factor from meters to kilometers.

                                                                                                                                                                          |
| `CM2M`
                                                      | Conversion factor from centimeters to meters.

                                                                                                                                                                         |
| `M2CM`
                                                      | Conversion factor from meters to centimeters.

                                                                                                                                                                         |
| `UM2M`
                                                      | Conversion factor from micrometers to meters.

                                                                                                                                                                         |
| `M2UM`
                                                      | Conversion factor from meters to micrometers.

                                                                                                                                                                         |
| `D2SC`
                                                      | Conversion factor for converting degrees to semicircles.

                                                                                                                                                              |
| `SC2D`
                                                      | Conversion factor for converting semicircles to degrees.

                                                                                                                                                              |
| `R2SC`
                                                      | Conversion factor for converting radians to semicircles.

                                                                                                                                                              |
| `SC2R`
                                                      | Conversion factor for converting semicircles to radians.

                                                                                                                                                              |
| `R2DCC`
                                                     | Conversion factor for converting radians to degrees.

                                                                                                                                                                  |
| `D2RCC`
                                                     | Conversion factor for converting degrees to radians.

                                                                                                                                                                  |
| `earth_model_data`
                                          | Data from Different Models of the Earth

                                                                                                                                                                               |

### asapy.utils.json_to_df(self, json, id='id')
Convert a JSON object to a pandas DataFrame and set the index to the given id column.


* **Parameters**

    
    * **json** (*dict*) – A JSON object.


    * **id** (*str*) – The name of the column to set as the index. Default is ‘id’.



* **Returns**

    A DataFrame representation of the JSON object.



* **Return type**

    pandas.DataFrame



### asapy.utils.list_to_df(arr, id='id')
Convert a list of dictionaries to a pandas DataFrame and set the index to the given id column.


* **Parameters**

    
    * **arr** (*list*) – A list of dictionaries.


    * **id** (*str*) – The name of the column to set as the index. Default is ‘id’.



* **Returns**

    A DataFrame representation of the list of dictionaries.



* **Return type**

    pandas.DataFrame



### asapy.utils.unique_list(list1)
Return a list of unique values in the given list.


* **Parameters**

    **list1** (*list*) – A list of values.



* **Returns**

    A list of unique values in the input list.



* **Return type**

    list



### asapy.utils.get_parents_dict(dic, value)
Return a list of keys that lead to the given value in the given dictionary.


* **Parameters**

    
    * **dic** (*dict*) – A dictionary to search.


    * **value** – The value to search for in the dictionary.



* **Returns**

    A list of keys that lead to the given value in the dictionary.



* **Return type**

    list



### asapy.utils.check_samples_similar(new_sample, last_sample, threshold)
Checks if two samples are similar based on a given threshold.


* **Parameters**

    
    * **new_sample** (*np.ndarray*) – The new sample to compare.


    * **last_sample** (*np.ndarray*) – The last sample to compare.


    * **threshold** (*float*) – The threshold to use for comparison.



* **Returns**

    True if the samples are similar, False otherwise.



* **Return type**

    bool



### asapy.utils.test_t(sample1, sample2, alpha=0.05)
Performs a t-test and compares the p-value with a given alpha value.


* **Parameters**

    
    * **sample1** (*np.ndarray*) – The first sample.


    * **sample2** (*np.ndarray*) – The second sample.


    * **alpha** (*float**, **optional*) – The alpha value to use for comparison. Defaults to 0.05.



* **Returns**

    True if the samples are similar, False otherwise.



* **Return type**

    bool



### asapy.utils.FT2M(_ = 0.304_ )
Conversion factor from feet to meters.


### asapy.utils.M2FT()
Conversion factor from meters to feet.


### asapy.utils.IN2M(_ = 0.025_ )
Conversion factor from inches to meters.


### asapy.utils.M2IN()
Conversion factor from meters to inches.


### asapy.utils.NM2M(_ = 1852._ )
Conversion factor from nautical miles to meters.


### asapy.utils.M2NM()
Conversion factor from meters to nautical miles.


### asapy.utils.NM2FT()
Conversion factor from nautical miles to feet.


### asapy.utils.FT2NM()
Conversion factor from feet to nautical miles.


### asapy.utils.SM2M(_ = 1609.34_ )
Conversion factor from statute miles to meters.


### asapy.utils.M2SM()
Conversion factor from meters to statute miles.


### asapy.utils.SM2FT(_ = 5280._ )
Conversion factor from statute miles to feet.


### asapy.utils.FT2SM()
Conversion factor from feet to statute miles.


### asapy.utils.KM2M(_ = 1000._ )
Conversion factor from kilometers to meters.


### asapy.utils.M2KM()
Conversion factor from meters to kilometers.


### asapy.utils.CM2M(_ = 0.0_ )
Conversion factor from centimeters to meters.


### asapy.utils.M2CM()
Conversion factor from meters to centimeters.


### asapy.utils.UM2M(_ = 1e-0_ )
Conversion factor from micrometers to meters.


### asapy.utils.M2UM()
Conversion factor from meters to micrometers.


### asapy.utils.meters_to_micrometers(v)
Convert meters to micrometers.


* **Parameters**

    **v** (*float*) – Value in meters.



* **Returns**

    Value in micrometers.



* **Return type**

    float



### asapy.utils.micrometers_to_meters(v)
Convert micrometers to meters.


* **Parameters**

    **v** (*float*) – Value in micrometers.



* **Returns**

    Value in meters.



* **Return type**

    float



### asapy.utils.meters_to_centimeters(v)
Convert meters to centimeters.


* **Parameters**

    **v** (*float*) – Value in meters.



* **Returns**

    Value in centimeters.



* **Return type**

    float



### asapy.utils.centimeters_to_meters(v)
Convert centimeters to meters.


* **Parameters**

    **v** (*float*) – Value in centimeters.



* **Returns**

    Value in meters.



* **Return type**

    float



### asapy.utils.meters_to_kilometers(v)
Convert meters to kilometers.


* **Parameters**

    **v** (*float*) – Value in meters.



* **Returns**

    Value in kilometers.



* **Return type**

    float



### asapy.utils.kilometers_to_meters(v)
Convert kilometers to meters.


* **Parameters**

    **v** (*float*) – Value in kilometers.



* **Returns**

    Value in meters.



* **Return type**

    float



### asapy.utils.meters_to_inches(v)
Convert meters to inches.


* **Parameters**

    **v** (*float*) – Value in meters.



* **Returns**

    Value in inches.



* **Return type**

    float



### asapy.utils.inches_to_meters(v)
Convert inches to meters.


* **Parameters**

    **v** (*float*) – Value in inches.



* **Returns**

    Value



* **Return type**

    float



### asapy.utils.meters_to_feet(v)
Converts a distance in meters to feet.


* **Parameters**

    **v** (*float*) – distance in meters



* **Returns**

    distance in feet



* **Return type**

    float



### asapy.utils.feet_to_meters(v)
Converts a distance in feet to meters.


* **Parameters**

    **v** (*float*) – distance in feet



* **Returns**

    distance in meters



* **Return type**

    float



### asapy.utils.kilometers_to_nautical_miles(v)
Converts a distance in kilometers to nautical miles.


* **Parameters**

    **v** (*float*) – distance in kilometers



* **Returns**

    distance in nautical miles



* **Return type**

    float



### asapy.utils.nautical_miles_to_kilometers(v)
Converts a distance in nautical miles to kilometers.


* **Parameters**

    **v** (*float*) – distance in nautical miles



* **Returns**

    distance in kilometers



* **Return type**

    float



### asapy.utils.kilometers_to_statute_miles(v)
Converts a distance in kilometers to statute miles.


* **Parameters**

    **v** (*float*) – distance in kilometers



* **Returns**

    distance in statute miles



* **Return type**

    float



### asapy.utils.statute_miles_to_kilometers(v)
Converts a distance in statute miles to kilometers.


* **Parameters**

    **v** (*float*) – distance in statute miles



* **Returns**

    distance in kilometers



* **Return type**

    float



### asapy.utils.nautical_miles_to_statute_miles(v)
Converts a distance in nautical miles to statute miles.


* **Parameters**

    **v** (*float*) – distance in nautical miles



* **Returns**

    distance in statute miles



* **Return type**

    float



### asapy.utils.statute_miles_to_nautical_miles(v)
Converts a distance in statute miles to nautical miles.


* **Parameters**

    **v** (*float*) – distance in statute miles



* **Returns**

    distance in nautical miles



* **Return type**

    float



### asapy.utils.D2SC(_ = 0.005555555555555_ )
Conversion factor for converting degrees to semicircles.


### asapy.utils.SC2D(_ = 180._ )
Conversion factor for converting semicircles to degrees.


### asapy.utils.R2SC(_ = 0.318309886183790_ )
Conversion factor for converting radians to semicircles.


### asapy.utils.SC2R()
Conversion factor for converting semicircles to radians.


### asapy.utils.R2DCC()
Conversion factor for converting radians to degrees.


### asapy.utils.D2RCC()
Conversion factor for converting degrees to radians.


### asapy.utils.degrees_to_radians(v)
Converts a value from degrees to radians.


* **Parameters**

    **v** (*float*) – The value in degrees to convert to radians.



* **Returns**

    The value in radians.



* **Return type**

    float



### asapy.utils.degrees_to_semicircles(v)
Converts a value from degrees to semicircles.


* **Parameters**

    **v** (*float*) – The value in degrees to convert to semicircles.



* **Returns**

    The value in semicircles.



* **Return type**

    float



### asapy.utils.radians_to_degrees(v)
Converts a value from radians to degrees.


* **Parameters**

    **v** (*float*) – The value in radians to convert to degrees.



* **Returns**

    The value in degrees.



* **Return type**

    float



### asapy.utils.radians_to_semicircles(v)
Converts a value from radians to semicircles.


* **Parameters**

    **v** (*float*) – The value in radians to convert to semicircles.



* **Returns**

    The value in semicircles.



* **Return type**

    float



### asapy.utils.semicircles_to_radians(v)
Converts a value from semicircles to radians.


* **Parameters**

    **v** (*float*) – The value in semicircles to convert to radians.



* **Returns**

    The value in radians.



* **Return type**

    float



### asapy.utils.semicircles_to_degrees(v)
Converts a value from semicircles to degrees.


* **Parameters**

    **v** (*float*) – The value in semicircles to convert to degrees.



* **Returns**

    The value in degrees.



* **Return type**

    float



### asapy.utils.aepcd_deg(x)
The method aepcd_deg keeps an angle within the range -180.0 to 180.0 as presented in the figure bellow.
In the example of this figure, the angle of 225 degrees is converted to -135 degrees through aepcd_deg.


* **Parameters**

    **x** (*float*) – Angle in degrees.



* **Returns**

    The angle in degrees adjusted to lie within the range -180.0 to 180.0.



* **Return type**

    float



### asapy.utils.aepcd_rad(x)
Keeps an angle within the range -pi to pi, equivalent to the function aepcd_deg but in radians.


* **Parameters**

    **x** – float, the angle to be checked in radians.



* **Returns**

    float, the angle within the range -pi to pi, with the same orientation as the original angle.



### asapy.utils.alimd(x, limit)
Limits the value of x to +/- limit.


* **Parameters**

    
    * **x** (*float*) – The value to be limited.


    * **limit** (*float*) – The maximum absolute value allowed for x.



* **Returns**

    The limited value of x. If x is greater than limit, returns limit.

        If x is less than negative limit, returns -limit. Otherwise, returns x.




* **Return type**

    float



### asapy.utils.earth_model_data(_ = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,.._ )
Data from Different Models of the Earth
Data from 22 Earth surface models are stored in the array earthModelData.
Each cell of this array corresponds to a type of model and has the following information: semi major axis (a), in meters; and flattening (f).


### asapy.utils.gbd2ll(slat, slon, brg, dist, index_earth_model)
This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.


* **Parameters**

    
    * **latitude** (*-*) – 


    * **bearing** (*-*) – 


    * **distance** (*-*) – 


    * **(****default** (*- an index** of **an optional earth model*) – WGS-84 (indexEarthModel = 0)).



* **Returns**

    
    * latitude (dlat) and longitude (dlon) of the destination point.



**NOTE**: possible values for indexEarthModel.

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


### asapy.utils.fbd2ll(slat, slon, brg, dist)
This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

This method considers the flat-earth projection and a spherical earth radius of ‘ERAD60’. This method is similar to the method of the file nav_utils.inl of MIXR.


* **Parameters**

    
    * **latitude** (*-*) – 


    * **bearing** (*-*) – 


    * **distance** (*-*) – 



* **Returns**

    
    * latitude (dlat) and longitude (dlon) of the destination point.




### asapy.utils.gll2bd(slat, slon, dlat, dlon, index_earth_model)
This function computes the initial bearing and the distance from the starting point to the destination point.

This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.


* **Parameters**

    
    * **latitude** (*-*) – 


    * **latitude** – 


    * **(****default** (*- an index** of **an optional earth model*) – WGS-84 (indexEarthModel = 0)).



* **Returns**

    
    * bearing (brg), in degrees, between the starting and destination points; and


    * distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.



**NOTE**: possible values for indexEarthModel.

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


### asapy.utils.fll2bd(slat, slon, dlat, dlon)
This function computes the initial bearing and the distance from the starting point to the destination point.

This method considers a flat earth projection and a spherical earth radius of ‘ERAD60’.


* **Parameters**

    
    * **latitude** (*-*) – 


    * **latitude** – 



* **Returns**

    
    * bearing (brg), in degrees, between the starting and destination points; and


    * distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.




### asapy.utils.convert_ecef_to_geod(x, y, z, index_earth_model)
This function converts Earth Centered, Earth Fixed (ECEF) coordinates (x,y,z) to geodetic coordinates (latitude,longitude,altitude).


* **Parameters**

    
    * **coordinates** (*- ECEF*) – 


    * **(****default** (*- an index** of **an optional earth model*) – WGS-84 (indexEarthModel = 0)).



* **Returns**

    
    * geodetic coordinates (lat, lon, alt), considering lat and lon in degrees, and alt in meters.



**NOTE**: possible values for indexEarthModel.

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


### asapy.utils.convert_geod_to_ecef(lat, lon, alt, index_earth_model)
This function converts Geodetic ((Latitude,Longitude,Altitude) coordinates) to ECEF ((X,Y,Z) coordinates).


* **Parameters**

    
    * **coordinates** (*- geodetic*) – 


    * **(****default** (*- an index** of **an optional earth model*) – WGS-84 (indexEarthModel = 0)).



* **Returns**

    
    * ECEF coordinates (x,y,z), in meters.



**NOTE**: possible values for indexEarthModel.

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
