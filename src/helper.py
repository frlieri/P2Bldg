"""
src/helper.py

Utility functions used across the project for date/time handling, timeseries
processing, geographic lookups, holiday and typeday determination, basic
numeric checks, serialization helpers, and small data transformations.

Main responsibilities:

- Date and season helpers:
    - parsing and converting MM-DD strings to datetimes,
    - determining day-of-year, season, and typeday (workday/saturday/sunday/holiday),
    - checking whether a date or hour falls within a given period.

- Timeseries and aggregation utilities:
    - block-wise aggregation, linear extrapolation of DataFrame columns,
    - creation of reduced-timeseries MultiIndex and extraction of hour values,
    - identification of contiguous blocks in binary series and conversion of
      hour-period strings to tuple lists.

- Geographic and holiday support:
    - reverse geocoding coordinates to (country, state, city),
    - fetching holidays using the `holidays` package with fallbacks,
    - string reformatting for German umlauts and mapping to project enums.

- Data validation and IO helpers:
    - normalization with respect to sum, range checks, zero/NaN checks,
    - dump/load pickles to the project precalc folder, and result folder creation.

Dependencies and notes:

- Relies on pandas, geopy (Nominatim), holidays, and project constants from
  `src.const`. Many functions expect the project constants to be properly
  configured (e.g., `LATITUDE`, `LONGITUDE`, `MODELED_YEAR`, `PRECALC_FOLDER`).
- Designed for small, testable utility tasks; not intended as a replacement for
  specialized libraries for heavy data processing.
"""


import os
from datetime import datetime, date
from enum import Enum
import holidays
from geopy.geocoders import Nominatim
import pandas as pd
import pickle

from src import const
from src.const import Typeday, Season

geolocator = Nominatim(user_agent="geoapi")
# geolocator = Photon(user_agent="geoapi")


class Tee:
    """
    Class to write to multiple streams simultaneously, e.g., to console and a file.
    This class allows writing output to multiple streams at once. It can be used to
    duplicate output to both the console and a file, or any other combination of
    writable streams.

    :param streams: Variable number of stream objects to write to.
    :type streams: list

    """
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def get_var_t(var_t, args):
    """
    Retrieve or process a variable based on its type.

    This function determines the type of the input variable ``var_t`` and processes or retrieves
    its value accordingly. If ``var_t`` is a pandas Series, it retrieves the value at the
    specified index (``args``). If it is of type float or int, the value is directly returned.
    An error is raised if the type is not among the supported types.

    :param var_t: Input variable to be processed. Can be a pandas Series, a float, or an int.
    :type var_t: Union[pandas.Series, float, int]
    :param args: Index or key to access the data in ``var_t`` when it is a pandas Series.
    :return: The processed value of ``var_t`` as determined by its type.
    :rtype: Union[float, int]
    :raises TypeError: If ``var_t`` is not of type pandas Series, float, or int.
    """
    if isinstance(var_t, pd.Series):
        return var_t[args]
    elif isinstance(var_t, float):
        return var_t
    elif isinstance(var_t, int):
        return var_t
    else:
        raise TypeError("var_t can be of type pd.Series or float or int")


def mm_dd_str_to_datetime(d_str: str, year: int = const.MODELED_YEAR):
    """
    Converts a date string in MM-DD format into a complete datetime object by appending
    the provided year. The function parses the newly created string and returns it as a
    datetime object. If no year is provided, a default year from `const.MODELED_YEAR` is
    used.

    :param d_str: Date string in MM-DD format to be converted.
    :type d_str: str
    :param year: Year to be prepended to the date string. Defaults to `const.MODELED_YEAR`.
    :type year: int
    :return: A datetime object created by combining the given year and date string.
    :rtype: datetime
    """
    d_str = f"{year}-{d_str}"
    d = datetime.strptime(d_str, '%Y-%m-%d')
    return d


def day_of_year(d: pd.DatetimeIndex | datetime):
    """
    Calculate the day of the year for a given date or set of dates.

    This function determines the day of the year for a given date (`datetime` object)
    or for a Pandas `DatetimeIndex`. It raises an error when the input is of an
    unsupported type.

    :param d: The input date or set of dates. It must be either a single
        `datetime` object or a Pandas `DatetimeIndex`.
    :type d: pd.DatetimeIndex | datetime
    :return: The day of the year corresponding to the input date(s). For
        a `datetime` object, it returns an integer representing the day of
        the year (e.g., 1 for January 1st). For a `DatetimeIndex`, it
        returns a Pandas Series or an array-like object of integers.
    :rtype: int | pd.Series
    :raises ValueError: If the input type is neither a `datetime` object
        nor a Pandas `DatetimeIndex`.
    """
    if isinstance(d, datetime):
        return d.timetuple().tm_yday
    elif isinstance(d, pd.DatetimeIndex):
        return d.day_of_year
    else:
        raise ValueError(f"Invalid type: {type(d)}")


def date_within_period(period_tuple: tuple, d: datetime):
    """
    check if date is within a period of a year (e.g. season), not year specific
    :param period_tuple: (start, end) start and end are str objects of format 'mm-dd'
    :param d:
    :return: bool whether d lies within period
    """
    yday_p_start = day_of_year(mm_dd_str_to_datetime(period_tuple[0]))
    yday_p_end = day_of_year(mm_dd_str_to_datetime(period_tuple[1]))

    if yday_p_end > yday_p_start:
        return day_of_year(d) >= yday_p_start and day_of_year(d) <= yday_p_end
    elif yday_p_end < yday_p_start:
        return day_of_year(d) <= yday_p_end or day_of_year(d) >= yday_p_start
    else:
        raise AssertionError("Period start and end are on the same date")


def hour_within_period(period_tuple: tuple, hour: int) -> bool:
    """
    check whether an hour is within a given period, e.g. hour 7 is within [4,8] --> True
    :param period_tuple: tuple or list of two integers
    :param hour: integer
    :return: boolean
    """

    # type conversion if needed
    period_tuple = (int(period_tuple[0]), int(period_tuple[1]))

    if period_tuple[0] < period_tuple[1]:
        return hour >= period_tuple[0] and hour < period_tuple[1]
    elif period_tuple[0] > period_tuple[1]:
        return hour >= period_tuple[0] or hour < period_tuple[1]
    else:
        raise AssertionError("Period start and end is the same hour")


def calc_hours_in_period(period_tuple: tuple) -> int:
    """
    calculate the hours of a period given as (start hour, end hour)
    :param period_tuple: (start hour: int|float, end hour: int|float)
    :return: period length: int
    """

    if period_tuple[0] <= period_tuple[1]:
        return period_tuple[1] - period_tuple[0]
    elif period_tuple[0] > period_tuple[1]:
        return 24 - period_tuple[0] + period_tuple[1]


def get_holidays_from_lib(years: list, state: str = const.STATE) -> list:
    """use holidays package to retrieve a list of holidays for a given state"""
    state_abbr = get_state_abbr_for_state_name(state)
    return list(holidays.country_holidays(country=const.COUNTRY, years=years, state=state_abbr).keys())


def get_holidays(years: list = None) -> list:
    """
    Retrieves a list of holidays for specified years. If no years are provided,
    it uses a default range based on a modeled year constant. Holidays are
    determined based on the geographical information of the system, with
    specific state-dependent holidays fetched if possible. If this fails,
    a set of default holidays is used, including New Year's Day, Christmas Day,
    and Boxing Day.

    :param years: A list of integer years for which holidays will be fetched.
                  Defaults to None, in which case the modeled year constant is used.
    :return: A list of `date` objects representing the holidays for the
             specified years or the default holidays.
    """
    # get state-dependent information for typedays
    if years is None:
        years = list(range(const.MODELED_YEAR, const.MODELED_YEAR + 1))
    try:
        _, state, _ = get_country_state_city_for_coordinates(const.LATITUDE, const.LONGITUDE)
        holidays = get_holidays_from_lib(years, state=const.STATE)
    except:
        holidays = []
        for year in years:
            holidays.extend(
                [date(year, 1,1), date(year, 12, 25), date(year, 12, 26)]
            )
    return holidays


def get_state_abbr_for_state_name(state: str) -> str:
    """
    Fetches the abbreviation for a given state name by matching it against an enumeration
    after preprocessing and formatting it.

    The function retrieves the state name corresponding to coordinates, reformats the
    state name to handle capitalization, special characters, and underscores, and then
    matches the formatted state name with the predefined enumeration to find its
    abbreviation.

    :param state: The name of the state whose abbreviation is to be fetched.
    :type state: str
    :return: The abbreviation of the state as a string.
    :rtype: str
    """
    _, state, _ = get_country_state_city_for_coordinates(const.LATITUDE, const.LONGITUDE)
    state = reformat_string_w_umlauts(state).upper().replace('-', '_')

    return match_str_with_enum(state)


def get_country_state_city_for_coordinates(latitude: (str, int, float), longitude: (str, int, float)) \
        -> (str, str, str):
    """
    Get the country, state, and city for the provided geographical coordinates.

    This function takes geographical coordinates (latitude and longitude) and uses a
    geo-coding service to determine the respective country, state, and city for the provided
    location. It returns these as a tuple.

    :param latitude: Latitude of the location as a string, integer, or float.
    :param longitude: Longitude of the location as a string, integer, or float.
    :return: A tuple containing the country, state, and city as strings.
    """
    coord = f"{latitude}, {longitude}"
    location = geolocator.reverse(coord, exactly_one=True)
    # # Photon
    # properties = location.raw['properties']
    # Nominatim
    properties = location.raw['address']
    city = properties.get('city', '')
    state = properties.get('state', '')
    country = properties.get('country', '')

    return country, state, city


def reformat_string_w_umlauts(string: str):
    """
    Reformats a given string by replacing German umlauts and the 'ß' character
    with their respective transliterations. This function uses a mapping of
    special characters to substitute 'ä' with 'ae', 'ü' with 'ue', 'ö' with 'oe',
    and 'ß' with 'ss'.

    :param string: Input string to be reformatted. It may contain German umlauts
        or the 'ß' character to be replaced.
    :type string: str
    :return: Reformatted string with all applicable characters substituted based on
        the defined mapping.
    :rtype: str
    """
    special_char_map = {
        ord('ä'): 'ae',
        ord('ü'): 'ue',
        ord('ö'): 'oe',
        ord('ß'): 'ss'
    }
    return string.translate(special_char_map)


def match_str_with_enum(search_str: str, enum: Enum = const.State):
    """
    Matches a string with an enumeration and retrieves the corresponding value from the enum.

    This function facilitates mapping a string representation of an enum name
    to its corresponding value in the provided enumeration. It employs a
    dictionary comprehension to construct a mapping of enum names to values,
    and retrieves the value associated with the provided string key.

    :param search_str: The name of the enumeration to be matched.
    :param enum: The enumeration from which the value is fetched. Defaults
        to const.State.
    :return: The value of the enum corresponding to the provided
        ``search_str``.
    :rtype: Any
    """
    dct = {i.name: i.value for i in enum}

    return dct[search_str]


def normalize_wrt_sum(s: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series with respect to its sum.

    This function takes a pandas Series and normalizes each element by dividing it
    by the sum of all elements in the Series. The result is a Series where the
    values add up to 1, unless the input Series contains only zeros.

    :param s: A pandas Series to be normalized.
    :type s: pd.Series
    :return: A normalized pandas Series where the values sum to 1.
    :rtype: pd.Series
    """
    return s/s.sum()


def dump_pickle(fname: str, data):
    """
    Dumps the given data into a pickle file located in a predefined folder with the
    specified filename. This function serializes the data and writes it into a file
    using the Python `pickle` module. The resulting file is stored with the `.pkl`
    extension.

    :param fname: The name of the file (without extension) to save the serialized
                  data to.
    :type fname: str
    :param data: The data object to serialize and dump into the file.
    :return: None
    """
    fpath = const.PRECALC_FOLDER + fname + '.pkl'
    with open(fpath, 'wb') as handle:
        pickle.dump(data, handle)
    print('dumped ', fpath)


def load_pickle(fname: str, folder: str = const.PRECALC_FOLDER):
    """
    Loads a pickle file from a specified folder and filename. The method constructs
    the file path by combining the folder path with the filename and '.pkl'
    extension, then deserializes the file content into a Python object using the
    pickle module.

    :param fname: Name of the pickle file (without extension) to be loaded.
    :type fname: str
    :param folder: Folder path where the pickle file is located. Defaults to
        `const.PRECALC_FOLDER`.
    :type folder: str
    :return: The deserialized Python object from the pickle file.
    :rtype: Any
    """
    fpath = folder + fname + '.pkl'
    with open(fpath, 'rb') as handle:
        pkl = pickle.load(handle)
    print('loaded ', fpath)
    return pkl


def check_if_values_outside_range(s: pd.Series, min=0, max=10 ** 9):
    """
    Checks if the values in the provided pandas Series are outside of the specified range.

    This function examines whether any of the values in the given pandas Series are
    outside the defined minimum and maximum range. If any value falls below the
    minimum or exceeds the maximum, an assertion error is raised, indicating that
    values in the Series are out of the acceptable range.

    :param s: A pandas Series to be checked for out-of-range conditions.
    :type s: pd.Series
    :param min: The minimum acceptable value for elements in the Series
        (inclusive). Defaults to 0.
    :type min: int, optional
    :param max: The maximum acceptable value for elements in the Series
        (inclusive). Defaults to 10 ** 9.
    :type max: int, optional
    :raises AssertionError: If any value in the Series is outside the specified
        [min, max] range.
    """
    lower_than_min = s[s < min].any()
    higher_than_max = s[s > max].any()

    assert not (lower_than_min or higher_than_max), "Values of series outside range"


def yield_pos_of_next_higher_number_in_ordered_list(ordered_list: list, number: (float, int)):
    """
    Yield the position of the next higher (or equal) number in the ordered list compared to the given number.

    This function iterates through an ordered list and identifies the position at which a number that is
    greater than or equal to the input number occurs. If no such number exists, it returns the position
    of the last element in the list. The input list must be ordered to ensure correct functionality.

    :param ordered_list: A list of numbers that is expected to be sorted in ascending order.
    :param number: A number (float or int) to search for the next higher or equal value in the
        given ordered list.
    :return: Position (integer index) of the closest number in the list that is greater than or
        equal to the given number. If no such number is found, returns the position of the last
        element in the list.
    """
    for i, e in enumerate(ordered_list):
        if number <= float(e):
            return i
    return i


HOLIDAYS = get_holidays()
def get_typeday(d: datetime, holidays=HOLIDAYS):
    """
    Determine the type of day (Sunday, Saturday, or Workday) for a given date.

    This function takes a specific date and an optional list of holiday dates
    and returns the type of the day based on whether it falls on a weekend or
    is a holiday.

    :param d: The date to evaluate.
    :type d: datetime
    :param holidays: A collection of holiday dates. If the given date matches one
        of these dates, it will be treated as a Sunday. Defaults to HOLIDAYS.
    :type holidays: set
    :return: Numeric representation of the day type: Sunday, Saturday, or Workday.
    :rtype: int
    """

    if d.weekday() == 6 or d.date() in holidays:
        return Typeday.Sunday.value
    elif d.weekday() == 5:
        return Typeday.Saturday.value
    else:
        return Typeday.Workday.value


def get_season(d: datetime):
    """
    Determines the season corresponding to the given date based on predefined seasonal
    periods. The function checks whether the given date falls within the defined periods
    for Winter, Summer, or Transition. If the date does not match any defined period, an
    AssertionError is raised.

    :param d: The date for which the season needs to be determined.
    :type d: datetime
    :return: The season value corresponding to the date
    :rtype: int
    :raises AssertionError: If the date does not fall into any defined seasonal period.
    """
    if date_within_period(const.SEASON_PERIODS[Season.Winter.value][0], d):
        return Season.Winter.value
    elif date_within_period(const.SEASON_PERIODS[Season.Summer.value][0], d):
        return Season.Summer.value
    elif (date_within_period(const.SEASON_PERIODS[Season.Transition.value][0], d) or
          date_within_period(const.SEASON_PERIODS[Season.Transition.value][1], d)):
        return Season.Transition.value
    else:
        raise AssertionError('Invalid season')


def calc_for_blocks_of_n(df: (pd.DataFrame, pd.Series), block_size=None, func='mean') -> pd.Series:
    """
    Calculate aggregated values for blocks of specified size from a pandas DataFrame or Series.

    This function divides the input data into blocks of a given size and applies an
    aggregation function to each block. The aggregated values are then aligned to
    the original index of the input data.

    :param df: The input data to process, either a pandas DataFrame or Series.
    :param block_size: The size of each block for grouping, defaults to 24.
    :type block_size: int, optional
    :param func: The aggregation function to apply to each block. Can be a string
        representing a common pandas aggregation function (e.g., 'mean', 'sum') or
        a callable function, defaults to 'mean'.
    :type func: str or callable, optional
    :return: A pandas Series containing the aggregated values aligned to the
        original index structure of the input data.
    :rtype: pd.Series
    """
    if block_size == None:
        block_size = const.TS_PER_HOUR * 24

    old_index = df.index
    df = df.reset_index(drop=True)
    group_number = df.index // block_size
    group_means = df.groupby(group_number).transform(func)
    group_means.index = old_index

    return group_means


def linear_extrapolate_all_columns(df: pd.DataFrame, new_x: int|float):
    """
    Linearly extrapolates to estimate new y values for all columns, using the DataFrame index as x values.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data with x values as the index.
        new_x (float): The new x value for which y values will be extrapolated.

    Returns:
        pd.Series: A Series with extrapolated values for each column.
    """
    if len(df) < 2:
        raise ValueError("DataFrame must contain at least two rows for extrapolation.")

    # Ensure the index is sorted
    df = df.sort_index()

    extrapolated_values = {}

    # Get the last two rows of the DataFrame
    x1, x2 = df.index[-2], df.index[-1]

    for col in df.columns:
        y1, y2 = df.iloc[-2][col], df.iloc[-1][col]

        # Calculate the slope (m)
        slope = (y2 - y1) / (x2 - x1)

        # Extrapolate to find y at new_x
        new_y = y2 + slope * (new_x - x2)
        extrapolated_values[col] = new_y

    # Return the results as a pandas Series
    return pd.Series(extrapolated_values)


def calc_th_demand_summary(dem: dict, imp: dict, cost_factors: pd.Series):
    """
    Calculates the thermal demand summary for room heating and domestic hot water (DHW).
    This function computes thermal losses, gains, and demand factors based on the input
    data and applies cost factors to generate a detailed summary.

    :param dem: Dictionary containing the thermal room heating and DHW demand data.
    :param imp: Dictionary containing the impact data for solar and internal heat gains.
    :param cost_factors: Series containing cost factors to scale the computed values.
    :return: A DataFrame with detailed thermal demand summary including transmission
        losses, ventilation losses, solar heat gains, internal heat gains, DHW, and
        aggregated sums.
    """
    th_dem_summary = pd.DataFrame({
        'trans_loss': dem['th_roomheat'].transm_loss * dem['th_roomheat'].delta_temp_t,
        'vent_loss': dem['th_roomheat'].vent_loss * dem['th_roomheat'].delta_temp_t,
        'solar_gains': - imp['heat_gains'].solar_gains_t,
        'internal_gains': - imp['heat_gains'].internal_gains_t,
        'dhw': dem['th_dhw'].p_t,
    }).apply(lambda x: x * cost_factors)

    th_dem_summary['roomheat_sum'] = th_dem_summary[[
        'trans_loss', 'vent_loss', 'solar_gains', 'internal_gains'
    ]].apply(lambda x: x.sum() if x.sum() > 0 else 0, axis=1)
    th_dem_summary['sum'] = th_dem_summary['roomheat_sum'] + th_dem_summary['dhw']

    return th_dem_summary


def create_ts_data_dump_id():
    """
    Generates a unique identifier for a timeseries data dump based on geographic
    position, modeled year, and reduction settings.

    The function creates an identifier string composed of predefined constants
    representing latitude, longitude, modeled year, and optionally includes a tag
    to signify timeseries reduction if that setting is applied.

    :return: A string representing the unique identifier for the timeseries data
        dump.
    :rtype: str
    """
    dump_id = f"{const.LATITUDE}_{const.LONGITUDE}_{const.MODELED_YEAR}"
    if const.APPLY_TIMESERIES_REDUCTION:
        dump_id += f"_ts_sel_weeks"

    return dump_id

def create_climate_data_dump_id():
    """
    Generates a unique identifier for a climate data dump based on pre-defined constants.
    The identifier combines geographic and temporal attributes to ensure uniqueness.

    :return: A string representing the unique climate data dump identifier.
    :rtype: str
    """
    dump_id = (f"{const.LATITUDE}_{const.LONGITUDE}_{const.CLIMATE_DATA_YEARS[0]}-{const.CLIMATE_DATA_YEARS[1]}_"
               f"climate_summary")
    return dump_id


def check_if_zero_or_nan(val) -> bool:
    """
    Checks if the given value is either zero or NaN.

    This function evaluates whether the provided value is equal to 0.0 or is a
    string representation of 'nan'. It returns a boolean reflecting the result
    of this evaluation.

    :param val: The value to be checked
    :type val: Any
    :return: True if the value is either 0.0 or 'nan', otherwise False
    :rtype: bool
    """
    if val == 0.0:
        return True
    elif str(val) == 'nan':
        return True
    return False


def create_result_folder(proj_name, top_folder=const.PATH_TO_WD + "/data/output/results"):
    """
    Creates a result folder for a given project and organizes it in a specified top directory.

    This function generates a unique folder name for a project by appending the current
    timestamp to the project's name. If the folder does not exist, it is created. The function
    returns the path to the newly created folder.

    :param proj_name: Name of the project used to create a uniquely named folder.
    :type proj_name: str
    :param top_folder: The base directory where the result folder will be created. Defaults
        to a predefined directory within the workspace.
    :type top_folder: str, optional
    :return: The full path to the newly created result folder.
    :rtype: str
    """
    new_res_folder = f"{top_folder}/{proj_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    if not os.path.exists(new_res_folder):
        os.makedirs(new_res_folder)

    return new_res_folder


def create_ts_multiindex(nr_of_days_in_red_ts=None, apply_ts_reduction=None) -> pd.MultiIndex:
    """
    create a pandas Multiindex of the format (Season, timestep) for calculations with reduced timeseries
    :param nr_of_days_in_red_ts: number of days per season in reduced timeseries
    :return: pd.Multiindex
    """
    if nr_of_days_in_red_ts is None:
        nr_of_days_in_red_ts = 7

    if apply_ts_reduction is None:
        apply_ts_reduction = const.APPLY_TIMESERIES_REDUCTION

    if apply_ts_reduction:
        multi_ind = pd.MultiIndex.from_tuples(
        [(season.value, t) for season in Season for t in range(0, nr_of_days_in_red_ts * 24 * const.TS_PER_HOUR)]
    )
    else:
        multi_ind = pd.MultiIndex.from_tuples(
            [(const.MODELED_YEAR, t) for t in range(0, 8760 * const.TS_PER_HOUR)]
        )

    return multi_ind


def get_hour_from_multiindex(ind: pd.MultiIndex) -> list:
    """
    returns the hours for the Multi-Index (Season, timestep) of a reduced timeseries
    :param: ind: pd.Mulit-Index of the format (Season, timestep)
    :return: list
    """
    return [int(i) for i in ind.get_level_values(1) % (24 * const.TS_PER_HOUR) / const.TS_PER_HOUR]


def identify_blocks(series: pd.Series) -> dict[int, list[int]]:
    """
    Identify contiguous blocks of 1s in a pandas Series.

    Parameters
    ----------
    series : pd.Series
        A pandas Series containing 0s and 1s.

    Returns
    -------
    dict
        A dictionary {block_number: [indices]} representing each block of 1s.
    """
    # Ensure the series is boolean for logical operations
    s = series.astype(bool)

    # Identify where blocks start: True when s[i]=1 and s[i-1]=0
    block_starts = (s != s.shift()).cumsum() * s  # increment on state changes

    # Group by the block id (only for 1s)
    blocks = {}
    for i, (block_id, idxs) in enumerate(s[block_starts > 0].groupby(block_starts), start=1):
        blocks[i] = list(idxs.index)

    return blocks


def tuple_list_from_hour_periods(hh_str: str) -> list[tuple]:
    """
    convert a string of e.g. format "HH-HH,HH-HH" to [(HH,HH),(HH,HH)]
    :param hh_str: string with hour periods
    :return: list of tuples
    """
    hh_str = hh_str.replace(' ', '')
    if hh_str == "" or hh_str == "nan":
        return []
    try:
        tup_lst = [(int(period.split('-')[0]),int(period.split('-')[1])) for period in hh_str.split(',')]
    except ValueError:
        raise Exception(f"Invalid sequence for hour period '{hh_str}'")
    return tup_lst








