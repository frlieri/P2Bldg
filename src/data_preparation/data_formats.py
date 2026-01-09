"""
src/data_preparation/data_formats.py

This module provides core data structures and utilities for preparing and
handling time-series and scenario data used by the optimization workflows.

Primary classes and responsibilities:

- Timeseries:
    Encapsulates time-indexed data using a pandas DataFrame and provides
    utilities for indexing, resampling (hourly, quarter-hourly, daily),
    filtering by season/typeday/week/hour ranges, aggregation, normalization,
    and plotting helpers for typical-day analysis.

- ScenarioData:
    Wraps scenario-specific input values extracted from the project Excel
    workbook. It enforces declared datatypes and yields structured sheet-level
    data objects for downstream initialization of model components.

- SheetData:
    Dynamically organizes sheet parameters and component entries (e.g.
    pv, batt, hp, st) into attributes and nested dictionaries. Empty component
    entries are dropped to simplify component initialization.

- BuildingModelInput:
    Loads and validates the structured Excel input file, enforces consistent
    scenario definitions across sheets, and prepares scenario dictionaries and
    heat-pump specification tables for use by model builders.

- ScenarioResults:
    Converts a solved Pyomo ConcreteModel into a set of labeled time-series
    tables and time-invariant component summaries. It exposes helpers to query
    component-level costs, balances and time-series slices by regex and season.

- prepare_dummy_ts:
    Convenience function to create a Timeseries instance populated with a
    constant value for quick testing or placeholder inputs.

Notes and dependencies:

- This module depends on pandas and pyomo (ConcreteModel types are referenced),
  and it integrates with project constants and helper functions defined in
  `src.const` and `src.helper`.
- Timeseries index/metadata columns follow the project `TimeseriesCols`
  enumeration and rely on modeled year and time-step constants for correct
  indexing and resampling behaviour.
- Designed to be agnostic to solver details; conversion of model outputs to
  pandas DataFrames is intended to simplify postprocessing, plotting and
  export.

"""

from datetime import datetime
from typing import Callable
import pandas as pd
from pyomo.environ import ConcreteModel
from math import floor

from src import const
from src.const import TimeseriesCols, Season
from src.helper import get_season, get_typeday, normalize_wrt_sum, hour_within_period, create_ts_multiindex
from src.export.analysis_plots import plot_typical_day_quantiles, plot_timeseries


class Timeseries():
    """
    Handles time-series data and provides functionality for manipulation, aggregation, resampling,
    filtering, and visualization. This class is primarily designed to leverage pandas DataFrame under
    the hood to enable robust time-series data analysis.

    The class organizes time-series data within a structured DataFrame, allowing users to define
    attributes such as frequency, temporal periods, and specific data columns. It supports operations
    like normalization, averaging, resampling between resolutions, and specialized plots for
    analytic visualizations.

    :ivar years: List of years covered by the time-series data.
    :type years: list[int]
    :ivar freq: Frequency/step size of the time-series data.
    :type freq: Any
    :ivar index_columns: List of pre-defined index columns for time-series metadata.
    :type index_columns: list[str]
    :ivar data_columns: List of data columns that represent the core time-series values.
    :type data_columns: list[str]
    :ivar df: The primary DataFrame representation of the time-series data.
    :type df: pandas.DataFrame
    """

    def __init__(self, df: pd.DataFrame = None, freq=const.TSStepSize.HOURLY, columns=None, start_year=const.MODELED_YEAR,
                 end_year=None):
        """
        Initializes the timeseries object with data and parameters for analysis. The
        method sets up the time series structure, including index and data columns,
        and populates initial values if a DataFrame is provided.

        :param df: Input data as a pandas DataFrame. If not provided, an empty
            DataFrame is created.
        :type df: pandas.DataFrame, optional

        :param freq: Time step size for the timeseries. Default is `HOURLY`.
            Defines the temporal resolution of the time series.
        :type freq: const.TSStepSize, optional

        :param columns: List of column names for the data portion of the timeseries.
            If `None` and `df` is provided, column names are derived from the DataFrame;
            otherwise, defaults to an empty list.
        :type columns: list of str, optional

        :param start_year: The starting year of the time series. Determines the first
            year in the year range.
        :type start_year: int, optional

        :param end_year: The ending year of the time series. If not provided, it
            defaults to `start_year`. Defines the last year in the year range.
        :type end_year: int, optional
        """

        if columns is None:
            if df is None:
                columns = []
            else:
                columns = df.columns.tolist()
        if end_year is None:
            end_year = start_year
        self.years = list(range(start_year, end_year + 1))
        self.freq = freq
        self._set_ts_per_day()
        self.index_columns = [TimeseriesCols.DTime, TimeseriesCols.Hour, TimeseriesCols.Step, TimeseriesCols.TSDay,
                              TimeseriesCols.Yday, TimeseriesCols.Week, TimeseriesCols.Month, TimeseriesCols.Year,
                              TimeseriesCols.Season, TimeseriesCols.Typeday]
        self.data_columns = columns

        self.df = pd.DataFrame(columns=[*self.index_columns, *self.data_columns])
        self._init_index_cols()

        # set values from df given
        if df is not None:
            df.index = self.df.index
            for col in self.data_columns:
                self[col] = df[col]

    def _set_ts_per_day(self):
        if self.freq == const.TSStepSize.HOURLY:
            self.ts_per_day = 24
        elif self.freq == const.TSStepSize.QUARTER_HOURLY:
            self.ts_per_day = 96
        elif self.freq == const.TSStepSize.DAILY:
            self.ts_per_day = 1
        else:
            raise ValueError("invalid freq for Timeseries")

    def _init_index_cols(self):
        # add columns for day of year and month
        self[TimeseriesCols.DTime] = pd.date_range(
            start=datetime(self.years[0], 1, 1),
            end=datetime(self.years[-1], 12, 31, 23, 59),
            freq=self.freq
        )
        self[TimeseriesCols.Step] = range(0, len(self[TimeseriesCols.DTime]))
        self[TimeseriesCols.Hour] = self[TimeseriesCols.DTime].apply(lambda x: x.hour)
        self[TimeseriesCols.TSDay] = self[TimeseriesCols.Step].apply(lambda x: floor(x/self.ts_per_day)+1)
        self[TimeseriesCols.Yday] = self[TimeseriesCols.DTime].apply(lambda x: x.timetuple().tm_yday)
        self[TimeseriesCols.Week] = self[TimeseriesCols.DTime].apply(lambda x: x.week)
        self[TimeseriesCols.Month] = self[TimeseriesCols.DTime].apply(lambda x: x.month)
        self[TimeseriesCols.Year] = self[TimeseriesCols.DTime].apply(lambda x: x.year)
        self[TimeseriesCols.Season] = self[TimeseriesCols.DTime].apply(lambda x: get_season(x))
        self[TimeseriesCols.Typeday] = self[TimeseriesCols.DTime].apply(lambda x: get_typeday(x))

        self.df.index = self[TimeseriesCols.Step]

    def __setitem__(self, key, value):
        # setting dataseries by name using ts['name']
        self.df.loc[:, key] = value
        if key not in self.index_columns + self.data_columns:
            self.data_columns.append(key)

    def __getitem__(self, key):
        # getting dataseries by name using ts['name']
        return self.df.loc[:, key]

    def __repr__(self):
        return self.df.__repr__()

    def no_of_timesteps(self):
        return self.df.shape[0]

    def sum(self):
        return self.df[self.data_columns].sum()

    def max(self):
        return self.df[self.data_columns].max()

    def min(self):
        return self.df[self.data_columns].min()

    def mean(self):
        return self.df[self.data_columns].mean()

    def normalize(self, columns: (list, str)=None):
        """
        Normalize specified columns or data columns of the dataframe with respect
        to their sum. This can help in bringing the values of the dataset into
        a specific range by dividing them by their respective sum.

        :param columns: List of column names or string representing a column name
            to normalize. If None, all data columns will be normalized.
        :type columns: list or str, optional
        :return: The dataframe after performing normalization on the specified
            columns.
        :rtype: None
        """
        if columns is None:
            self.df.loc[:, self.data_columns] = normalize_wrt_sum(self.df.loc[:, self.data_columns])
        else:
            self.df.loc[:, columns] = normalize_wrt_sum(self.df.loc[:, columns])

    def monthly_avg(self, columns: str | list = None):
        return self.df.groupby(TimeseriesCols.Month)[columns].mean()

    def daily_avg(self, columns: str | list = None):

        return self.apply_agg(groupby_col=TimeseriesCols.TSDay, func=pd.DataFrame.mean, data_cols=columns)

    def apply_agg(self, groupby_col: str, func: Callable, data_cols: list):
        """
        Applies an aggregation function across specified columns after grouping the
        dataframe by a given column.

        This method groups the dataframe by the specified column and applies the given
        function to the data in the specified columns. If no specific columns are
        provided, the aggregation function is applied across all columns in the
        dataframe.

        :param groupby_col: The name of the column by which the dataframe should
            be grouped.
        :type groupby_col: str
        :param func: The function to be applied to the grouped data.
        :type func: Callable
        :param data_cols: A list of column names specifying the subset of the
            dataframe to which the function should be applied. If None, all columns
            are included.
        :type data_cols: list
        :return: The resulting dataframe after applying the aggregation function to
            the grouped data.
        :rtype: pandas.DataFrame
        """
        if data_cols is None:
            data_cols = self.df.columns

        return self.df.groupby(groupby_col)[data_cols].apply(func)

    def convert_to_hourly(self):
        self._resample(res_target_profile_in_h=1.0)     # instead pandas resample function could be used
        self.freq = const.TSStepSize.HOURLY
        self._set_ts_per_day()
        self._init_index_cols()

    def convert_to_quarter_hourly(self):
        self._resample(res_target_profile_in_h=0.25)
        self.freq = const.TSStepSize.QUARTER_HOURLY
        self._set_ts_per_day()
        self._init_index_cols()

    def convert_to_daily(self):
        self._resample(res_target_profile_in_h=24)
        self.freq = const.TSStepSize.DAILY
        self._set_ts_per_day()
        self._init_index_cols()

    def _resample(self, res_target_profile_in_h, downsampling_val_used='mean'):
        """
        Resamples the timeseries data to the specified target profile resolution. This method adjusts the resolution of the
        data in the DataFrame either from a higher to a lower resolution (down-sampling) or from a lower to a higher
        resolution (up-sampling), based on the provided target profile resolution. The method supports various aggregation
        functions for down-sampling such as mean, min, and max.

        :param res_target_profile_in_h: Desired target resolution for the timeseries profile in hours.
        :type res_target_profile_in_h: float
        :param downsampling_val_used: The aggregation function to use for down-sampling. Valid options are 'mean', 'min',
            and 'max'. Default is 'mean'.
        :type downsampling_val_used: str
        :return: None
        """


        if self.freq == const.TSStepSize.HOURLY:
            res_self_in_h = 1.0
        elif self.freq == const.TSStepSize.QUARTER_HOURLY:
            res_self_in_h = 0.25
        else:
            raise ValueError("Invalid frequency for Timeseries")

        res_ratio = res_self_in_h / res_target_profile_in_h

        # from higher to lower resolution
        if res_ratio < 1:
            high_low_ratio = int(1 / res_ratio)
            target_profile = pd.DataFrame(
                columns=self.df.columns, index=range(0, int(self[TimeseriesCols.Step].max() / high_low_ratio) + 1)
            )
            for i in target_profile.index:
                i_start = i * high_low_ratio
                i_end = (i + 1) * high_low_ratio - 1
                if downsampling_val_used == "mean":
                    target_profile.loc[i, self.data_columns] = self.df[i_start: i_end][self.data_columns].mean()
                elif downsampling_val_used == "min":
                    target_profile.loc[i, self.data_columns] = self.df[i_start: i_end][self.data_columns].min()
                elif downsampling_val_used == "max":
                    target_profile.loc[i, self.data_columns] = self.df[i_start: i_end][self.data_columns].max()
                else:
                    raise ValueError("Invalid func_for_upsampling param")

        # from lower to higher resolution
        elif res_ratio > 1:
            high_low_ratio = int(res_ratio)
            target_profile = pd.DataFrame(
                columns=self.df.columns,
                index=range(0, int(self.df.index.max() * high_low_ratio) + high_low_ratio)
            )
            for i in self.df.index:
                target_profile[(i * high_low_ratio):((i + 1) * high_low_ratio)] = self.df.iloc[i]

        else:
            target_profile = self.df

        self.df = target_profile
        self.freq = f"{res_target_profile_in_h:.2f}h"

    def get_specs(self):
        return pd.DataFrame({
            'min': self.min(),
            'max': self.max(),
            'avg': self.mean(),
            'sum': self.sum()
        })

    def plot_ts(self, columns: list = None):
        """
        Plots the timeseries data for specified columns in the DataFrame. If no columns
        are specified, it defaults to plotting all columns defined in the dataset's
        data_columns attribute.

        :param columns: A list of column names to plot. If None, all relevant columns
                        in the dataset will be used for plotting.
        :type columns: list, optional
        :return: None
        """
        if not columns:
            columns = self.data_columns
        plot_timeseries(self.df[columns])

    def plot_typeday(self, column: str, season=None, typeday=None, title=None):
        """
        Plots the typical day quantiles for a given column in the dataframe. This method filters
        the data based on specified season and type of day, and then plots the quantiles for
        the specified column. The plot can also have a custom title if provided.

        :param column: The name of the column in the dataframe to be plotted.
        :type column: str
        :param season: Optionally specify a season for filtering the data.
        :type season: Optional
        :param typeday: Optionally specify a type of day for filtering the data.
        :type typeday: Optional
        :param title: Optional custom title for the plot. Defaults to the column name.
        :type title: Optional
        :return: None
        """
        if title is None:
            title = column
        self.df.index = self.df[TimeseriesCols.DTime]
        plot_typical_day_quantiles(self.filter(season=season, typeday=typeday)[column].astype(float), title=title)
        self.df.index = self.df[TimeseriesCols.Step]

    def filter(self, season=None, weeks: list|int = None, typeday=None, daysofweek: list = None, hour_periods=None,
               data_only=False):
        """
        Filters the dataset based on various temporal and categorical parameters.
        The method enables filtering by seasons, weeks, typedays, days of the week,
        specific hour periods, and can return data-only columns if required.

        :param season: The season to filter the dataset, must match values
                       defined in ``const.Season``
        :param weeks: A list or single integer representing specific week numbers
                      to filter, ranging from 0 to 52
        :param typeday: The type of day to filter, must match values defined in
                        ``const.Typeday``
        :param daysofweek: A list of integers (ranging from 0 to 6) representing
                           days of the week to filter, where 0 is Monday and 6 is
                           Sunday
        :param hour_periods: A collection of hour periods used for filtering. Each
                             period should define a valid range of hours
        :param data_only: A boolean flag. If set to ``True``, only columns specified
                          in ``self.data_columns`` are included in the returned DataFrame
        :returns: A filtered pandas DataFrame based on the specified parameters
        :rtype: pandas.DataFrame
        """
        filtered = self.df
        if season is not None:
            assert season in [s.value for s in const.Season], 'Invalid season'
            filtered = filtered[filtered[TimeseriesCols.Season] == season]
        if weeks is not None:
            if not isinstance(weeks, list):
                weeks = [int(weeks)]
            assert all(w in range(0,53) for w in weeks), 'Invalid week '
            filtered = filtered[filtered.apply(lambda x: x[TimeseriesCols.Week] in weeks, axis=1)]
        if typeday is not None:
            assert typeday in [t.value for t in const.Typeday], 'Invalid Typeday'
            filtered = filtered[filtered[TimeseriesCols.Typeday] == typeday]
        if daysofweek is not None:
            assert all(dow in range(0,7) for dow in daysofweek), 'Invalid daysofweek'
            filtered = filtered[filtered.apply(lambda x: x[TimeseriesCols.DTime].dayofweek in daysofweek, axis=1)]
        if hour_periods is not None:
            filter = pd.Series(False, index=filtered.index)
            for period in hour_periods:
                filter |= filtered.apply(lambda x: hour_within_period(period, x[TimeseriesCols.Hour]), axis=1)

            filtered = filtered[filter]

        if data_only:
            filtered = filtered[self.data_columns].reset_index(drop=True)

        return filtered

    def groupby(self, *args, **kwargs):
        return self.df.groupby(*args, **kwargs)

    def duration_curve(self, column: str, season=None, typeday=None) -> pd.DataFrame:
        return self.filter(season=season, typeday=typeday)[column].sort_values(ascending=False)

    def daysums(self, columns: list|str = None, season='all') -> pd.Series:
        """
        Sums up data values by day for specified columns in the dataset. If a season is
        specified, sums are computed only for data within the given season; otherwise,
        the entire dataset is used.

        :param columns: List of column names or a single column name to sum by day. If
            not provided, defaults to all data columns.
        :type columns: list | str, optional
        :param season: The season for which data should be filtered ('all' by default
            for no filtering).
        :type season: str
        :return: A pandas Series where the index corresponds to days, and the values
            are the sum of the specified column(s) for each day.
        :rtype: pandas.Series
        """
        if columns is None:
            columns = self.data_columns
        if isinstance(columns, str):
            columns = [columns]
        if season != "all":
            return self.filter(season).groupby(TimeseriesCols.TSDay)[columns].sum().astype(float)
        else:
            return self.groupby(TimeseriesCols.TSDay)[columns].sum().astype(float)

    @staticmethod
    def _get_quant_range(daysum: float, daysum_quants: pd.Series):
        is_larger_mask = (daysum_quants <= daysum)
        low_quant = (0.00 if daysum_quants[is_larger_mask].empty
                     else daysum_quants[is_larger_mask].reset_index().iloc[-1]['index'])
        is_smaller_mask = (daysum_quants > daysum)
        high_quant = (1.00 if daysum_quants[is_smaller_mask].empty
                      else daysum_quants[is_smaller_mask].reset_index().iloc[0]['index'])

        return f"{low_quant}-{high_quant}"

    def get_daysum_quant_range(self, columns: list|str = None, season='all', quantiles=None) -> pd.DataFrame:
        """
        Calculates the range of quantiles for the daily sums of specified columns within a dataset,
        optionally filtered by season. The method allows users to analyze the distribution and
        quantitative ranges of the daily aggregated values.

        :param columns: A list or string that specifies the data columns to process. If None,
            all data columns will be processed. If a string is provided, it will be
            converted into a single-element list.
            Type: list | str | None
        :param season: A string indicating the season for which data should be filtered.
            Default is 'all', which means no seasonal filtering is applied.
            Type: str
        :param quantiles: A list of float values indicating the quantiles to use for the
            calculation. If None, default quantiles from 0.0 to 0.9 (incremented by 0.1)
            will be used.
            Type: list[float] | None
        :return: A pandas DataFrame containing the quantile range classifications for the
            daily sums of the specified columns. Each column in the DataFrame corresponds
            to a column from the input, with values classified based on their respective
            quantile ranges.
            Type: pd.DataFrame
        """
        if columns is None:
            columns = self.data_columns
        if isinstance(columns, str):
            columns = [columns]
        if quantiles is None:
            quantiles = [i/10 for i in range(0, 10)]

        daysum_quant_ranges = {}
        for col in columns:
            daysums = self.daysums(col, season)
            daysum_quants = daysums.quantile(quantiles)
            daysum_quant_ranges[col] = daysums[col].apply(lambda x: self._get_quant_range(x, daysum_quants[col]))

        return pd.DataFrame(daysum_quant_ranges)

    def get_profiles_for_days(self, day_selection: list, data_only=False) -> pd.DataFrame:
        """
        for all days in dayselection get the corresponding dayprofile and concatenate them
        :param day_selection: ordered list for day selection to create profiles
        :param data_only: decision variable whether only data columns should be passed or also Timeseries indices
        :return: pd.DataFrame with the concatenated dayprofiles
        """

        ret_df = pd.DataFrame()
        for day in day_selection:
            ret_df = pd.concat([ret_df, self.df[self.df[TimeseriesCols.TSDay] == day]])

        if data_only:
            ret_df = ret_df[self.data_columns].reset_index(drop=True)

        return ret_df


class ScenarioData():
    """
    Represents a scenario-specific dataset extracted from the input excel sheet.

    This class is designed to manage and initialize data relevant to a specific
    scenario in a structured format. It extracts and converts data relevant to the
    scenario from the provided Excel sheets, ensuring proper datatype validation
    and consistency.

    :ivar scenario_name: The name of the scenario being processed.
    :type scenario_name: str
    :ivar location: Sheet-specific data for 'Location', initialized for the given scenario.
    :type location: SheetData
    :ivar consumption_data: Sheet-specific data for 'Consumption Data', initialized for the given scenario.
    :type consumption_data: SheetData
    :ivar building_data: Sheet-specific data for 'Building Data', initialized for the given scenario.
    :type building_data: SheetData
    :ivar tech_data: Sheet-specific data for 'Technical Component Data', initialized for the given scenario.
    :type tech_data: SheetData
    :ivar economic_data: Sheet-specific data for 'Economic Data', initialized for the given scenario.
    :type economic_data: SheetData
    :ivar model_settings: Sheet-specific data for 'Model Settings', initialized for the given scenario.
    :type model_settings: SheetData
    """
    def __init__(self, excel_data: dict, scenario: str):
        self.scenario_name = scenario
        self.location = self._init_sheet_data(excel_data['Location'], scenario)
        self.consumption_data = self._init_sheet_data(excel_data['Consumption Data'], scenario)
        self.building_data = self._init_sheet_data(excel_data['Building Data'], scenario)
        self.tech_data = self._init_sheet_data(excel_data['Technical Component Data'], scenario)
        self.economic_data = self._init_sheet_data(excel_data['Economic Data'], scenario)
        self.model_settings = self._init_sheet_data(excel_data['Model Settings'], scenario)

    def _init_sheet_data(self, df: pd.DataFrame, sce_name: str):
        param_vals = self._check_datatypes(df['SCENARIO VALUES'][sce_name].copy(), df['PARAMETERS']['datatype'])
        return SheetData(**param_vals.to_dict())

    def _check_datatypes(self, sce_data: pd.Series, datatypes: pd.Series):
        for var, datatype in datatypes.items():
            try:
                if str(sce_data[var]).lower() == 'nan':
                    continue
                if datatype == 'float':
                    sce_data.loc[var] = float(sce_data[var])
                elif datatype == 'int':
                    sce_data.loc[var] = int(sce_data[var])
                elif datatype == 'bool':
                    sce_data.loc[var] = bool(sce_data[var])
                elif datatype == 'str':
                    sce_data.loc[var] = str(sce_data[var])

            except ValueError:
                raise ValueError(f"{var}: {sce_data[var]} cannot be converted to {datatype}")

        return sce_data


class SheetData():
    """
    Holds and processes data related to various component types and their attributes.

    The `SheetData` class is designed to dynamically assign attributes based on
    provided data, categorizing attributes into components and supporting
    their initialization. Parameters not related to known components are stored
    directly as attributes, while recognized component-related attributes are
    organized into dictionaries for easier management. The class also removes
    component data that is entirely empty or contains only `NaN` values for
    better organization and relevance.

    :ivar attribute1: Description of the attribute. The attribute can be any key
                      passed to the constructor that does not belong to a recognized
                      component type.
    :type attribute1: Any
    :ivar component_type: A dictionary of dictionaries, categorizing attributes
                          related to known components by their types and instances.
                          For example, categories such as walls, roofs, PVs, etc.,
                          create separate structured data storage.
    :type component_type: Dict[str, Dict[str, Dict[str, Any]]]
    """
    def __init__(self, **params):
        comp_types = {}
        for key, val in params.items():
            if key.split('_')[0][:-1] in ['wall', 'roof', 'pv', 'batt', 'st', 'hp', 'ac', 'deh', 'buffsto', 'dhwsto',
                                          'boiler', 'refurb', 'ev']:
                # set dicitionaries as properties for components,
                # so that they can be used directly for component initialisation
                comp_name = key.split('_')[0]
                if comp_name[:-1] not in comp_types:
                    comp_types[comp_name[:-1]] = {}
                if comp_name not in comp_types[comp_name[:-1]]:
                    comp_types[comp_name[:-1]][comp_name] = {}
                if val != 'nan':
                    comp_types[comp_name[:-1]][comp_name][key[len(comp_name)+1:]] = val
            else:
                # write parameter as class attribute
                setattr(self, key, val)

        for comp_type in comp_types:
            # erase empty data
            comps_to_delete = []
            for comp_inst in comp_types[comp_type]:
                if pd.Series(comp_types[comp_type][comp_inst]).isna().all():
                    comps_to_delete.append((comp_type, comp_inst))
            for key in comps_to_delete:
                comp_types[key[0]].pop(key[1])
            # write dictionary as class attribute
            setattr(self, comp_type, comp_types[comp_type])


class BuildingModelInput():
    """
    Represents an input model for building simulations derived from Excel files.

    This class is designed to read and process input data from the structured Excel input workbook
    used in the building optimization. It handles multiple sheets with specific configurations,
    validates the scenario data across sheets, and processes heat pump specifications (HP Specifications).
    The class ensures consistency in scenario data and prepares the data for further usage.

    :ivar fpath: Full file path to the input Excel file.
    :type fpath: str
    :ivar sheet_names: List of sheet names expected in the input Excel file.
    :type sheet_names: list[str]
    :ivar sheet_names_sce_data: List of sheet names containing scenario-related data.
    :type sheet_names_sce_data: list[str]
    :ivar excel_data: Dictionary containing processed data from each sheet in the input Excel file.
                      This excludes empty columns and fills missing data with placeholder values ('nan').
    :type excel_data: dict
    :ivar scenarios: List of scenario names extracted from the 'Location' sheet of the input file.
    :type scenarios: list[str]
    :ivar scenario_data: Dictionary holding `ScenarioData` objects for each scenario.
    :type scenario_data: dict
    :ivar hp_specs: Dictionary containing processed heat pump specifications indexed by heat pump type.
    :type hp_specs: dict
    """

    def __init__(self, fname: str):
        self.fpath = const.PATH_TO_WD + "/data/input/" + fname
        self.sheet_names = ['Location', 'Consumption Data', 'Building Data', 'Technical Component Data',
                            'Economic Data', 'Model Settings', 'HP Specifications']
        self.sheet_names_sce_data = ['Location', 'Consumption Data', 'Building Data', 'Technical Component Data',
                                     'Economic Data', 'Model Settings',]
        self.excel_data = pd.read_excel(
            self.fpath,
            sheet_name=['Location', 'Consumption Data', 'Building Data', 'Technical Component Data', 'Economic Data',
                        'Model Settings', 'HP Specifications'],
            header=[0, 1],
            index_col=[0],
        )
        self.excel_data = {
            sheet: self.excel_data[sheet].iloc[:, :13].dropna(how='all', axis=1).fillna('nan')  # max. 10 scenarios
            for sheet in self.excel_data
        }
        self.scenarios = self.excel_data['Location']['SCENARIO VALUES'].columns.to_list()

        # check if scenario columns match in all worksheets
        for sheet in self.sheet_names_sce_data:
            try:
                assert all(self.excel_data[sheet]['SCENARIO VALUES'].columns == self.scenarios), \
                    f"Scenarios must be the same for all sheets. Check for wrongly inserted data in excel input file." \
                    f"\n{sheet}: {self.excel_data[sheet]['SCENARIO VALUES'].columns.to_list()} != {self.scenarios}"
            except ValueError:
                raise ValueError(
                    f"Scenarios must be the same for all sheets. Check for wrongly inserted data in excel input file."
                    f"\n{sheet}: {self.excel_data[sheet]['SCENARIO VALUES'].columns.to_list()} != {self.scenarios}")

        self._init_scenario_data()
        self.hp_specs = self._set_hp_specs()

    def _init_scenario_data(self):

        self.scenario_data = {}

        for sce in self.scenarios:
            self.scenario_data[sce] = ScenarioData(self.excel_data, sce)

    def _set_hp_specs(self):
        hp_specs = {}
        for hp_type in self.excel_data['HP Specifications'].index.unique():
            hp_specs[hp_type] = self.excel_data['HP Specifications'].loc[hp_type]
            hp_specs[hp_type].index = pd.MultiIndex.from_tuples(hp_specs[hp_type]['INDEX'].values.tolist())
            hp_specs[hp_type] = hp_specs[hp_type]['VALUES']

        return hp_specs


class ScenarioResults:
    """
    Represents the results of one scenario of run_building_opt, handling time-series data,
    energy balances, and result summaries.

    The class processes inputs and model parameters, extracting various time-series datasets
    related to energy sources, sinks, storages, and other components. It computes yearly
    summaries and provides methods to query and filter the data for insights and analysis.

    The purpose of this class is to enable streamlined management and retrieval of simulation
    results, offering flexibility for further analysis and reporting.

    :ivar ts_inputs: Time-series input data used in simulation.
    :type ts_inputs: pd.DataFrame
    :ivar ts_el_sources: Time-series data for electrical energy sources.
    :type ts_el_sources: pd.DataFrame
    :ivar ts_el_sinks: Time-series data for electrical energy sinks.
    :type ts_el_sinks: pd.DataFrame
    :ivar ts_th_sources: Time-series data for thermal energy sources.
    :type ts_th_sources: pd.DataFrame
    :ivar ts_th_sinks: Time-series data for thermal energy sinks.
    :type ts_th_sinks: pd.DataFrame
    :ivar ts_storages: Time-series data for storages.
    :type ts_storages: pd.DataFrame
    :ivar ts_others: Time-series data for other parameters.
    :type ts_others: pd.DataFrame
    :ivar yearly_sums: Summarized yearly data grouped by energy sources, sinks, and others.
    :type yearly_sums: pd.DataFrame
    :ivar key_facts: Holds specific key insights or summaries extracted from the results. Default is None.
    :type key_facts: None or pd.DataFrame
    :ivar ts_el_balance: Represents the electrical energy balance time-series. Default is None.
    :type ts_el_balance: None or pd.DataFrame
    :ivar ts_th_balance: Represents the thermal energy balance time-series. Default is None.
    :type ts_th_balance: None or pd.DataFrame
    :ivar ti_components: Technical and investment-related data of components, extracted and structured.
    :type ti_components: pd.DataFrame
    """
    def __init__(self, ts_inputs: pd.DataFrame, m: ConcreteModel):
        self.ts_inputs = ts_inputs
        self._init_ti_components(m)
        self.ts_el_sources = pd.DataFrame(
            {str(comp): {(s, t): m.p_el_t_feed[comp, t, s].value for s in m.season for t in m.t}
             for comp in m.el_sources}
        ).sort_index()
        self.ts_el_sinks = pd.DataFrame(
            {str(comp): {(s, t): m.p_el_t_drain[comp, t, s].value for s in m.season for t in m.t}
             for comp in m.el_sinks}
        ).sort_index()
        self.ts_th_sources = pd.DataFrame(
            {f"{comp}:{temp.name}": {(s, t): m.p_th_t_feed[comp, temp, t, s].value for s in m.season for t in m.t}
             for comp in m.th_sources for temp in m.temp_levels}
        ).sort_index()
        self.ts_th_sinks = pd.DataFrame(
            {f"{comp}:{temp.name}": {(s, t): m.p_th_t_drain[comp, temp, t, s].value for s in m.season for t in m.t}
             for comp in m.th_sinks for temp in m.temp_levels}
        ).sort_index()
        self.ts_storages = pd.DataFrame(
            {str(comp): {(s, t): m.c_t[comp, t, s].value for s in m.season for t in m.t}
             for comp in m.storages}
        ).sort_index()
        self.ts_others = pd.DataFrame(
            {
                **{f"qmax_t: {st}: {temp.name}": st.qmax_t[temp] * self.ti_components.loc[str(st), 'p_inst']
                   for st in m.solar_thermal for temp in st.temp_levels_feed},
                **{f"pmax_t: {pv}": pv.potential.gen_ts_normalized *
                                    self.ti_components.loc[str(pv), 'p_inst'] * (1.0 + pv.pmax_at_lifeend)/2
                   for pv in m.pv },
                **{},
            }).sort_index()

        for imp in m.el_imports:
            self.ts_others[f"elec. price: {imp.name}"] =  imp.var_cost_t

        self.yearly_sums = pd.DataFrame({
            'sums_el_sources': self.ts_el_sources.apply(lambda x: x * m.cost_weights).sum().sort_index(),
            'sums_el_sinks': self.ts_el_sinks.apply(lambda x: x * m.cost_weights).sum().sort_index(),
            'sums_th_sources': self.ts_th_sources.apply(lambda x: x * m.cost_weights).sum().sort_index(),
            'sums_th_sinks': self.ts_th_sinks.apply(lambda x: x * m.cost_weights).sum().sort_index(),
            'sums_others': self.ts_others.apply(lambda x: x * m.cost_weights).sum().sort_index(),
        })
        self.key_facts = None
        self.ts_el_balance = None
        self.ts_th_balance = None

    def _init_ti_components(self, m: ConcreteModel):
        self.ti_components = pd.DataFrame(
            {str(var): {str(comp): var[comp].value for comp in m.components if comp in var}
             for var in
             [m.is_built, m.p_inst, m.c_inst, m.c_inst_l, m.inv_costs, m.fix_costs, m.var_costs, m.co2_costs,
              m.revenues, m.co2eq_balance]}
        ).sort_index()
        self.ti_components.loc[:, 'tot_costs'] = (self.ti_components.filter(regex='costs').sum(axis=1) -
                                                  self.ti_components.filter(regex='revenues').sum(axis=1))
        self.ti_components.loc['sum', :] = self.ti_components.sum()
        self.ti_components = self.ti_components[
            ['is_built', 'p_inst', 'c_inst', 'c_inst_l', 'inv_costs', 'fix_costs', 'var_costs', 'co2_costs',
             'revenues', 'tot_costs', 'co2eq_balance']]

    def find_vals(self, result_table_name, column_val: str, row_val: str = None) -> pd.DataFrame|pd.Series:
        """
        filters for values in the result tables (of type pd.DataFrame), supports regex
        :param result_table_name: needs to be an attribute of self
        :param first_val: first value to be filtered for (in column names)
        :param second_val: second value to be filtered for (in row names)
        :return: filtered values
        """

        filtered = getattr(self, result_table_name).filter(regex=column_val)
        if row_val is not None:
            filtered = filtered.filter(regex=row_val, axis=0)

        return filtered

    def get_ti_sum(self, col, row) -> float:
        return self.find_vals('ti_components', col, row).sum().sum()

    def get_en_balance(self, en_type: str, src_sink: str, search_val: str, season: Season = None):
        """
        Get the energy balance based on the specified energy type, source or sink,
        search value, and an optional season. This method provides the sum of energy
        values either for a specific season or throughout the year.

        :param en_type: Type of energy. Must be either 'el' (electric) or 'th' (thermal).
        :param src_sink: Determines whether to calculate for sources ('src') or sinks
            ('sink'). Must be either 'src' or 'sink'.
        :param search_val: The search value used to filter data for calculation.
        :param season: Optional. A season object that defines the time range for
            seasonal calculations. If not provided, yearly values are used.
        :return: The sum of energy values as a float based on the specified parameters.
        :rtype: float
        """
        assert en_type in ['el', 'th'], "en_type must be 'el' or 'th'"
        assert src_sink in ['src', 'sink'], "src_sink must be either 'src' or 'sink'"

        col = f"sums_{en_type}_{'sources' if src_sink == 'src' else 'sinks'}"

        if season is None:
            return self.find_vals('yearly_sums', col, search_val).sum().sum()
        else:
            return self.get_ts(en_type, src_sink, search_val, season=season).sum(axis=1).sum()

    def get_ts(self, en_type: str, src_sink: str, search_val: str, season: Season = None):
        """
        Retrieves a time series DataFrame filtered by the given search value. The function
        accesses the corresponding table for energy type ('el' or 'th') and source/sink,
        and optionally filters by a specific season if provided.

        :param en_type: A string specifying the energy type. Must be 'el' or 'th'.
        :param src_sink: A string indicating whether to access source or sink data.
            Must be either 'src' or 'sink'.
        :param search_val: A string or regex pattern used to filter the time series data.
        :param season: An optional `Season` object for filtering data by season.
            Defaults to None to include all seasons.
        :return: A filtered DataFrame based on the search value and optional season.
        :rtype: pandas.DataFrame
        """
        assert en_type in ['el', 'th'], "en_type must be 'el' or 'th'"
        assert src_sink in ['src', 'sink'], "src_sink must be either 'src' or 'sink'"

        res_table_name = f"ts_{en_type}_{'sources' if src_sink == 'src' else 'sinks'}"

        ts_df = getattr(self, res_table_name)

        if ts_df.empty:
            ts_df = pd.DataFrame(index=create_ts_multiindex())
        if season is not None:
            ts_df = ts_df.loc[season]

        return ts_df.filter(regex=search_val)

    def to_dict(self):

        return {k: v for k, v in self.__dict__.items() if v is not None}


def prepare_dummy_ts(data_columns: list, value=0.0, freq=None) -> Timeseries:
    """
    create a dummy instance for the Timeseries class
    :param data_columns:
    :param value:
    :param freq:
    :return:
    """
    if freq is None:
        if const.TS_PER_HOUR == 1:
            freq = const.TSStepSize.HOURLY
        else:
            freq = const.TSStepSize.QUARTER_HOURLY

    ts = Timeseries(freq=freq)
    ts[data_columns] = value

    return ts





