"""
src/data_preparation/timeseries_reduction.py

Utilities to prepare, aggregate and reduce time-series inputs used in building and
battery optimization workflows. This module contains functions to analyze climate
data and to assemble either full-resolution or reduced reference-week profiles
for downstream modelling.

Main responsibilities:

- climate_data_analysis(...):
    - Fetches or loads cached PVGIS climate data, computes daily aggregates
      (temperature averages, min/max, solar yield) and derived metrics
      (heating/cooling degree days).
    - Produces aggregated views by year, month, week and season, includes
      relative metrics and summary statistics, and optionally writes results to
      an Excel file.
    - Caches computed analysis via the project's pickle helpers.

- prepare_ts_profiles(...):
    - Builds model-ready input tables from a dictionary of Timeseries objects.
    - Supports two modes: full-resolution (no reduction) and reduced reference-
      week mode (per-season representative weeks with cost factors).
    - When reduction is enabled it selects representative weeks based on climate
      similarity metrics, assembles per-season reduced MultiIndex frames and
      fills them with climate, load, price and CO2 columns.
    - Returns a DataFrame of prepared/reduced inputs and a Series/DataFrame of
      selected reference weeks.

Behavior and notes:

- Integrates with project constants (`src.const`) and Timeseries classes
  (`src.data_preparation.data_formats`), relies on PVGIS climate fetchers and
  precomputed pickles for caching.
- Expects Timeseries inputs to have correct resolution (`TS_PER_HOUR * 24`).
- Caching and file I/O use `dump_pickle` / `load_pickle` helpers; callers should
  ensure correct `results_folder` when writing Excel outputs.
- Designed to be deterministic and reproducible for scenario runs; reduction
  logic stores selected weeks and reduced timeseries for reuse.

Dependencies:

- pandas, project `Timeseries` utilities, PVGIS climate helpers, and export
  utilities for optional xlsx output.
"""

import pandas as pd

from src import const
from src.const import Season, InputData
from src.data_preparation.demand_data import CO2ElecCols, ElPriceCols, DHWCols, ElLoadCols
from src.data_preparation.pvgis_climate_data import get_pvgis_hourly_data, PvgisDataCols
from src.data_preparation.data_formats import TimeseriesCols, Timeseries
from src.helper import (dump_pickle, create_ts_data_dump_id, create_ts_multiindex, load_pickle,
                             create_climate_data_dump_id)
from src.export.export_results import ClimateDataCols, write_climate_data_xlsx_analysis


def climate_data_analysis(write_xlsx=False, results_folder: str = None) -> dict[str, pd.DataFrame]:
    """
    Performs climate data analysis by processing and aggregating climate-related data. It calculates key
    metrics such as temperature averages, extremes, solar yield, cooling and heating degree days, and
    aggregates them across different temporal levels such as years, seasons, months, and weeks, providing
    insights into climate trends.

    :param write_xlsx: A boolean flag determining whether the resulting analysis should be saved
        as an Excel file.
    :param results_folder: A string specifying the directory in which the Excel file is to
        be saved; required if `write_xlsx` is set to True.
    :return: A dictionary where keys are aggregation levels (e.g., 'years', 'weeks', 'seasons') or
        summary types, and values are pandas DataFrames containing the aggregation results.
    """

    if write_xlsx:
        assert results_folder is not None, "results_folder needs to be given if write_xlsx is True"

    print("retrieving climate data...")

    try:
        climate_data_analysis = load_pickle(f"{create_climate_data_dump_id()}")
    except FileNotFoundError:
        daily_df = pd.DataFrame()

        pvgis_hourly_ts = get_pvgis_hourly_data(
            start_year=const.CLIMATE_DATA_YEARS[0], end_year=const.CLIMATE_DATA_YEARS[1])

        # Temp. avg./min./max.
        daily_df[ClimateDataCols.TEMP_AVG.value] = (
            pvgis_hourly_ts.groupby(TimeseriesCols.TSDay)[PvgisDataCols.Air_temp].mean())
        daily_df[ClimateDataCols.TEMP_MIN.value] = (
            pvgis_hourly_ts.groupby(TimeseriesCols.TSDay)[PvgisDataCols.Air_temp].min())
        daily_df[ClimateDataCols.TEMP_MAX.value] = (
            pvgis_hourly_ts.groupby(TimeseriesCols.TSDay)[PvgisDataCols.Air_temp].max())

        # solar yield
        daily_df[ClimateDataCols.P_SOLAR_SUM.value] = (
            pvgis_hourly_ts.groupby(TimeseriesCols.TSDay)[PvgisDataCols.P_el_t].sum())

        # calculate heating degree days
        heating_threshold = 15
        heating_temp = 18
        daily_df[ClimateDataCols.HDD_SUM.value] = daily_df.apply(
            lambda x: heating_temp - x[ClimateDataCols.TEMP_AVG]
            if x[ClimateDataCols.TEMP_AVG] <= heating_threshold else 0,
            axis=1)
        # calculate cooling degree days
        cooling_threshold = 24
        cooling_temp = 21
        daily_df[ClimateDataCols.CDD_SUM.value] = daily_df.apply(
            lambda x: x[ClimateDataCols.TEMP_AVG] - cooling_temp
            if x[ClimateDataCols.TEMP_AVG] >= cooling_threshold else 0,
            axis=1)

        daily_ts = Timeseries(daily_df, start_year=const.CLIMATE_DATA_YEARS[0], end_year=const.CLIMATE_DATA_YEARS[1],
                              freq=const.TSStepSize.DAILY)

        climate_data_analysis = {}

        for aggr in [TimeseriesCols.Year,
                     [TimeseriesCols.Year, TimeseriesCols.Month, TimeseriesCols.Season],
                     [TimeseriesCols.Year, TimeseriesCols.Week, TimeseriesCols.Season],
                     [TimeseriesCols.Year, TimeseriesCols.Season]]:

            # apply custom aggregation functions for groups
            agg_map = {
                ClimateDataCols.TEMP_AVG: "mean",
                ClimateDataCols.TEMP_MIN: "min",
                ClimateDataCols.TEMP_MAX: "max",
                ClimateDataCols.P_SOLAR_SUM: "sum",
                ClimateDataCols.HDD_SUM: "sum",
                ClimateDataCols.CDD_SUM: "sum",
            }
            aggr_df = daily_ts.df.groupby(aggr).agg(agg_map)
            aggr_df.columns = [col.value for col in aggr_df.columns]

            if aggr == TimeseriesCols.Year:
                # calculate relative values to avg. over years
                def rel_val_if_mean_not_zero(s: pd.Series):
                    if s.mean() == 0.0:
                        return 0.0
                    else:
                        return s/s.mean()

                aggr_df[ClimateDataCols.P_SOLAR_SUM_REL.value] = rel_val_if_mean_not_zero(
                    aggr_df[ClimateDataCols.P_SOLAR_SUM])
                aggr_df[ClimateDataCols.HDD_SUM_REL.value] = rel_val_if_mean_not_zero(
                    aggr_df[ClimateDataCols.HDD_SUM])
                aggr_df[ClimateDataCols.CDD_SUM_REL.value] = rel_val_if_mean_not_zero(
                    aggr_df[ClimateDataCols.CDD_SUM])
                # add average, min, max and standard deviation over all years
                aggr_df.loc['AVG.', :] = aggr_df.mean()
                aggr_df.loc['MIN.', :] = aggr_df.min()
                aggr_df.loc['MAX.', :] = aggr_df.max()
                aggr_df.loc['STD. DEV.', :] = aggr_df.std()
                climate_data_analysis['years'] = aggr_df

            elif aggr[1] == TimeseriesCols.Week or aggr[1] == TimeseriesCols.Month:
                # add weekly / monthly values
                climate_data_analysis[f"{aggr[1].value}s"] = aggr_df
                # add average, min, max and standard deviation for each week/month over all years
                metrics = ['AVG.', 'MIN.', 'MAX.', 'STD. DEV.']
                summary_df = pd.DataFrame(
                    columns=pd.MultiIndex.from_product([metrics, aggr_df.columns])
                )
                summary_df['AVG.'] = aggr_df.groupby(level=[aggr[1], aggr[2]]).mean()
                summary_df['MIN.'] = aggr_df.groupby(level=[aggr[1], aggr[2]]).min()
                summary_df['MAX.'] = aggr_df.groupby(level=[aggr[1], aggr[2]]).max()
                summary_df['STD. DEV.'] = aggr_df.groupby(level=[aggr[1], aggr[2]]).std()

                agg_map = {(metric, col.value): func for metric in metrics for col, func in agg_map.items()}
                summary_df = summary_df.groupby(level=aggr[1]).agg(agg_map)

                climate_data_analysis[f'{aggr[1].value}s summary'] = summary_df

            elif aggr[1] == TimeseriesCols.Season:
                # add season values
                weeks_per_season = (daily_ts.df.groupby([TimeseriesCols.Year, TimeseriesCols.Season])
                                    [ClimateDataCols.P_SOLAR_SUM].count() / 7)
                aggr_df[ClimateDataCols.P_SOLAR_SUM_WEEKLY.value] = aggr_df[ClimateDataCols.P_SOLAR_SUM] / weeks_per_season
                aggr_df[ClimateDataCols.HDD_WEEKLY.value] = aggr_df[ClimateDataCols.HDD_SUM] / weeks_per_season
                aggr_df[ClimateDataCols.CDD_WEEKLY.value] = aggr_df[ClimateDataCols.CDD_SUM] / weeks_per_season
                climate_data_analysis[f"seasons"] = aggr_df
                # add average, min, max and standard deviation for each season over all years
                summary_df = pd.DataFrame(
                    columns=pd.MultiIndex.from_product([['AVG.', 'MIN.', 'MAX.', 'STD. DEV.'], aggr_df.columns])
                )
                summary_df['AVG.'] = aggr_df.groupby(level=TimeseriesCols.Season).mean()
                summary_df['MIN.'] = aggr_df.groupby(level=TimeseriesCols.Season).min()
                summary_df['MAX.'] = aggr_df.groupby(level=TimeseriesCols.Season).max()
                summary_df['STD. DEV.'] = aggr_df.groupby(level=TimeseriesCols.Season).std()
                climate_data_analysis[f'seasons summary'] = summary_df

            else:
                raise ValueError("Invalid aggregation.")

    dump_pickle(f"{create_climate_data_dump_id()}", climate_data_analysis)

    if write_xlsx:
        write_climate_data_xlsx_analysis(climate_data_analysis, results_folder=results_folder)

    return climate_data_analysis


def prepare_ts_profiles(ts_inputs: dict, climate_data: dict = None, sel_weeks: pd.DataFrame = None,
                        apply_ts_reduction=None, ts_red_climate_data: dict=None,
                        ) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare time series profiles by either reducing them to reference weeks or using the full data set. The function can
    handle various time series data, including climate data, load profiles, and other relevant datasets. Depending on
    the `apply_ts_reduction` flag, this function either reduces the input time series for computational optimization or
    processes it fully. It also aligns timeseries data with the required time resolution constraints.

    :param ts_inputs: A dictionary of time series input data for a whole year, categorized by type.
        Each input time series must have the correct resolution.
    :type ts_inputs: dict
    :param climate_data: Optional climate data dictionary, which may be used to compute reduced
        time series if `sel_weeks` or `ts_red_climate_data` is not provided.
        Default is None.
    :type climate_data: dict, optional
    :param sel_weeks: Optional DataFrame indicating selected reference weeks for timeseries reduction.
        If provided, this overrides the need to calculate the reference weeks from `climate_data`.
        Default is None.
    :type sel_weeks: pd.DataFrame, optional
    :param apply_ts_reduction: Flag indicating whether to apply timeseries reduction. If None, a default value
        is fetched from the constants. Default is None.
    :type apply_ts_reduction: bool, optional
    :param ts_red_climate_data: Optional pre-calculated reduced climate data. If provided, it is used instead of
        deriving reduced data and reference weeks from scratch. Default is None.
    :type ts_red_climate_data: dict, optional
    :return: A tuple containing a DataFrame of processed or reduced timeseries inputs (e.g., load profiles, climate data,
        etc.) indexed appropriately and a Series giving the selected reference weeks. Both
        outputs are formatted for compatibility with downstream modeling or analysis tasks.
    :rtype: tuple[pd.DataFrame, pd.Series]
    """

    print("preparing timeseries profiles...")

    if apply_ts_reduction is None:
        apply_ts_reduction = const.APPLY_TIMESERIES_REDUCTION

    assert all([ts.ts_per_day == const.TS_PER_HOUR * 24 for ts in ts_inputs.values()]), \
        f"All timeseries need to have the correct resolution"

    # case 1: no timeseries reduction
    if not apply_ts_reduction:

        t_idx = create_ts_multiindex(apply_ts_reduction=False)
        ts_red_inputs = pd.DataFrame(index=t_idx)

        # climate data from pvgis
        pvgis_cols = ts_inputs[InputData.PvgisData].data_columns
        ts_red_inputs[pvgis_cols] = ts_inputs[InputData.PvgisData][pvgis_cols].values
        # add load and other data
        try:
            ts_red_inputs[ElLoadCols.TOTAL_HH] = \
                ts_inputs[InputData.ElDemand][ts_inputs[InputData.ElDemand].data_columns].sum(axis=1).values
            ts_red_inputs[ElLoadCols.TOTAL_EV_STAT] = (
                ts_inputs[InputData.StatEmobDemand][ElLoadCols.TOTAL_EV_STAT].values)
            for col in ts_inputs[InputData.FlexEmobDemand].data_columns:
                ts_red_inputs[col] = ts_inputs[InputData.FlexEmobDemand][col].values
            for col in ts_inputs[InputData.ElPrice].data_columns:
                ts_red_inputs[col] = ts_inputs[InputData.ElPrice][col].values
            ts_red_inputs[DHWCols.Load] = ts_inputs[InputData.DhwDdemand][DHWCols.Load].values
            ts_red_inputs[CO2ElecCols.Co2eq_lca] = ts_inputs[InputData.CO2perKwhEl][CO2ElecCols.Co2eq_lca].values

        except KeyError:
            pass

        # reduce data size
        ts_red_inputs = ts_red_inputs.apply(pd.to_numeric, downcast='float')

        # dummy parameters (needed for timeseries reduction, here only for compatibility)
        ts_red_inputs['cost_factors'] = pd.Series(1.0, index=ts_red_inputs.index)

    # case 2: timeseries reduction to reference weeks assigned for each season
    else:

        if sel_weeks is None and ts_red_climate_data is None:
            assert climate_data is not None, \
                "if ts_red_climate_data and sel_weeks are not given, climate_data needs to be given"

        # prepare empty indexed dataframes to be filled with reduced timeseries data
        reduced_t_idx = create_ts_multiindex(apply_ts_reduction=True)
        ts_red_inputs = pd.DataFrame(index=reduced_t_idx)

        # if reduced timeseries for climate data have already been calculated previously, use them, else calculate
        if ts_red_climate_data is None:

            if sel_weeks is None:
                weekly = pd.DataFrame()
                weekly['solar yield'] = climate_data['weeks'].loc[const.MODELED_YEAR][ClimateDataCols.P_SOLAR_SUM]
                weekly['HDD'] = climate_data['weeks'].loc[const.MODELED_YEAR][ClimateDataCols.HDD_SUM]
                weekly['el. price'] = (
                    ts_inputs[InputData.ElPrice].df.groupby([TimeseriesCols.Week, TimeseriesCols.Season])[
                        ElPriceCols.Wholesale].mean())

                # reference can be avg. of modelled year or avg. in climate data
                weekly_ref = pd.DataFrame()
                weekly_ref['solar yield'] = climate_data['seasons summary'][
                    ('AVG.', ClimateDataCols.P_SOLAR_SUM_WEEKLY)]
                weekly_ref['HDD'] = climate_data['seasons summary'][('AVG.', ClimateDataCols.HDD_WEEKLY)]
                weekly_ref['el. price'] = weekly['el. price'].groupby([TimeseriesCols.Season]).mean()

                # go through weekly values, drop weeks with two seasons and calculate values relative to reference
                last_week = 0
                for i, row in weekly.iterrows():
                    week = i[0]
                    seas = i[1]
                    if week == last_week:
                        weekly = weekly.drop(index=week)
                    else:
                        weekly.loc[i, :] = weekly.loc[i, :] / weekly_ref.loc[seas]

                    last_week = i[0]

                # calculate weighted mean squared error
                weights = weekly_ref / weekly_ref.sum()
                weekly['Error'] = ((weekly - 1) ** 2).mul(weights, level="season").mean(axis=1).astype(float)
                # for each season select week with least error
                sel_weeks = pd.DataFrame()
                sel_weeks['nr'] = weekly['Error'].groupby(level=TimeseriesCols.Season).idxmin().map(lambda x: x[0])
                # cost factors giving the number of weeks in a season are calculated
                sel_weeks['cost_factor'] = (climate_data['seasons summary'][('AVG.', ClimateDataCols.P_SOLAR_SUM)] /
                            climate_data['seasons summary'][('AVG.', ClimateDataCols.P_SOLAR_SUM_WEEKLY)])
                print("Reference weeks selected: \n", sel_weeks.to_string())

                # add sel weeks to climate data
                for col in [ClimateDataCols.P_SOLAR_SUM, ClimateDataCols.HDD_SUM, ClimateDataCols.CDD_SUM]:
                    sel = climate_data['weeks'].loc[const.MODELED_YEAR].loc[sel_weeks['nr']][col]
                    sel.index = sel.index.levels[1]
                    climate_data['seasons summary'][('SEL. WEEK', col.value)] = sel

                dump_pickle(create_climate_data_dump_id(), climate_data)

            for seas in Season:
                ts_red_inputs.loc[(seas, reduced_t_idx.levels[1]), ts_inputs[InputData.PvgisData].data_columns] = (
                    ts_inputs[InputData.PvgisData].filter(weeks=sel_weeks['nr'][seas], data_only=True)).values
                ts_red_inputs.loc[(seas, reduced_t_idx.levels[1]), 'cost_factors'] = sel_weeks['cost_factor'][seas]

            # store reduced timeseries for climate data for later use
            ts_red_climate_data = {
                'ts_red_inputs': ts_red_inputs,
                'sel_weeks': sel_weeks,
            }
            dump_pickle(create_ts_data_dump_id(), ts_red_climate_data)



        else:
            ts_red_inputs = ts_red_climate_data['ts_red_inputs']
            sel_weeks = ts_red_climate_data['sel_weeks']

        # complete reduced timeseries profiles with load data and others
        for seas in Season:
            try:
                idx = (seas, reduced_t_idx.levels[1])
                ts_red_inputs.loc[idx, ElLoadCols.TOTAL_HH] = ts_inputs[InputData.ElDemand].filter(
                    weeks=sel_weeks['nr'][seas], data_only=True).sum(axis=1).values
                ts_red_inputs.loc[idx, ElLoadCols.TOTAL_EV_STAT] = (ts_inputs[InputData.StatEmobDemand].filter(
                    weeks=sel_weeks['nr'][seas], data_only=True)[ElLoadCols.TOTAL_EV_STAT].values)
                for col in ts_inputs[InputData.FlexEmobDemand].data_columns:
                    ts_red_inputs.loc[idx, col] = ts_inputs[InputData.FlexEmobDemand].filter(
                    weeks=sel_weeks['nr'][seas], data_only=True)[col].values
                for col in ts_inputs[InputData.ElPrice].data_columns:
                    ts_red_inputs.loc[idx, col] = ts_inputs[InputData.ElPrice].filter(
                        weeks=sel_weeks['nr'][seas], data_only=True)[col].values
                ts_red_inputs.loc[idx, CO2ElecCols.Co2eq_lca] = ts_inputs[InputData.CO2perKwhEl].filter(
                        weeks=sel_weeks['nr'][seas], data_only=True)[CO2ElecCols.Co2eq_lca].values
                ts_red_inputs.loc[idx, DHWCols.Load] = ts_inputs[InputData.DhwDdemand].filter(
                    weeks=sel_weeks['nr'][seas], data_only=True)[DHWCols.Load].values
            except KeyError:
                pass

    # reduce data size
    ts_red_inputs = ts_red_inputs.apply(pd.to_numeric, downcast='float')

    return ts_red_inputs, sel_weeks


