"""
src/data_preparation/demand_data.py

Utilities to create and process electricity, domestic hot water (DHW), and
e-mobility demand time series as well as other timeseries (prices) used by the building and battery optimization
workflows.

Responsibilities:

- Load and process external resource CSVs and precomputed profiles (synPRO,
  DHW Calc, electricity prices, CO₂ intensity) and convert them into project
  `Timeseries` objects.
- Produce variable electricity price series with optional seasonal/time-based
  variable grid fees and VAT adjustments.
- Generate normalized household electrical demand profiles and aggregate
  building-level loads from household mixes.
- Compute DHW energy demand from precomputed volume/temperature profiles.
- Build static and flexible EV charging profiles (charge types: flat, fast,
  optimized) and extract flexible charging blocks for time-series reduction.
- Provide enums for common column names and grid-fee specifications.

Main functions:

- get_elec_co2_intensity(...) -> Timeseries
- get_var_elec_prices(...) -> Timeseries
- get_dhwcalc_series(...) -> Timeseries
- get_synpro_el_hh_profiles(...) -> Timeseries
- get_el_hh_profiles(...) -> Timeseries
- calc_flex_dem_from_ts_red_inputs(...) -> dict
- calc_emob_demands(...) -> tuple[Timeseries, Timeseries]

Dependencies and notes:

- Uses `Timeseries` from `src.data_preparation.data_formats` and project constants
  from `src.const` (e.g. `PATH_TO_WD`, `PRECALC_FOLDER`, `TS_PER_HOUR`).
- Some outputs are cached via `dump_pickle` / `load_pickle` to `PRECALC_FOLDER`.
- Expects resource files under `data/resources` relative to `const.PATH_TO_WD`.
- External data sources (e.g. electricitymaps, synPRO files) may be subject to
  their respective licensing and usage restrictions.
"""

import os
from enum import Enum
import math

import pandas as pd

from src import const

from src.data_preparation.data_formats import Timeseries
from src.helper import dump_pickle, load_pickle, Season, calc_hours_in_period, identify_blocks


class CO2ElecCols(str, Enum):
    Co2eq_lca = 'Carbon Intensity gCO₂eq/kWh (LCA)'
    Co2eq_direct = 'Carbon Intensity gCO₂eq/kWh (direct)'


class ElPriceCols(str, Enum):
    Wholesale = 'Wholesale market price EEX Day-Ahead (EUR/kWh)'
    VarGridFees = 'Variable grid fees acc. to EnWG 14a (EUR/kWh)'
    ImportDynTariff = 'Dynamic el. tariff w/o var. grid fees (EUR/kWh)'
    ImportVarGridFeeTariff = 'El. tariff w/ var. grid fees (EUR/kWh)'
    ImportDynAndVarGridFeeTariff = 'Dynamic el. tariff w/ var. grid fees (EUR/kWh)'


class DHWCols(str, Enum):
    Load = 'DHW'


class ElLoadCols(str, Enum):
    FULLTIME_EMPLOYEES = 'synPRO_el_2_fulltime_employees'
    OVER_65 = 'synPRO_el_2_persons_over65'
    FAMILY = 'synPRO_el_family'
    SINGLE_UNDER_30 = 'synPRO_el_single_person_under30'
    TOTAL_HH = 'total_el_hh_load'
    TOTAL_EV_STAT = 'total_ev_static_charging_load'

class VarGridFeeSpecs(str, Enum):
    HT_WINDOWS = 'ht_windows'
    HT_VAL = 'ht_value'
    LT_WINDOWS = 'lt_windows'
    LT_VAL = 'lt_value'

class EmobChargetype(str, Enum):
    flat = 'flat'
    fast = 'fast'
    optimized = 'optimized'


def get_elec_co2_intensity(folderpath=const.PATH_TO_WD + "/data/resources/co2_intensity_electricity_maps/",
                           default_val_lca=const.CO2_g_per_kWh_EL_LCA, default_val_direct=const.CO2_g_per_kWh_EL_DIR,
                           year=2023) -> Timeseries:
    """
    Calculates and returns the electricity CO2 intensity as a timeseries.
    The data can be downloaded from https://app.electricitymaps.com. However, its commercial use is forbidden.

    This function reads hourly CO2 intensity data for electricity from a predefined
    CSV file, processes it, and converts it into a timeseries object. The timeseries
    contains lifecycle CO2 equivalent and direct CO2 emissions data.
    If no file is given, the default values for hourly CO2 intensity are used.

    :return: Processed timeseries object containing CO2 intensity data for electricity.
    :rtype: Timeseries
    """

    try:
        filepath = folderpath + f"DE_{year}_hourly.csv"
        co2_per_kwh_hourly = pd.read_csv(filepath).iloc[:, 4:-1]
    except FileNotFoundError:
        print("No hourly CO2 intensity data found. Using default values.")
        co2_per_kwh_hourly = pd.DataFrame(index=range(8760), columns=[CO2ElecCols.Co2eq_lca, CO2ElecCols.Co2eq_direct])
        co2_per_kwh_hourly.loc[:, CO2ElecCols.Co2eq_lca] = default_val_lca
        co2_per_kwh_hourly.loc[:, CO2ElecCols.Co2eq_direct] = default_val_direct

    # plotting for debugging purposes
    # co2_per_kwh_hourly.hist()
    # plt.show()
    # plot_timeseries(co2_per_kwh_hourly)

    ts = Timeseries(co2_per_kwh_hourly[[CO2ElecCols.Co2eq_lca, CO2ElecCols.Co2eq_direct]]/1000)

    # ts.plot_typeday(CO2ElecCols.Co2eq_lca)

    return ts


def get_var_elec_prices(grid_fees_st=0.1, other_taxes_and_charges=0.05, vat=0.19, var_grid_fees=None, year=2023) -> Timeseries:
    """
    Compute variable electricity prices based on day-ahead wholesale prices, grid fees,
    other taxes and charges, as well as VAT. Optionally, seasonal and time-dependent
    variable grid fees can be applied.

    This function processes electricity price data for a given year, applies grid fees,
    and adjusts values based on provided parameters.

    :param grid_fees_st: Base grid fee in €/kWh. Default is 0.1.
    :param other_taxes_and_charges: Fixed value for other taxes and charges in €/kWh. Default is 0.05.
    :param vat: Value-added tax (VAT) represented as a decimal (e.g., 0.19 for 19%). Default is 0.19.
    :param var_grid_fees: Optional dictionary defining seasonal variable grid fees and their respective
        time windows for high and low tariffs, e.g.
        var_grid_fees={
            Season.Summer: {
                VarGridFeeSpecs.HT_WINDOWS: [(17,20)],
                VarGridFeeSpecs.HT_VAL: 0.2,
                VarGridFeeSpecs.LT_WINDOWS: [(11,15)],
                VarGridFeeSpecs.LT_VAL: 0.01,
            },
            Season.Transition: {
                VarGridFeeSpecs.HT_WINDOWS: [(17,20)],
                VarGridFeeSpecs.HT_VAL: 0.2,
                VarGridFeeSpecs.LT_WINDOWS: [],
                VarGridFeeSpecs.LT_VAL: 0.01,
            },
            Season.Winter: {
                VarGridFeeSpecs.HT_WINDOWS: [(6,8), (17,20)],
                VarGridFeeSpecs.HT_VAL: 0.2,
                VarGridFeeSpecs.LT_WINDOWS: [(23,3)],
                VarGridFeeSpecs.LT_VAL: 0.01,
            },
        }
    :param year: Year for which the electricity prices are computed. Default is 2023.
    :return: Timeseries object containing computed electricity prices with applied
        taxes, fees, and optional variable components.
    :rtype: Timeseries
    """

    hourly_day_ahead_prices =  pd.read_csv(const.PATH_TO_WD +
        f"/data/resources/energy_charts/energy-charts_Stromproduktion_und_Börsenstrompreise_in_Deutschland_{year}.csv",
                                           header=[1]).iloc[:35040]

    if var_grid_fees is not None:
        grid_fees = Timeseries(freq='15min')
        grid_fees['grid fees [€/kWh]'] = grid_fees_st
        for season in Season:
            # set high tariff
            idx_ht = grid_fees.filter(
                season=season, hour_periods=var_grid_fees[season][VarGridFeeSpecs.HT_WINDOWS]).index
            grid_fees.df.loc[idx_ht, 'grid fees [€/kWh]'] = \
                var_grid_fees[season][VarGridFeeSpecs.HT_VAL]
            # set low tariff
            idx_lt = grid_fees.filter(
                season=season, hour_periods=var_grid_fees[season][VarGridFeeSpecs.LT_WINDOWS]).index
            grid_fees.df.loc[idx_lt, 'grid fees [€/kWh]'] = \
                var_grid_fees[season][VarGridFeeSpecs.LT_VAL]

        grid_fees = grid_fees['grid fees [€/kWh]']

    else:
        grid_fees = grid_fees_st

    hourly_day_ahead_prices[ElPriceCols.Wholesale] = hourly_day_ahead_prices['Preis (EUR/MWh, EUR/tCO2)']*0.001
    hourly_day_ahead_prices[ElPriceCols.VarGridFees] = grid_fees
    hourly_day_ahead_prices[ElPriceCols.ImportDynTariff] = (
            hourly_day_ahead_prices[ElPriceCols.Wholesale] + grid_fees_st + other_taxes_and_charges)
    hourly_day_ahead_prices[ElPriceCols.ImportVarGridFeeTariff] = (
            hourly_day_ahead_prices[ElPriceCols.Wholesale].mean() + grid_fees + other_taxes_and_charges)
    hourly_day_ahead_prices[ElPriceCols.ImportDynAndVarGridFeeTariff] = (
            hourly_day_ahead_prices[ElPriceCols.Wholesale] + grid_fees + other_taxes_and_charges)

    # Apply VAT
    for col in [ElPriceCols.ImportDynTariff, ElPriceCols.ImportVarGridFeeTariff,
                ElPriceCols.ImportDynAndVarGridFeeTariff]:
        hourly_day_ahead_prices[col] += hourly_day_ahead_prices[col].apply(lambda x: max(0, vat*x))

    ts = Timeseries(hourly_day_ahead_prices[[col.value for col in ElPriceCols]], freq='15min')

    ts.convert_to_hourly()

    # ts.plot_typeday(ElPriceCols.DayAhead)

    return ts


def get_dhwcalc_series(nr_of_households, l_per_person_day, temp_dhw):
    """
    Computes a domestic hot water (DHW) time series based on the number of households, daily water
    consumption per person, and desired DHW temperature. The function loads a pre-computed DHW
    profile and scales it to energy demand according to the given parameters.

    The pre-computed load profiles have been prepared with DHW Calc from the University of Kassel.
    https://www.uni-kassel.de/maschinenbau/institute/thermische-energietechnik/fachgebiete/solar-und-anlagentechnik/downloads.html
    Pre-computed load profiles are available for 1 - 10 households a 2 persons, for an average hot water consumption of
    30,40 or 50 liters per person per day.

    :param nr_of_households: Number of households to consider for the DHW calculation.
    :type nr_of_households: int
    :param l_per_person_day: Daily water consumption per person in liters.
    :type l_per_person_day: float
    :param temp_dhw: Desired temperature of the domestic hot water in degrees Celsius.
    :type temp_dhw: float
    :return: Time series data representing the calculated DHW energy demand.
    :rtype: Timeseries
    """
    fname = f"DHW_{nr_of_households}HH_{int(l_per_person_day)}LpP_DHW.txt"
    dhw_profile = pd.read_csv(const.PATH_TO_WD + "/data/resources/DHW_calc/" + fname, header=None, dtype=float)
    dhw_profile.columns = ['DHW']

    # calculate energy demand
    dhw_demand_kwh_per_l = const.TH_CAP_WATER_Wh_per_kg_K / 1000 * (temp_dhw - const.TEMP_EARTH)
    dhw_profile *= dhw_demand_kwh_per_l

    ts = Timeseries(dhw_profile)
    # ts.plot_ts()
    # ts.plot_typeday('DHW')
    print('DHW demand: \n', ts.get_specs())

    return ts


def get_synpro_el_hh_profiles(hourly=True):
    """
    Retrieve and process synthetic electric household electricity load profiles.
    The load profiles have been prepared with the synPRO tool from Fraunhofer ISE https://synpro-lastprofile.de/

    This function processes synthetic electric household profiles for different exemplary users from CSV files in
    a specified directory. It reads the data, extracts the relevant electric power
    values, and stores them into a Timeseries object. Optionally, it converts the
    profiles to hourly intervals depending on the `hourly` parameter.

    :param hourly: Specifies whether the timeseries should be converted to hourly
                   intervals. Defaults to True.
    :type hourly: bool
    :return: A Timeseries object containing the processed household electric
             profiles.
    :rtype: Timeseries
    """
    folder = f"{const.PATH_TO_WD}/data/resources/synPRO_households/electric"

    ts = Timeseries(freq='15min')
    for fname in os.listdir(folder):
        df = pd.read_csv(folder + "/" + fname, header=[8], sep=';')
        pfname = fname.split('.')[0]
        ts[pfname] = df['P_el'].values

        # ts.plot_typeday(pfname, title=fname)

    if hourly:
        ts.convert_to_hourly()

    return ts




def get_el_hh_profiles(el_demand: float, nr_of_households: int, nr_of_household_types: dict = None) -> Timeseries:
    """
    returns a Timeseries containing load profiles for each user type.
    The profiles sum up to the demand of one household.
    Like that, in a later step a load profile for the building can be calculated by adding up
    exemplary daily profiles for each household.
    This leads to more homogenous load profiles with a rising number of modelled households.

    :param el_demand:
    :param nr_of_households:
    :param nr_of_household_types:
    :return:
    """
    if nr_of_household_types is None:
        nr_of_household_types = {
            ElLoadCols.FULLTIME_EMPLOYEES.value: nr_of_households,
            ElLoadCols.OVER_65.value: 0,
            ElLoadCols.FAMILY.value: 0,
            ElLoadCols.SINGLE_UNDER_30.value: 0,
        }

    assert sum(nr for nr in nr_of_household_types.values()) == nr_of_households, \
        "nr of households for different user types doesn't match total number of households"
    assert all(float(nr).is_integer() for nr in nr_of_household_types.values()), \
        "nr_of_synPro_hh_1 .. 4 have to be integers"

    if os.path.exists(const.PRECALC_FOLDER + "synpro_el_hh_normalized.pkl"):
        ts_std = load_pickle("synpro_el_hh_normalized")
    else:
        ts_std = get_synpro_el_hh_profiles(hourly=True)
        # normalize
        ts_std.normalize()
        # store for later usage
        dump_pickle("synpro_el_hh_normalized", ts_std)

    # apply user shares and normalize on demand per household
    ts = Timeseries()
    for hh_type, hh_count in nr_of_household_types.items():
        for i in range(int(hh_count)):
            ts[f"{hh_type}_{i+1}"] = ts_std[hh_type] * 1/nr_of_households * el_demand

    return ts


def calc_flex_dem_from_ts_red_inputs(ts_red_inputs: pd.DataFrame) -> dict:
    """
    Calculates flexible energy demands from a given charging data DataFrame.

    This function processes the input DataFrame containing flexible charging demands for
    electric vehicles (EVs). It identifies charging blocks for each EV using their
    charging status, and computes the demand and associated periods for each block.
    The results are then aggregated into a dictionary.

    :param ts_red_inputs: A pandas DataFrame containing flexible charging data. The column
                       names should follow the format "<vehicle_id>_<metric>", where
                       <metric> includes "charging_status" and "delivered_energy".
    :type ts_red_inputs: pd.DataFrame
    :return: A dictionary where each key corresponds to a charging block in the format
             "<vehicle_id>_<block_id>", and the value is another dictionary with
             'period' (indices of periods in the block) and 'demand' (energy delivered
             in the block).
    :rtype: dict
    """

    # get list of vehicles with flexible charging demands
    evs = list(set([col.split("_")[1] for col in ts_red_inputs.columns if col.split("_")[0] == "flex"] ))

    flex_demands = {}
    for ev in evs:
        charge_block_idx = identify_blocks(ts_red_inputs[f"flex_{ev}_charging_status"])
        for block, idx in charge_block_idx.items():
            flex_demands[f"{ev}_{block}"] = {
                'period': idx,
                'demand': ts_red_inputs.loc[idx[-1], f"flex_{ev}_delivered_energy"],
            }

    return flex_demands


def calc_emob_demands(ev_data: dict) -> tuple[Timeseries, Timeseries]:
    """
    Calculate e-mobility charging demands based on electric vehicle data.

    This function processes electric vehicle (EV) data to calculate both static (fixed)
    and flexible charging demands over specified charging periods. It computes the charging
    statuses and energy delivered for different charge types flat, fast, and optimized
    charging. Depending on the charging type, it determines whether the charging energy
    distribution is flexible or fixed during the charging period.

    :param ev_data: Dictionary containing electric vehicle data. The data includes details like
        charging type, driving distance, energy consumption rate (kWh/100km), charging periods,
        and other relevant parameters for charging demands of each vehicle, e.g.
        {
            'ev1': {
                'max_p': 20,
                'period': '18-7',
                'drivedist': 50,
                'kwhper100km': 20,
                'daysinweek': "1,2,3,4,5",
                'chargetype': 'optimized'
            },
            'ev2': {... above params must be given...},
        }
    :return: A tuple of two Timeseries objects:
        - The first Timeseries object contains static charging demands, with details of
          non-flexible loads and energy consumption during specified periods.
        - The second Timeseries object includes flexible charging demands, incorporating
          the capability to distribute charging energy more flexibly across the periods.
    :rtype: tuple[Timeseries, Timeseries]
    """

    stat_demands = Timeseries()
    flex_demands = Timeseries()

    for ev in ev_data:
        ts = Timeseries()
        charge_start = int(ev_data[ev]['period'].split('-')[0])
        charge_end = int(ev_data[ev]['period'].split('-')[1])
        ch_period_len = calc_hours_in_period((charge_start, charge_end))
        ch_energy = (ev_data[ev]['drivedist']
                     *ev_data[ev]['kwhper100km']/100)
        daysofweek = [int(dow)-1 for dow in str(ev_data[ev]['daysinweek']).split(',')]
        idx_ch_start = stat_demands.filter(hour_periods=[(charge_start, charge_start+1)], daysofweek=daysofweek).index

        ts[f"{ev}_charging_status"] = 0
        for i_start in idx_ch_start:
            i_end = min(i_start+ch_period_len, ts.df.shape[0]-1)
            ts.df.loc[i_start:i_end, f"{ev}_charging_status"] = 1

        # get the indices for each charging block
        charge_block_idx = identify_blocks(ts[f"{ev}_charging_status"])

        # optimized charging: energy has to be delivered somewhere in period, but is flexible when
        if ev_data[ev]['chargetype'] == EmobChargetype.optimized.value:
            flex_demands[f"flex_{ev}_charging_status"] = ts[f"{ev}_charging_status"]
            flex_demands[f"flex_{ev}_delivered_energy"] = 0
            for block, idx in charge_block_idx.items():
                flex_demands.df.loc[idx[-1], f"flex_{ev}_delivered_energy"] = ch_energy

        # flat charging: constant charging throughout charing period
        elif ev_data[ev]['chargetype'] == EmobChargetype.flat.value:
            stat_demands[f"{ev}_charging_status"] = ts[f"{ev}_charging_status"]
            stat_demands[f"{ev}_load"] = 0
            for block, idx in charge_block_idx.items():
                stat_demands.df.loc[idx, f"{ev}_load"] = ch_energy/ch_period_len

        # fast charging: charing with full power until full
        elif ev_data[ev]['chargetype'] == EmobChargetype.fast.value:
            stat_demands[f"{ev}_charging_status"] = ts[f"{ev}_charging_status"]
            stat_demands[f"{ev}_load"] = 0
            ch_power = ev_data[ev]['max_p']
            max_ch_steps = math.floor(ch_energy / ch_power * const.TS_PER_HOUR)
            ch_power_rest = ch_energy * const.TS_PER_HOUR - max_ch_steps * ch_power
            for block, idx in charge_block_idx.items():
                max_p_idx = [t for t in range(idx[0], idx[0] + max_ch_steps)]
                rest_idx = [idx[0] + max_ch_steps]
                stat_demands.df.loc[max_p_idx, f"{ev}_load"] = ch_power
                stat_demands.df.loc[rest_idx, f"{ev}_load"] = ch_power_rest

        else:
            raise AssertionError(f'Invalid charge type. Not in {[e.value for e in EmobChargetype]}')

    stat_demands[ElLoadCols.TOTAL_EV_STAT] = stat_demands[
        [col for col in stat_demands.data_columns if col.split('_')[1] == 'load']].sum(axis=1)

    return stat_demands, flex_demands


if __name__ == "__main__":
    get_elec_co2_intensity()
    # price_ts = get_var_elec_prices(
    #     var_grid_fees={
    #         Season.Summer: {
    #             VarGridFeeSpecs.HT_WINDOWS: [(17,20)],
    #             VarGridFeeSpecs.HT_VAL: 2.0,
    #             VarGridFeeSpecs.LT_WINDOWS: [(11,15)],
    #             VarGridFeeSpecs.LT_VAL: 0.1,
    #         },
    #         Season.Transition: {
    #             VarGridFeeSpecs.HT_WINDOWS: [(17,20)],
    #             VarGridFeeSpecs.HT_VAL: 2.0,
    #             VarGridFeeSpecs.LT_WINDOWS: [],
    #             VarGridFeeSpecs.LT_VAL: 0.1,
    #         },
    #         Season.Winter: {
    #             VarGridFeeSpecs.HT_WINDOWS: [(6,8), (17,20)],
    #             VarGridFeeSpecs.HT_VAL: 2.0,
    #             VarGridFeeSpecs.LT_WINDOWS: [(23,3)],
    #             VarGridFeeSpecs.LT_VAL: 0.1,
    #         },
    #     }
    # )
    # price_ts = pd.DataFrame()
    # for year in [2019,2020,2021,2022,2023,2024]:
    #     price_ts[year] = get_var_elec_prices(year=year)[ElPriceCols.FeedIn]
    # price_ts.astype(float).boxplot(column=price_ts.columns.tolist())
    # plt.show()

    # get_dhwcalc_series(nr_of_households=10, l_per_person_day=30,temp_dhw=65)