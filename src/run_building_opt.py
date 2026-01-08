"""
src/run_building_opt.py

This module prepares input data for a building energy system optimization,
constructs component objects, initializes and solves a Pyomo optimization model,
and exports the results.

Main responsibilities:

- prepare_input_data(sce_data, climate_data):
    Processes scenario and climate inputs to produce reduced time series (electricity
    prices, household loads, PV/climate series), computes potentials and temperature
    levels, assembles demands, imports/exports, links and technology objects, and
    returns a components dictionary plus reduced time series.

- run_scenario(components, cost_weight_factors, co2_price):
    Validates components, initializes the Pyomo ConcreteModel via init_pyomo_model,
    solves the model (supports NEOS/CPLEX or local solvers), logs infeasible
    constraints to `infeasile.log`, and returns the solved model.

Script entrypoint (when run as __main__):

- Loads BuildingModelInput and scenarios from an Excel file.
- Applies global constants from scenario settings.
- Performs climate analysis and timeseries preparation.
- Builds components, runs the optimization for each scenario,
  and writes result files (Excel, logs, pickles).

Notes:

- Uses timeseries reduction and supports dynamic tariffs, variable grid fees,
  multi-temperature storages, heat pumps, PV, batteries, and other technologies.
- Outputs include scenario result workbooks and a pickled results object
  for postprocessing.
"""

import os
import sys
from datetime import datetime
import logging
from contextlib import redirect_stdout, redirect_stderr

from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.environ import ConcreteModel, SolverManagerFactory

from src.model.components import *
from src.const import Season
from src.data_preparation.demand_data import (get_elec_co2_intensity, get_var_elec_prices, get_dhwcalc_series,
                                              DHWCols, ElPriceCols, CO2ElecCols, get_el_hh_profiles, ElLoadCols,
                                              calc_emob_demands, calc_flex_dem_from_ts_red_inputs, VarGridFeeSpecs)
from src.data_preparation.pvgis_climate_data import get_pvgis_hourly_data, PvgisDataCols
from src.data_preparation.timeseries_reduction import prepare_ts_profiles, InputData, climate_data_analysis
from src.data_preparation.data_formats import BuildingModelInput, ScenarioData
from src.helper import load_pickle, dump_pickle, calc_th_demand_summary, check_if_zero_or_nan, \
    create_result_folder, create_ts_data_dump_id, tuple_list_from_hour_periods, Tee
from src.model.pyomo_model import check_components, init_pyomo_model
from src.export.analysis_plots import *
from src.export.excel_generator import save_xlsx_wb
from src.export.export_results import write_scenario_results, write_results_summary, write_climate_data_xlsx_analysis


def prepare_input_data(sce_data: ScenarioData, climate_data: dict) -> tuple[dict, pd.DataFrame]:
    """
    Prepare input data required for a simulation or analysis. This function processes
    scenario data and climate data to generate the necessary timeseries and input
    parameters, including electricity prices, demands, climate-based timeseries,
    temperature levels, and potentials. It handles exceptions for missing data
    and ensures that all necessary values are computed or retrieved.

    :param sce_data: Scenario data object containing consumption, economic, building,
                     and technical specifications.
    :type sce_data: ScenarioData
    :param climate_data: Dictionary of climate data used to compute temperature and
                         other climate-related timeseries.
    :type climate_data: dict
    :return: A tuple containing:
                 1. A dictionary of input data for the energy system components.
                 2. A DataFrame of (reduced) climate timeseries inputs.
    :rtype: tuple[dict, pd.DataFrame, pd.DataFrame]
    """

    # prepare timeseries for electricity import prices
    # (including variable prices from Day Ahead market and variable grid fees)
    var_elec_prices = get_var_elec_prices(
        grid_fees_st=float(sce_data.economic_data.grid_fees_st)/100,
        var_grid_fees={
            Season.Summer: {
                VarGridFeeSpecs.HT_WINDOWS:
                    tuple_list_from_hour_periods(sce_data.economic_data.tvg_ht_windows_summer),
                VarGridFeeSpecs.HT_VAL: float(sce_data.economic_data.tvg_ht_summer)/100,
                VarGridFeeSpecs.LT_WINDOWS:
                    tuple_list_from_hour_periods(sce_data.economic_data.tvg_lt_windows_summer),
                VarGridFeeSpecs.LT_VAL: float(sce_data.economic_data.tvg_lt_summer)/100,
            },
            Season.Transition: {
                VarGridFeeSpecs.HT_WINDOWS:
                    tuple_list_from_hour_periods(sce_data.economic_data.tvg_ht_windows_transition),
                VarGridFeeSpecs.HT_VAL: float(sce_data.economic_data.tvg_ht_transition)/100,
                VarGridFeeSpecs.LT_WINDOWS:
                    tuple_list_from_hour_periods(sce_data.economic_data.tvg_lt_windows_transition),
                VarGridFeeSpecs.LT_VAL: float(sce_data.economic_data.tvg_lt_transition)/100,
            },
            Season.Winter: {
                VarGridFeeSpecs.HT_WINDOWS:
                    tuple_list_from_hour_periods(sce_data.economic_data.tvg_ht_windows_winter),
                VarGridFeeSpecs.HT_VAL: float(sce_data.economic_data.tvg_ht_winter)/100,
                VarGridFeeSpecs.LT_WINDOWS:
                    tuple_list_from_hour_periods(sce_data.economic_data.tvg_lt_windows_winter),
                VarGridFeeSpecs.LT_VAL: float(sce_data.economic_data.tvg_lt_winter)/100,
            },
        }
    )
    emob_fix, emob_flex = calc_emob_demands(sce_data.consumption_data.ev)

    # prepare other timeseries data
    try:
        # use previously calculated timeseries for climate data...
        ts_red_climate_data = load_pickle(fname=create_ts_data_dump_id())
        ts_inputs = {
            InputData.DhwDdemand: get_dhwcalc_series(nr_of_households=const.NR_OF_HOUSEHOLDS,
                                                     l_per_person_day=sce_data.consumption_data.l_per_person_day,
                                                     temp_dhw=sce_data.consumption_data.temp_dhw),
            InputData.ElDemand: get_el_hh_profiles(
                el_demand=sce_data.consumption_data.el_demand,
                nr_of_households=const.NR_OF_HOUSEHOLDS,
                nr_of_household_types={
                    ElLoadCols.FULLTIME_EMPLOYEES.value: sce_data.consumption_data.nr_of_synPro_hh_1,
                    ElLoadCols.OVER_65.value: sce_data.consumption_data.nr_of_synPro_hh_2,
                    ElLoadCols.FAMILY.value: sce_data.consumption_data.nr_of_synPro_hh_3,
                    ElLoadCols.SINGLE_UNDER_30.value: sce_data.consumption_data.nr_of_synPro_hh_4,
                }
            ),
            InputData.StatEmobDemand: emob_fix,
            InputData.FlexEmobDemand: emob_flex,
            InputData.CO2perKwhEl: get_elec_co2_intensity(year=const.MODELED_YEAR),
            InputData.ElPrice: var_elec_prices,
        }
        ts_red_inputs, sel_weeks = prepare_ts_profiles(
            ts_inputs, ts_red_climate_data=ts_red_climate_data)

    except FileNotFoundError:
        # ... or prepare all timeseries data including climate data
        ts_inputs = {
            InputData.PvgisData: get_pvgis_hourly_data(),
            InputData.DhwDdemand: get_dhwcalc_series(nr_of_households=const.NR_OF_HOUSEHOLDS,
                                                     l_per_person_day=sce_data.consumption_data.l_per_person_day,
                                                     temp_dhw=sce_data.consumption_data.temp_dhw),
            InputData.ElDemand: get_el_hh_profiles(
                el_demand=sce_data.consumption_data.el_demand,
                nr_of_households=const.NR_OF_HOUSEHOLDS,
                nr_of_household_types={
                    ElLoadCols.FULLTIME_EMPLOYEES.value: sce_data.consumption_data.nr_of_synPro_hh_1,
                    ElLoadCols.OVER_65.value: sce_data.consumption_data.nr_of_synPro_hh_2,
                    ElLoadCols.FAMILY.value: sce_data.consumption_data.nr_of_synPro_hh_3,
                    ElLoadCols.SINGLE_UNDER_30: sce_data.consumption_data.nr_of_synPro_hh_4,
                }
            ),
            InputData.StatEmobDemand: emob_fix,
            InputData.FlexEmobDemand: emob_flex,
            InputData.CO2perKwhEl: get_elec_co2_intensity(),
            InputData.ElPrice: var_elec_prices,
        }
        ts_red_inputs, sel_weeks = prepare_ts_profiles(ts_inputs, climate_data=climate_data)

    # potentials
    potentials = {
        f"{key}": SolarPotential(f"{key}", sel_weeks=sel_weeks, **val)
        for key, val in {**sce_data.building_data.roof, **sce_data.building_data.wall}.items()
    }

    # demands / import / export
    temp_levels = {
        'air': TempLevel('air', ts_red_inputs[PvgisDataCols.Air_temp].mean(),
                         temp_t=ts_red_inputs[PvgisDataCols.Air_temp]),
        'roomheat': TempLevel('roomheat', const.NOM_TEMP_INDOOR),
        'ht_flow': FlowTempLevel('ht_flow',
                                 flow_temp=sce_data.consumption_data.flow_temp_ref,
                                 return_temp=30 + (sce_data.consumption_data.flow_temp_ref - 30) * 0.6,
                                 apply_heating_curve=True,
                                 temp_air_t=ts_red_inputs[PvgisDataCols.Air_temp], outtemp_min_ref=-14,
                                 flow_temp_min=25),
        'ht_flow_high': FlowTempLevel('ht_flow_high',
                                      flow_temp=sce_data.consumption_data.flow_temp_ref,
                                      return_temp=30 + (sce_data.consumption_data.flow_temp_ref - 30) * 0.6,
                                      apply_heating_curve=True,
                                      temp_air_t=ts_red_inputs[PvgisDataCols.Air_temp], outtemp_min_ref=-14,
                                      flow_temp_min=25,
                                      raised_storage_temp=sce_data.tech_data.buffsto['buffsto1'].get('max_temp')),
        'dhw': FlowTempLevel('dhw', flow_temp=sce_data.consumption_data.temp_dhw, return_temp=const.TEMP_EARTH,
                             apply_heating_curve=False),
    }

    assert sce_data.building_data.transm_loss or sce_data.consumption_data.heat_demand, \
        "either transmission losses or yearly heat demand need to be given"

    if check_if_zero_or_nan(sce_data.building_data.transm_loss):
        sce_data.building_data.transm_loss = 1

    dem = {
        'el_hh': ElDemand('el_hh', ts_red_inputs[ElLoadCols.TOTAL_HH]),
        'emob_fix': ElDemand('emob_fix', ts_red_inputs[ElLoadCols.TOTAL_EV_STAT]),
        'emob_flex': FlexElDemand('emob_flex', flex_demands=calc_flex_dem_from_ts_red_inputs(ts_red_inputs)),
        'th_roomheat': RoomheatDemand('th_roomheat',
                                      temp_levels_drain=[temp_levels['roomheat']],
                                      indoor_temp=sce_data.consumption_data.avg_indoor_temp,
                                      indoor_temp_night=sce_data.consumption_data.avg_indoor_temp_night,
                                      temp_air_t=ts_red_inputs[PvgisDataCols.Air_temp],
                                      vent_rate=sce_data.building_data.vent_rate,
                                      air_change=sce_data.building_data.air_change,
                                      transm_loss=sce_data.building_data.transm_loss,
                                      use_dayavg_heating_threshold=(1 if sce_data.consumption_data.heating_threshold
                                                                    else 0),
                                      ),
        'th_dhw': ThDemand('th_dhw', ts_red_inputs[DHWCols.Load], temp_levels_drain=[temp_levels['dhw']], ),

    }

    assert sce_data.economic_data.el_hh_is_dynamic_tariff or sce_data.economic_data.el_hh_kwh_rate != 'nan', \
        "el_hh_kwh_rate needs to be given or el_hh_is_dynamic_tariff must be enabled"
    assert sce_data.economic_data.el_hp_is_dynamic_tariff or sce_data.economic_data.el_hp_kwh_rate != 'nan', \
        "el_hp_kwh_rate needs to be given or el_hp_is_dynamic_tariff must be enabled"

    # calculate series for electricity prices: el_hh, el_hp, el_emob
    el_prices = {}
    for grid_conn in ['el_hh', 'el_hp', 'el_emob']:
        el_prices[grid_conn] = sce_data.economic_data.__getattribute__(f"{grid_conn}_kwh_rate") /100
        if (sce_data.economic_data.__getattribute__(f"{grid_conn}_is_dynamic_tariff")
            and not sce_data.economic_data.__getattribute__(f"{grid_conn}_var_grid_fees")):
            el_prices[grid_conn] += ts_red_inputs[ElPriceCols.ImportDynTariff]
        elif (sce_data.economic_data.__getattribute__(f"{grid_conn}_is_dynamic_tariff")
              and sce_data.economic_data.__getattribute__(f"{grid_conn}_var_grid_fees")):
            el_prices[grid_conn] += ts_red_inputs[ElPriceCols.ImportDynAndVarGridFeeTariff]
        elif (not sce_data.economic_data.__getattribute__(f"{grid_conn}_is_dynamic_tariff")
              and sce_data.economic_data.__getattribute__(f"{grid_conn}_var_grid_fees")):
            el_prices[grid_conn] += ts_red_inputs[ElPriceCols.ImportVarGridFeeTariff]
        else:
            pass

    imp = {
        'el_std': ElImport(
            'el_std',
            var_cost_t=el_prices['el_hh'],
            fix_cost_yearly= float(sce_data.economic_data.el_hh_basic_charge) * 12,
            co2eq_per_kwh_t=ts_red_inputs[CO2ElecCols.Co2eq_lca],
            p_max=sce_data.consumption_data.p_el_imp_max,
        ),
        'el_hp': ElImport(
            'el_hp',
            var_cost_t=el_prices['el_hp'],
            fix_cost_yearly=sce_data.economic_data.el_hp_basic_charge * 12,
            co2eq_per_kwh_t=ts_red_inputs[CO2ElecCols.Co2eq_lca],
            p_max=sce_data.consumption_data.p_el_imp_max,
        ),
        'el_emob': ElImport(
            'el_emob',
            var_cost_t=el_prices['el_emob'],
            fix_cost_yearly=sce_data.economic_data.el_emob_basic_charge * 12,
            co2eq_per_kwh_t=ts_red_inputs[CO2ElecCols.Co2eq_lca],
            p_max=sce_data.consumption_data.p_el_imp_max,
        ),
        'air': ThImport('th_air', temp_levels_feed=[temp_levels['air']]),
        'heat_gains': HeatGains(
            'heat_gains',
            temp_levels_feed=[temp_levels['roomheat']],
            demand=dem['th_roomheat'],
            internal_gains=sce_data.building_data.internal_gains,
            solar_potentials=[potentials[pot] for pot in potentials]
        )

    }

    if not check_if_zero_or_nan(sce_data.consumption_data.heat_demand):
        th_dem_summary = calc_th_demand_summary(dem, imp, cost_factors=ts_red_inputs[InputData.CostFactors])
        dem_factor = sce_data.consumption_data.heat_demand / th_dem_summary['sum'].sum()
        sce_data.building_data.transm_loss *= dem_factor
        dem['th_roomheat'] = RoomheatDemand(
            'th_roomheat',
            temp_levels_drain=[temp_levels['roomheat']],
            indoor_temp=sce_data.consumption_data.avg_indoor_temp,
            indoor_temp_night=sce_data.consumption_data.avg_indoor_temp_night,
            temp_air_t=ts_red_inputs[PvgisDataCols.Air_temp],
            transm_loss=sce_data.building_data.transm_loss,
            vent_rate=sce_data.building_data.vent_rate,
            air_change=sce_data.building_data.air_change,
            use_dayavg_heating_threshold=1 if sce_data.consumption_data.heating_threshold else 0,
        )

    # print out sums of demands for checking
    print('Electrical demand in kWh: ', (dem['el_hh'].p_t * ts_red_inputs[InputData.CostFactors]).sum())
    print('Thermal demand in kWh: \n',
          calc_th_demand_summary(dem, imp, cost_factors=ts_red_inputs[InputData.CostFactors]).sum().to_string())

    feedin_prices = pd.Series(index=ts_red_inputs.index)
    if sce_data.economic_data.el_feedin_is_directly_marketed:
        feedin_prices = -ts_red_inputs[ElPriceCols.Wholesale]
    else:
        feedin_prices.loc[:] = -sce_data.economic_data.el_feedin_kwh_rate/100
        # no feed-in remuneration at negative wholesale prices
        feedin_prices[ts_red_inputs[ElPriceCols.Wholesale] < 0] = 0
    exp = {
        'el_pv': ElExport(
            'el_pv',
            var_cost_t=feedin_prices
        ),
        'excess_heat': ThExport('excess_heat', temp_levels_drain=[temp_levels['roomheat']], var_cost_t=0.0)
    }

    assert sum([sce_data.economic_data.el_hh_grid_to_battery, sce_data.economic_data.el_hp_grid_to_battery,
                sce_data.economic_data.el_emob_grid_to_battery]) <= 1, \
        "battery can only be charged from one grid connection"

    links = {
        'el_pv-el_imp': ElectricLine('el_pv-el_imp'),
        'el_imp-el_pv': ElectricLine('el_imp-el_pv',
                                     p_max=const.MAX_P * sce_data.economic_data.el_hh_grid_to_battery, eff=0.99),
        'el_hp-el_pv': ElectricLine('el_hp-el_pv',
                                    p_max=const.MAX_P * sce_data.economic_data.el_hp_grid_to_battery, eff=0.99),
        'el_emob-el_pv': ElectricLine('el_emob-el_pv',
                                      p_max=const.MAX_P * sce_data.economic_data.el_emob_grid_to_battery, eff=0.99),
        'el_pv-el_hp': ElectricLine('el_pv-el_hp'),
        'el_pv-el_emob': ElectricLine('el_pv-el_emob'),
        'th_ht_flow-th_roomheat': HeatExchanger('th_ht_flow-th_roomheat',
                                                temp_levels_drain=[temp_levels['ht_flow']],
                                                temp_levels_feed=[temp_levels['roomheat']]),
        'th_ht_flow_high-th_roomheat': HeatExchanger('th_ht_flow_high-th_roomheat',
                                                     temp_levels_drain=[temp_levels['ht_flow_high']],
                                                     temp_levels_feed=[temp_levels['roomheat']]),
    }

    # technologies
    pv = {
        f"{key}": PV(f"{key}", potentials[f"{val.pop('mount_comp')}"], **val)
        for key, val in sce_data.tech_data.pv.items()
    }
    st = {
        f"{key}": SolarThermal(
            f"{key}", potentials[f"{val.pop('mount_comp')}"],
            temp_levels_feed=[temp_levels['ht_flow'], temp_levels['ht_flow_high'], temp_levels['dhw']],
            temp_air_t=ts_red_inputs[PvgisDataCols.Air_temp],
            **val
        )
        for key, val in sce_data.tech_data.st.items()
    }
    batteries = {
        f"{key}": ElStorage(f"{key}", **val)
        for key, val in sce_data.tech_data.batt.items()
    }
    buffer_storages = {
        f"{key}": MultiTempStorage(
            f"{key}",
            temp_levels=[temp_levels['ht_flow'], temp_levels['ht_flow_high']],
            var_cost_t=0.0000001,
            **val)
        for key, val in sce_data.tech_data.buffsto.items()
    }
    dhw_storages = {
        f"{key}": MultiTempStorage(f"{key}", temp_levels=[temp_levels['dhw'],], **val)
        for key, val in sce_data.tech_data.dhwsto.items()
    }
    th_inertia = {
        'th-inertia': ThInertia(
            'th-inertia',
            temp_levels=[temp_levels['roomheat']],
            comp_volume=sce_data.building_data.wall_volume,
            th_cap=sce_data.building_data.wall_th_cap,
            density=sce_data.building_data.wall_density,
            transm_loss=sce_data.building_data.transm_loss,
            delta_temp_max=sce_data.consumption_data.delta_temp_max,
            c_init_rel=0.0,
        ),
    }
    hp = {
        **{
            f"{key}": Heatpump(
                f"{key}",
                temp_levels_feed=[temp_levels['ht_flow'], temp_levels['ht_flow_high'], temp_levels['dhw']],
                temp_levels_drain=[temp_levels['air']],
                specs_table=bm_input.hp_specs[val.pop('type')].copy(),
                apply_heating_curve=True,
                indoortemp_t=dem['th_roomheat'].indoor_temp_t,
                **val)
            for key, val in sce_data.tech_data.hp.items()
        },
    }
    ac = {
        ** {
           f"{key}": Heatpump(
               f"{key}",
               temp_levels_feed=[temp_levels['roomheat']],
               temp_levels_drain=[temp_levels['air']],
               specs_table=bm_input.hp_specs[val.pop('type')].copy(),
               apply_heating_curve=False,
               indoortemp_t=dem['th_roomheat'].indoor_temp_t,
               **val)
           for key, val in sce_data.tech_data.ac.items()
       },
    }

    # # plot VL Temperatur and HP efficiency
    # vl_temp_df = pd.DataFrame(
    #     {'Aussentemperatur': temp_levels['air'].temp_t.loc['Winter'].sort_values(ascending=False).T.reset_index(drop=True)})
    # vl_temp_df['Vorlauftemperatur'] = vl_temp_df['Aussentemperatur'].apply(
    #     lambda x: temp_levels['ht_flow']._calc_t_vl(x - 20) if x - 20 < 0 else 0)
    # vl_temp_df['COP'] = hp['hp'].cop_t[temp_levels['ht_flow']].loc['Winter'].sort_values(ascending=False).reset_index(drop=True)
    # plot_heat_curve(vl_temp_df)
    #

    deh = {
        f"{key}": DirectElHeater(
            f"{key}",
            temp_levels_feed=[temp_levels['ht_flow'], temp_levels['ht_flow_high'], temp_levels['dhw']],
            **val
        )
        for key, val in sce_data.tech_data.deh.items()
    }
    boiler = {
        f"{key}": CombustionHeater(
            f"{key}",
            temp_levels_feed=[temp_levels['ht_flow'], temp_levels['ht_flow_high'], temp_levels['dhw']],
            **val
        )
        for key, val in sce_data.tech_data.boiler.items()
    }
    refurbishments = {
        f"{key}": Refurbishment(f"{key}", **val)
        for key, val in sce_data.tech_data.refurb.items()
    }

    # calculate reference heat load (Heizlast)
    if sce_data.building_data.ref_heat_load == 'nan':
        sce_data.building_data.ref_heat_load = (
                (dem['th_roomheat'].transm_loss + dem['th_roomheat'].vent_loss)
                * (const.NOM_TEMP_INDOOR - const.T_OUT_REF)
        )

    print('Reference heat load:', sce_data.building_data.ref_heat_load, 'kW')
    for refurb in sce_data.tech_data.refurb.keys():
        print(f'Reference heat load with {refurb}:',
              (1 - sce_data.tech_data.refurb[refurb]['dem_red_factor'])*sce_data.building_data.ref_heat_load, 'kW')

    # nodes
    nodes = {
        'el_pv': ElNode(
            'el_pv',
            sources=[*pv.values(), *batteries.values(), links['el_imp-el_pv'], links['el_hp-el_pv'],
                     links['el_emob-el_pv'],],
            sinks=[links['el_pv-el_imp'], links['el_pv-el_hp'], links['el_pv-el_emob'], exp['el_pv'],
                   *batteries.values(), ],
        ),
        'el_imp': ElNode(
            'el_imp',
            sources=[links['el_pv-el_imp'], imp['el_std'], ],
            sinks=[dem['el_hh'], links['el_imp-el_pv'], ],
        ),
        'el_hp': ElNode(
            'el_hp',
            sources=[links['el_pv-el_hp'], imp['el_hp']],
            sinks=[*hp.values(), *deh.values(), links['el_hp-el_pv'], ],
        ),
        'el_emob': ElNode(
            'el_emob',
            sources=[links['el_pv-el_emob'], imp['el_emob']],
            sinks=[dem['emob_fix'], dem['emob_flex'], links['el_emob-el_pv'],],
        ),
        'th_ht_flow': ThNode(
            'th_ht_flow',
            sources=[*hp.values(), *buffer_storages.values(), *deh.values(), *boiler.values(), *st.values(), ],
            sinks=[*buffer_storages.values(), links['th_ht_flow-th_roomheat']],
            temp_level=temp_levels['ht_flow'],
            p_min=sce_data.building_data.ref_heat_load,
        ),
        'th_ht_flow_high': ThNode(
            'th_ht_flow_high',
            sources=[*hp.values(), *buffer_storages.values(), *deh.values(), *boiler.values(), *st.values(), ],
            sinks=[*buffer_storages.values(), links['th_ht_flow_high-th_roomheat']],
            temp_level=temp_levels['ht_flow_high'],
        ),
        'th_roomheat': RoomheatNode(
            'th_roomheat',
            sources=[*ac.values(), *th_inertia.values(), links['th_ht_flow-th_roomheat'],
                     links['th_ht_flow_high-th_roomheat'], imp['heat_gains']],
            sinks=[dem['th_roomheat'], *th_inertia.values(), exp['excess_heat']],
            temp_level=temp_levels['roomheat'],
        ),
        'th_dhw': ThNode(
            'th_dhw',
            sources=[*hp.values(), *deh.values(), *boiler.values(), *dhw_storages.values(), *st.values(), ],
            sinks=[dem['th_dhw'], *dhw_storages.values()],
            temp_level=temp_levels['dhw'],
        ),
        'th_air': ThNode(
            'th_air',
            sources=[imp['air']],
            sinks=[*hp.values(), *ac.values()],
            temp_level=temp_levels['air'],
        ),
    }

    components = {
        'potentials': potentials,
        'nodes': nodes,
        'links': links,
        'pv': pv,
        'solar_thermal': st,
        'batteries': batteries,
        'buffer_storages': buffer_storages,
        'dhw_storages': dhw_storages,
        'th_inertia': th_inertia,
        'demand': dem,
        'import': imp,
        'export': exp,
        'heatpump': {**hp, **ac},
        'deh': deh,
        'refurbishments': refurbishments,
        'boiler': boiler,
    }

    return components, ts_red_inputs


def run_scenario(components: dict, cost_weight_factors: pd.Series, co2_price) -> ConcreteModel:
    """
    Executes a scenario using the given components defined in the prepare_input_data function,
    cost weight factors, and co2 price.
    This function initializes a Pyomo model, solves it using a solver (either local e.g. GLPK or
    a solver online on the neos server), and logs any infeasible constraints if present.

    :param components: Dict of system components to be used in the model.
    :type components: dict
    :param cost_weight_factors: Cost weight factors applied in the optimization.
    :type cost_weight_factors: pd.Series
    :param co2_price: The price of CO2 to be considered in the optimization model.
    :type co2_price: float
    :return: A Pyomo `ConcreteModel` object representing the solved optimization model.
    :rtype: ConcreteModel
    """

    check_components(components)

    start = datetime.now()
    # pyomo Model
    m = init_pyomo_model(components, cost_weight_factors=cost_weight_factors, co2_price=co2_price)
    # solve
    print('solving model...')

    # use online solver: https://neos-server.org/neos/solvers/index.html
    solver_manager = SolverManagerFactory('neos')
    solver_manager.solve(m, solver="cplex").write()

    # # use local solver: glpk
    # SolverFactory('glpk').solve(m).write()

    # log errors if needed
    logging.basicConfig(filename='infeasile.log', encoding='utf-8', level=logging.DEBUG)
    log_infeasible_constraints(m, log_expression=True, log_variables=True)

    print("solving time: ", datetime.now() - start)

    return m


if __name__ == "__main__":

    proj_name = "Example_building_opt"

    # set up logging to file and console with tee
    out_path = f'{const.PATH_TO_WD}/data/output/logs/run_{proj_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    with open(out_path, 'w', encoding='utf-8') as out_file:
        tee = Tee(sys.stdout, out_file)
        with redirect_stdout(tee), redirect_stderr(tee):

            # start main routine
            start = datetime.now()
            bm_input = BuildingModelInput(f"{proj_name}.xlsx")
            results = {}
            res_summary = pd.DataFrame()
            results_folder = create_result_folder(proj_name)
            save_xlsx_wb(results_folder + "/InputData.xlsx", bm_input.excel_data)

            skip_scenarios = [] #['BAU', 'PV']
            for scenario in bm_input.scenarios:
                start = datetime.now()
                if scenario not in skip_scenarios:
                    print(f'running scenario {scenario}...')
                    sce_data = bm_input.scenario_data[scenario]

                    # apply model settings
                    const.LATITUDE = sce_data.location.latitude
                    const.LONGITUDE = sce_data.location.longitude
                    const.MODELED_YEAR = sce_data.model_settings.modeled_year
                    const.APPLY_TIMESERIES_REDUCTION = sce_data.model_settings.apply_timeseries_reduction
                    const.REAL_INTEREST_RATE = sce_data.economic_data.real_interest_rate / 100
                    const.DURATION = sce_data.economic_data.duration

                    # general building data
                    const.NR_OF_HOUSEHOLDS = int(sce_data.consumption_data.nr_of_households)
                    const.NOM_TEMP_INDOOR = sce_data.consumption_data.avg_indoor_temp
                    const.NIGHT_TEMP_INDOOR = sce_data.consumption_data.avg_indoor_temp_night
                    if sce_data.consumption_data.heating_threshold is not None:
                        const.HEATING_THRESHOLD = sce_data.consumption_data.heating_threshold
                    const.T_OUT_REF = sce_data.location.T_out_ref
                    const.HEATING_AREA_m2 = sce_data.building_data.heated_area
                    const.HEATING_VOLUME_m3 = sce_data.building_data.heated_volume
                    const.VENT_RATE = sce_data.building_data.vent_rate
                    const.AIR_CHANGE = sce_data.building_data.air_change
                    const.INTERNAL_GAINS = sce_data.building_data.internal_gains

                    # create climate data summary
                    climate_data = climate_data_analysis(write_xlsx=True, results_folder=results_folder)

                    # input data preparation
                    components, ts_red_inputs = prepare_input_data(sce_data, climate_data)

                    # export updated climate data summary to output folder (added selected weeks)
                    write_climate_data_xlsx_analysis(results_folder=results_folder, overwrite=True)

                    # model scenario and run
                    m = run_scenario(
                        components,
                        cost_weight_factors=ts_red_inputs[InputData.CostFactors],
                        co2_price=(sce_data.economic_data.co2_price / 1000)
                    )

                    # write results
                    results[scenario] = write_scenario_results(
                        sce_data, m, ts_red_inputs, results_folder=results_folder
                    )
                    print('scenario runtime ', datetime.now() - start)

            # dump pickle for manual result postprocessing
            dump_pickle('last_results', results)

            # export results summary to output folder
            write_results_summary(results, results_folder=results_folder)


