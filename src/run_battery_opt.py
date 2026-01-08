"""
src/run_battery_opt.py

This module executes the optimization for battery storage systems and related components.
It reads inputs from an Excel workbook, prepares time series including high-load time
windows (Hochlastzeitfenster HLZF), constructs components (PV, storage, nodes, import/export, demand),
initializes and solves a Pyomo ConcreteModel, and exports results.

Main responsibilities:

- run_optimization(components, cost_weight_factors, co2_price):
    Validates components, creates the Pyomo model, solves it (e.g., via NEOS/CPLEX
    or a local solver), logs infeasible constraints, and returns the solved ConcreteModel.

- apply_hlzf_windows(hlzf_df, df_target, year=2024):
    Marks time steps in the provided time series that fall within defined high-load windows (Hochlastzeitfenster)
    per season and daily intervals.

- write_results(m):
    Extracts time-series and time-invariant outputs from the solved model, computes KPIs,
    and returns a results dictionary containing DataFrames and summary metrics.

Script entrypoint (when run as __main__):

- Sets relevant constants and environment variables (e.g., NEOS_EMAIL, TS_PER_HOUR).
- Loads `Example_battery_opt.xlsx`, prepares time series and HLZF markers.
- Defines scenarios, potentials, technologies, storages and nodes.
- Runs the optimization for each scenario, saves results (pickle, Excel) and writes a summary.

Notes:

- Optimization is performed for whole year without timeseries reduction in 15min steps.
- Using the NEOS service requires a valid email address set in the environment.
- The module depends on project-specific component, model and helper functions.
"""

import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta
import logging
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.environ import ConcreteModel, SolverManagerFactory

from src.const import *
from src.model.components import *
from src.helper import load_pickle, dump_pickle, Tee
from src.model.pyomo_model import check_components, init_pyomo_model
from src.export.analysis_plots import *
from src.model.economics import calc_present_value


def run_optimization(components, cost_weight_factors, co2_price) -> ConcreteModel:
    """
    Runs an optimization process for the given components and cost parameters using
    a Pyomo model. The function initializes a Pyomo ConcreteModel, solves it using
    the specified solver, logs errors if necessary, and reports the solving time.

    :param components: List of system components to be optimized.
    :type components: list
    :param cost_weight_factors: Dictionary mapping cost categories to their weights.
    :type cost_weight_factors: dict
    :param co2_price: Price of CO2 emissions to incorporate into the optimization model.
    :type co2_price: float
    :return: A Pyomo ConcreteModel instance representing the solved optimization model.
    :rtype: ConcreteModel
    """

    check_components(components)

    start = datetime.now()
    # create pyomo Concrete Model
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


def apply_hlzf_windows(hlzf_df, df_target, year=2024):
    """
    Apply HLZF (High Load Zone Factor) calculation to a given dataframe based on seasonal
    time ranges and daily time intervals.

    The function processes data to identify specific periods within each season. These
    periods are defined by start and end dates, extracted from the input `hlzf_df`. For each
    season, start and end times for specific intervals are iterated over. The function updates
    a target column in the provided dataframe, marking rows that fall within the defined
    timeframes.

    :param hlzf_df: A pandas DataFrame containing seasonal date and time intervals.
        - The first row includes season start and end dates.
        - Rows starting from index 3 provide start and end times for the high load time
          intervals across the respective seasons.
    :param df_target: A pandas DataFrame containing datetime-indexed data. The index is
        expected to represent datetime objects, which are used for filtering data according
        to the seasonal time periods and intervals.
    :param year: An integer representing the year for which to calculate the seasonal
        time ranges. Defaults to 2024.
    :return: A pandas Series indicating rows that match the high load time conditions.
        The series ("HLZF") is binary, where 1 represents a match with the defined time
        periods and 0 represents no match.
    """

    # Mapping der Saisonnamen auf die Spalten
    seasons = ['Frühling', 'Sommer', 'Herbst', 'Winter']
    season_date_ranges = {}

    # Zeile 1 im HLZF-DF enthält die Saison-Zeiträume (Anfang/Ende)
    date_row = hlzf_df.iloc[1]

    # Hole Start- und Enddaten jeder Saison
    for i, season in enumerate(seasons):
        start_str = date_row[i * 2]
        end_str = date_row[i * 2 + 1]
        # Parsing mit Dummy-Jahr
        start_date = pd.to_datetime(f"{start_str}{year}", format="%d.%m.%Y")
        end_date = pd.to_datetime(f"{end_str}{year}", format="%d.%m.%Y")
        season_date_ranges[season] = (start_date, end_date)

    # Zeilen ab Index 3 enthalten HLZF-Start-/Endzeiten je Saison
    df_target["HLZF"] = 0

    for season_idx, season in enumerate(seasons):
        start_col = season_idx * 2
        end_col = season_idx * 2 + 1
        start_date, end_date = season_date_ranges[season]

        # Filter df_target für die Saisonperiode
        if season == 'Winter':
            mask = (
                ((df_target.index >= start_date) & (df_target.index <= datetime(year,12,31,0,0,0)))
                |((df_target.index >= datetime(year,1,1,0,0,0)) & (df_target.index <= end_date))
            )
        else:
            mask = (df_target.index >= start_date) & (df_target.index <= end_date)
        season_df = df_target[mask]

        # Iteriere über HLZF-Zeilen
        for row_idx in range(3, hlzf_df.shape[0]):
            start_time_str = hlzf_df.iloc[row_idx, start_col]
            end_time_str = hlzf_df.iloc[row_idx, end_col]
            if pd.notna(start_time_str) and pd.notna(end_time_str):
                start_time = pd.to_datetime(start_time_str, format="%H:%M:%S").time()
                end_time = (pd.to_datetime(end_time_str, format="%H:%M:%S") - timedelta(hours=0, minutes=15)).time()

                # Erstelle Maske für Zeiten in jedem Tag
                time_mask = season_df.between_time(start_time, end_time).index
                df_target.loc[time_mask, 'HLZF'] = 1


    return df_target['HLZF']


def write_results(m: ConcreteModel) -> dict:
    """
    Generates detailed results and summaries based on the provided optimization model. The function processes time-series
    and time-invariant data from the model, creates summarized data structures like DataFrames, calculates key
    performance indicators (KPIs), and prints formatted outputs to aid in analysis. It is specifically designed for
    energy systems modeling and assumes the `ConcreteModel` instance contains particular variables and parameters.

    :param m: The ConcreteModel instance containing optimization results and parameter definitions.
    :type m: ConcreteModel
    :return: A dictionary containing detailed results and summaries:
             - 'ts_el_sources': Time-series data of electricity sources as a DataFrame.
             - 'ts_el_sinks': Time-series data of electricity sinks as a DataFrame.
             - 'ts_th_sources': Time-series data of thermal sources as a DataFrame.
             - 'ts_th_sinks': Time-series data of thermal sinks as a DataFrame.
             - 'ts_storages': Time-series data of storages as a DataFrame.
             - 'ti_components': Time-invariant variables such as installed capacity, costs, and CO2 balance as a DataFrame.
             - 'sums': Summed-up yearly values for energy sources and sinks, scaled to hours, as a DataFrame.
             - 'key_facts': Key performance indicators as a Pandas Series.
    :rtype: dict
    """
    results = {
        'ts_el_sources': pd.DataFrame(
            {str(comp): {(s, t): m.p_el_t_feed[comp, t, s].value for s in m.season for t in m.t}
             for comp in m.el_sources}).sort_index(),
        'ts_el_sinks': pd.DataFrame(
            {str(comp): {(s, t): m.p_el_t_drain[comp, t, s].value for s in m.season for t in m.t}
             for comp in m.el_sinks}).sort_index(),
        'ts_th_sources': pd.DataFrame(
            {f"{comp}:{temp.name}": {(s, t): m.p_th_t_feed[comp, temp, t, s].value for s in m.season for t in m.t}
             for comp in m.th_sources for temp in m.temp_levels}).sort_index(),
        'ts_th_sinks': pd.DataFrame(
            {f"{comp}:{temp.name}": {(s, t): m.p_th_t_drain[comp, temp, t, s].value for s in m.season for t in m.t}
             for comp in m.th_sinks for temp in m.temp_levels}).sort_index(),
        'ts_storages': pd.DataFrame(
            {str(comp): {(s, t): m.c_t[comp, t, s].value for s in m.season for t in m.t}
             for comp in m.storages}).sort_index(),
        'ti_components': pd.DataFrame(
            {str(var): {str(comp): var[comp].value for comp in m.components if comp in var}
             for var in [m.is_built, m.p_inst, m.p_peak_hltf, m.c_inst, m.c_inst_l, m.inv_costs, m.fix_costs,
                         m.var_costs, m.revenues, m.co2eq_balance]}
        ).sort_index(),
    }
    results['ti_components'].loc[:, 'tot_costs'] = (results['ti_components'].filter(regex='costs').sum(axis=1) -
                                                    results['ti_components'].filter(regex='revenues').sum(axis=1))

    results['ti_components'].loc['sum', :] = results['ti_components'].sum()
    results['ti_components'] = results['ti_components'][
        ['is_built', 'p_inst', 'p_peak_hltf', 'c_inst', 'c_inst_l', 'inv_costs', 'fix_costs','var_costs', 'revenues',
         'tot_costs', 'co2eq_balance']]

    results['sums'] = pd.DataFrame({
            'sums_el_sources': (results['ts_el_sources'].sum() / const.TS_PER_HOUR).sort_index(),
            'sums_el_sinks': (results['ts_el_sinks'].sum() / const.TS_PER_HOUR).sort_index(),
            'sums_th_sources': (results['ts_th_sources'].sum() / const.TS_PER_HOUR).sort_index(),
            'sums_th_sinks': (results['ts_th_sinks'].sum() / const.TS_PER_HOUR).sort_index(),
        })

    key_facts = pd.Series()
    key_facts['Jährliche Kosten [T€/Jahr]'] = results['ti_components'].loc['sum', 'tot_costs'] / 10 ** 3
    key_facts['Jährliche Kosten Strombezug [T€/Jahr]'] = \
        results['ti_components'].loc['ElImport: el_std', 'tot_costs'] / 10 ** 3
    key_facts['Investitionskosten Speicher (Annuität) [T€/Jahr]'] = \
        results['ti_components'].loc['ElStorage: battery', 'inv_costs'] / 10 ** 3
    key_facts['Jährliche Erlöse Direktvermarktung [T€/Jahr]'] = \
        results['ti_components'].loc['ElExport: el_pv', 'revenues'] / 10 ** 3
    key_facts['Jährliche Kosten Leistungspreis Strom [T€/Jahr]'] = \
        results['ti_components'].loc['ElImport: el_std', 'inv_costs'] / 10 ** 3

    key_facts['Speicherkapazität [kWh]'] = results['ti_components'].loc['ElStorage: battery', 'c_inst']
    key_facts['Jährlicher Strombezug [MWh]'] = results['sums'].loc['ElImport: el_std', 'sums_el_sources']/10**3
    key_facts['PV-Einspeisung [MWh]'] = results['sums'].loc['ElExport: el_pv', 'sums_el_sinks'] / 10 ** 3
    key_facts['Spitzenlast [kW]'] = results['ti_components'].loc['ElImport: el_std', 'p_inst']
    key_facts['Spitzenlast Hochlastzeitfenster [kW]'] = results['ti_components'].loc['ElImport: el_std', 'p_peak_hltf']
    key_facts['Benutzungsdauer [h/a]'] = (key_facts['Jährlicher Strombezug [MWh]'] * 10**3
                                          / key_facts['Spitzenlast [kW]'])


    # key_facts['full_load_hours'] = results['ti_components'].loc['ElImport']
    # key_facts['pvgen'] = results['sums']['sums_el_sources'].filter(regex='PV').sum().sum()
    results['key_facts'] = key_facts

    print("Time invariant variables:\n", results['ti_components'].to_string())
    print("Yearly energy sums:\n ", results['sums'].to_string())
    print(key_facts.to_string())

    return results


if __name__ == "__main__":

    proj_name = "Example_battery_opt"
    # set up logging to file and console with tee
    out_path = f'{const.PATH_TO_WD}/data/output/logs/run_{proj_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    with open(out_path, 'w', encoding='utf-8') as out_file:
        tee = Tee(sys.stdout, out_file)
        with redirect_stdout(tee), redirect_stderr(tee):

            # start main routine
            const.APPLY_TIMESERIES_REDUCTION = False
            const.TS_PER_HOUR = const.TS_RES.QUARTER_HOURLY.value


            # PREPARE INPUT DATA
            fpath = PATH_TO_WD + f"/data/input/{proj_name}.xlsx"
            input_sheets = pd.read_excel(
                fpath,
                sheet_name=['Techno-oekonomische Daten', 'Hochlastzeitfenster', 'Zeitreihen'],
                header=[0],
                index_col=[0],
            )
            input_sheets['Techno-oekonomische Daten'] = input_sheets['Techno-oekonomische Daten']['Wert']
            input_sheets['Zeitreihen'] = input_sheets['Zeitreihen'][4:]
            input_sheets['Zeitreihen'].index = pd.to_datetime(input_sheets['Zeitreihen'].index)

            input_sheets['Zeitreihen']['HLZF'] = apply_hlzf_windows(
                input_sheets['Hochlastzeitfenster'], pd.DataFrame(index=input_sheets['Zeitreihen'].index))

            # adjust index to fit optimization problem
            input_sheets['Zeitreihen'].index = pd.MultiIndex.from_tuples(
                [(2024, t) for t in range(0, 366 * 24 * const.TS_PER_HOUR)]
            )

            # SET INPUT PARAMETERS
            scenarios = {
                "Fixer Tarif ohne Speicher": {
                    'Speicher': 0,
                    'Dyn. Tarif': 0,
                    'Atyp. NN': 0
                },
                "Fixer Tarif mit Speicher (Eigenverbrauchsoptimierung)": {
                    'Speicher': 1,
                    'Dyn. Tarif': 0,
                    'Atyp. NN': 0
                },
                "Fixer Tarif mit Speicher + atypische Netznutzung": {
                    'Speicher': 1,
                    'Dyn. Tarif': 0,
                    'Atyp. NN': 1
                },
                "Day Ahead 2024 mit Speicher": {
                    'Speicher': 1,
                    'Dyn. Tarif': 1,
                    'Atyp. NN': 0
                },
                "Day Ahead 2024 + atypische Netznutzung mit Speicher": {
                    'Speicher': 1,
                    'Dyn. Tarif': 1,
                    'Atyp. NN': 1
                }
            }

            const.REAL_INTEREST_RATE = input_sheets['Techno-oekonomische Daten']['Kalkulationszins']


            # potentials
            pv_cap = input_sheets['Techno-oekonomische Daten']['Leistung PV [kWp]']
            potentials = {
                f'pv': SolarPotentialDummy('pv', input_sheets['Zeitreihen']['PV-Erzeugung [kW]']/pv_cap)
            }

            dem = {
                'el': ElDemand('el', input_sheets['Zeitreihen']['Lastgang [kW]']),
            }


            # input_xlsx['Zeitreihen']['Day-Ahead Preise [€/MWh]']
            results = {}
            for sce in scenarios:

                print("running scenario: ", sce)

                sce_params = scenarios[sce]

                if sce_params['Dyn. Tarif']:
                    el_price_euro_per_kwh = (
                        + input_sheets['Techno-oekonomische Daten']['Netzentgelte Arbeitspreis [ct./kWh]']
                        + input_sheets['Techno-oekonomische Daten']['Abgaben [ct./kWh]']
                    ) / 100 + input_sheets['Zeitreihen']['Day-Ahead Preise [€/MWh]'] / 1000
                else:
                    el_price_euro_per_kwh = (
                        input_sheets['Techno-oekonomische Daten']['Arbeitspreis Beschaffung [ct./kWh]']
                        + input_sheets['Techno-oekonomische Daten']['Netzentgelte Arbeitspreis [ct./kWh]']
                        + input_sheets['Techno-oekonomische Daten']['Abgaben [ct./kWh]']
                    ) / 100


                el_price_euro_per_kw = input_sheets['Techno-oekonomische Daten']['Netzentgelte Leistungspreis [€/kW]']
                feed_in_price_euro_per_kwh = - input_sheets['Zeitreihen']['Day-Ahead Preise [€/MWh]'] / 1000
                # feed_in_price_euro_per_kwh = -0.02

                imp = {
                    'el_std': ElImport(
                        'el_std',
                        var_cost_t=el_price_euro_per_kwh,
                        fix_cost_yearly=0.0,
                        co2eq_per_kwh_t=0.0,
                        inv_cost=calc_present_value(el_price_euro_per_kw,
                                                    duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE),
                        lifetime=const.DURATION,
                        atypical_consumption=sce_params['Atyp. NN'],
                        peak_load_timesteps=input_sheets['Zeitreihen']['HLZF'],
                        t_full_min=2500 if input_sheets['Techno-oekonomische Daten']['Benutzungsdauer größer 2500h'] else 0),
                }
                exp = {
                    'el_pv': ElExport('el_pv', var_cost_t=feed_in_price_euro_per_kwh),
                }
                # technologies
                inv_cost = input_sheets['Techno-oekonomische Daten']['Investitionskosten PV gesamt [€]']
                pv = {
                    'pv': PV(
                        'pv',
                        potentials['pv'],
                        inv_cost=0.0 if str(inv_cost) == 'nan' else inv_cost,
                        lifetime=input_sheets['Techno-oekonomische Daten']['Lebensdauer PV [Jahre]'],
                        p_max=pv_cap,
                        pmax_at_lifeend=input_sheets['Techno-oekonomische Daten']['PV Degradationsfaktor EOL'],
                        is_built=1)
                }

                # in and out losses together make up round trip losses
                in_out_eff = (1.0 + input_sheets['Techno-oekonomische Daten']['Effizienz (Round trip)'])/2
                lifetime = input_sheets['Techno-oekonomische Daten']['Lebensdauer BS [Jahre]']
                max_cycles = 365 * input_sheets['Techno-oekonomische Daten']['Zyklen pro Tag'] * lifetime
                max_cap = input_sheets['Techno-oekonomische Daten']['Kapazität BS [kWh]'] + 0.01
                min_cap = input_sheets['Techno-oekonomische Daten']['Kapazität BS [kWh]']
                max_p = input_sheets['Techno-oekonomische Daten']['Leistung Wechselrichter BS [kW]'] + 0.01
                min_p = input_sheets['Techno-oekonomische Daten']['Leistung Wechselrichter BS [kW]']

                storages = {
                    'battery': ElStorage(
                        'battery',
                        fix_cost_inv=0,
                        inv_cost=input_sheets['Techno-oekonomische Daten']['Investitionskosten BS [€/kWh]'],
                        lifetime=input_sheets['Techno-oekonomische Daten']['Lebensdauer BS [Jahre]'],
                        c_max=MAX_C if str(max_cap) == 'nan' else max_cap,
                        c_min=0.0 if str(min_cap) == 'nan' else min_cap,
                        p_max=MAX_P if str(max_p) == 'nan' else max_p,
                        p_min=0.0 if str(min_p) == 'nan' else min_p,
                        c_rate_max=input_sheets['Techno-oekonomische Daten']['max. C-rate'],
                        eff_in=in_out_eff,
                        eff_out=in_out_eff,
                        max_cycles=max_cycles,
                        is_built=sce_params['Speicher'],
                    ),
                }

                # nodes
                nodes = {
                    'el_pv': ElNode(
                        'el_pv',
                        sources=[*pv.values(), storages['battery'], imp['el_std'], ],
                        sinks=[storages['battery'], dem['el'], exp['el_pv']],
                    ),
                }

                components = {
                    'potentials': potentials,
                    'nodes': nodes,
                    'links': {},
                    'pv': pv,
                    'solar_thermal': {},
                    'storages': storages,
                    'demand': dem,
                    'import': imp,
                    'export': exp,
                    'heatpump': {},
                    'deh': {},
                    'refurbishments': {},
                    'boiler': {},
                }

                m = run_optimization(components,
                                     cost_weight_factors=pd.Series(1.0, index=input_sheets['Zeitreihen'].index), # dummy
                                     co2_price=0.0)

                results[sce] = write_results(m)

            dump_pickle('last_results', results)

            res_summary = pd.DataFrame(
                {sce_name: sce_res['key_facts'] for sce_name, sce_res in results.items()}
            )
            # sys.stdout = cmd
            # out_file.close()
            res_summary.to_excel(f'{PATH_TO_WD}/data/output/results/{proj_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.xlsx')
            print(res_summary.to_string())


