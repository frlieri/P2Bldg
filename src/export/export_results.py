"""
`src/export/export_results.py`

Helpers to assemble, summarise and export scenario results produced by the optimisation
model. The module collects time-series and aggregated metrics from a ScenarioResults
object, builds human-readable key-facts, prepares plotting DataFrames and writes
Excel workbooks with configurable charts.

Main responsibilities:

- create_keyfacts_summary(res, sce_data, printout=False) -> pd.Series
    Compute a compact summary of costs, installed capacities, yearly energy balances,
    performance indicators (self-sufficiency, COP, LCOE/LCOH, battery cycles, etc.)
    from a ScenarioResults instance and its ScenarioData inputs.

- define_ts_plots(res) -> None
    Build per-season time-series DataFrames combining electrical, thermal and
    auxiliary series (temperature, CO₂ intensity, electricity prices) and attach
    them to the ScenarioResults instance for downstream plotting/export.

- write_scenario_results(sce_data, m, ts_red_inputs, results_folder) -> ScenarioResults
    Persist intermediate pickles, construct a ScenarioResults object, generate the
    key-facts summary, prepare seasonal time-series for plotting and export a
    detailed `.xlsx` workbook with configured graphs.

- write_results_summary(results, results_folder, print_summary=True) -> None
    Aggregate multiple scenario key-facts into a summary workbook, export associated
    charts and optionally print the summary table.

- write_climate_data_xlsx_analysis(wb_dict=None, results_folder=None, overwrite=False) -> None
    Export or update a climate-data analysis workbook produced from PVGIS/TMY-derived
    aggregates and configured graph specifications.

Behavior and side effects:

- Produces on-disk artifacts (pickles and `.xlsx` files) and may print status.
- Relies on the project's `ScenarioResults`/`ScenarioData` data formats and helpers
  (`dump_pickle`, `load_pickle`, `create_result_folder`, `create_climate_data_dump_id`).
- Uses `save_xlsx_wb` to create Excel workbooks with declarative chart specifications.
- Expects properly populated model results and time-series inputs; computations assume
  consistent indexing and energy-balance conventions used across the project.

Key dependencies:

- pandas, pyomo (ConcreteModel), project modules under `src.*` (data preparation,
  model/economics, export/excel_generator, helper utilities) and project constants.

Intended use:
- Final result export and light reporting after optimisation runs.
"""

import os.path

import pandas as pd
from enum import Enum

from pyomo.environ import ConcreteModel

from src import const
from src.data_preparation.data_formats import ScenarioData, ScenarioResults
from src.model.economics import calc_present_value
from src.data_preparation.demand_data import CO2ElecCols
from src.data_preparation.pvgis_climate_data import PvgisDataCols
from src.export.excel_generator import save_xlsx_wb
from src.helper import load_pickle, create_result_folder, dump_pickle, create_climate_data_dump_id, create_ts_multiindex


class Colors(str, Enum):
    LIGHT_YELLOW = "#FFF3A1"  # softened
    YELLOW = "#FFE066"        # pastel yellow
    DARK_YELLOW = "#E6B800"   # golden mustard
    LIGHT_ORANGE = "#FFD3A1"  # soft apricot
    ORANGE = "#FFB347"        # warm pastel orange
    DARK_ORANGE = "#CC7000"   # muted burnt orange
    LIGHT_RED = "#FFB3B3"     # salmon pink
    RED = "#E57373"           # pastel red
    DARK_RED = "#993333"      # softened dark red
    LIGHT_BLUE = "#A7D3F2"    # soft sky blue
    BLUE = "#64B5F6"          # pastel blue
    DARK_BLUE = "#2C6CB8"     # softened royal blue
    LIGHT_GREEN = "#AEDFA4"   # mint green
    GREEN = "#81C784"         # soft forest green
    DARK_GREEN = "#4B8B4B"    # muted dark green
    LIGHT_BROWN = "#C9B7A5"   # beige brown
    BROWN = "#A1887F"         # muted brown
    DARK_BROWN = "#6D4C41"    # warm dark brown
    LIGHT_GREY = "#E0E0E0"    # softened light grey
    GREY = "#BDBDBD"          # middle grey
    DARK_GREY = "#757575"     # softer charcoal
    PINK = "#F48FB1"          # pastel pink
    NEON = "#8CFF61"          # softened neon green
    WHITE = "#FAFAFA"         # off-white
    BLACK = "#212121"         # soft black


class StrEnum(str, Enum):
    @classmethod
    def to_list(cls):
        return [member.value for member in cls]


class TsElCols(StrEnum):
    HELPER_EL = 'Helper variable: neg. sum el'
    FEEDIN = 'Feed-in'
    BATTERY_CHARGE = 'Battery-charge'
    BATTERY_DISCHARGE = 'Battery-discharge'
    GRID_DEMAND = 'Grid-demand'
    GRID_HEATPUMP = 'Grid-heatpump'
    GRID_EMOB = 'Grid-eMob'
    PV = 'PV'
    PV_CURTAILED = 'PV curtailed'
    EL_DEMAND = 'El. Demand'
    EL_DEM_HEAT = 'El. Demand with heating / AC'
    EL_DEM_TOT = 'Total El. Demand incl. EV charging'


class TsThCols(StrEnum):
    HELPER_TH = 'Helper variable: neg. sum th'
    DHW_STORAGE_CHARGE = 'Dhw-storage-charge'
    BUFFER_STORAGE_CHARGE = 'Buffer-storage-charge'
    BUFFER_STORAGE_CHARGE_HT = 'Buffer-storage-charge-HT'
    TH_INERTIA_CHARGE = 'Th. inertia-charge'
    HEAT_GAINS_USED = 'Heat gains used'
    SOLAR_THERMAL = 'Solar-thermal'
    HEATPUMP_EL = 'Heatpump-el'
    HEATPUMP_AMBIENT_HEAT = 'Heatpump-ambient heat'
    GAS_BOILER = 'Gas boiler'
    DIRECT_EL_HEATER = 'Direct el. heater'
    DHW_STORAGE_DISCHARGE = 'Dhw-storage-discharge'
    BUFFER_STORAGE_DISCHARGE = 'Buffer-storage-discharge'
    BUFFER_STORAGE_DISCHARGE_HT = 'Buffer-storage-discharge-HT'
    TH_INERTIA_DISCHARGE = 'Th. inertia-discharge'
    ROOMHEAT_DEMAND = 'Roomheat Demand'
    TH_DEMAND = 'Th. Demand'
    HEAT_GAINS = 'Heat gains'
    QMAX_SOLAR_THERMAL = 'Qmax solar-thermal'


class TsOtherCols(StrEnum):
    TEMP_AIR = 'Temp. air in °C'
    CO2_EQ = 'CO2 eq. [kg/kWh]'
    ELEC_PRICE_HH = 'elec. price household [€/kWh]'
    ELEC_PRICE_HEAT = 'elec. price heating [€/kWh]'
    ELEC_PRICE_EMOB = 'elec. price e-mob. [€/kWh]'


class SmryCols(StrEnum):
    TOTAL_COSTS = 'Total costs (annuity) [€/a]'
    OPEX_TOTAL = 'OPEX total [€/a]'
    CO2_COSTS = 'CO2 costs [€/a]'
    CAPEX_ANNUITY = 'CAPEX (annuity) [€/a]'
    CAPEX = 'CAPEX (present values) [€]'
    CAPEX_PV = 'CAPEX PV (present values) [€]'
    CAPEX_BATT = 'CAPEX Battery Storage (present values) [€]'
    CAPEX_HEAT = 'CAPEX Heating (present values) [€]'
    CAPEX_REFURB = 'CAPEX Refurbishment (present values) [€]'
    EL_COST_HH = 'Electricity costs household [€/a]'
    EL_COST_HEAT = 'Electricity costs heat [€/a]'
    EL_COST_EMOB = 'Electricity costs e-mob [€/a]'
    FUEL_COSTS = 'Fuel costs [€/a]'
    FEEDIN_REV = 'Feed-in revenues [€/a]'

    PV_KWP = 'PV [kWp]'
    HP_KW = 'Heatpump [kW_th]'
    AC_KW = 'Air Conditioning [kW_th]'
    DEH_KW = 'Direct Electric Heater [kW_th]'
    BOILER_KW = 'Boiler [kW_th]'
    BATT_KWH = 'Battery Storage [kWh]'
    BUFFER_L = 'Buffer storage [l]'
    BUFFER_KWH = 'Buffer storage energy @dT=10°C [kWh_th]'
    DHW_L = 'Hot water storage [l]'
    DHW_KWH = 'Hot water storage energy @dT=55°C [kWh_th]'
    TH_INERTIA = 'Thermal inertia used [kWh_th]'

    PV_GEN = 'PV generation [kWh]'
    PV_GEN_CURTAILED = 'PV curtailment [kWh]'
    PV_GEN_USED = 'PV generation used [kWh]'
    EL_IMP_HH = 'Electricity import Household [kWh]'
    EL_IMP_HEAT = 'Electricity import Heating [kWh]'
    EL_IMP_EMOB = 'Electricity import e-mobility [kWh]'
    EL_DEM_HH = 'Electricity demand Household [kWh]'
    EL_DEM_HP = 'Electricity demand Heatpump [kWh]'
    EL_DEM_AC = 'Electricity demand Air Conditioning [kWh]'
    EL_DEM_DEH = 'Electricity demand Direct Electric Heater [kWh]'
    EL_DEM_HEAT_TOTAL = 'Electricity demand for heating total [kWh]'
    EL_DEM_EMOB = 'Electricity demand for e-mobility [kWh]'
    EL_DEM_TOTAL = 'Electricity demand total [kWh]'
    TH_DEM_ROOM = 'Thermal demand roomheat'
    TH_DEM_DHW = 'Thermal demand hot water'
    TH_DEM_TOTAL = 'Thermal demand total'
    TH_GEN_BOILER = 'Thermal generation Boiler [kWh]'
    TH_GEN_HP = 'Thermal generation Heatpump [kWh]'
    TH_GEN_AC = 'Thermal generation Air Conditioning [kWh]'
    TH_GEN_DEH = 'Thermal generation Direct Electric Heater [kWh]'
    PV_USED_HEAT = 'PV generation used for heating [kWh]'
    PV_USED_EMOB = 'PV generation used for charging EVs [kWh]'
    PV_USED_HH = 'PV generation used for household [kWh]'
    PV_FEEDIN = 'Excess PV generation Feed-in [kWh]'

    SELF_SUFF = 'Self-sufficiency ratio (autarky)'
    SELF_CONS = 'Self-consumption ratio'
    SELF_SUFF_HEAT = 'Self-sufficiency electrical heat generation'
    SELF_SUFF_HH = 'Self-sufficiency household'
    SELF_SUFF_EMOB = 'Self-sufficiency EV charging'
    GREEN_CHARGE_SHARE = 'Share of green el. charge into battery'
    COP = 'COP (yearly average)'
    SCOP_WINTER = 'SCOP Winter'
    BATT_CYCLES = 'Battery cycles'
    SPEC_PV_GEN = 'Specific PV generation [kWh/kWp]'
    LCOE_PV = 'LCOE PV generation'
    LCOE_SELF = 'LCOE self generation'
    LCOE_TOTAL = 'LCOE total demand'
    LCOH = 'LCOH'


TS_PLOT_STYLES_XLSX = {
    TsElCols.EL_DEMAND.value: {'line': {'color': Colors.DARK_BLUE.value, 'width': 1.0}},
    TsElCols.EL_DEM_HEAT.value: {'line': {'color': Colors.DARK_RED.value, 'width': 1.0}},
    TsElCols.EL_DEM_TOT.value: {'line': {'color': Colors.BLACK.value, 'width': 1.0}},
    TsElCols.PV.value: {'line': None, 'fill': {'color': Colors.LIGHT_YELLOW.value}},
    TsElCols.PV_CURTAILED.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_downward_diagonal',
            'fg_color': Colors.LIGHT_YELLOW.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },
    TsElCols.GRID_HEATPUMP.value: {'line': None, 'fill': {'color': Colors.LIGHT_RED.value}},
    TsElCols.GRID_EMOB.value: {'line': None, 'fill': {'color': Colors.NEON.value}},
    TsElCols.GRID_DEMAND.value: {'line': None, 'fill': {'color': Colors.BLUE.value}},
    TsElCols.BATTERY_CHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_downward_diagonal',
            'fg_color': Colors.LIGHT_BLUE.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsElCols.BATTERY_DISCHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.LIGHT_BLUE.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsElCols.FEEDIN.value: {'line': None, 'fill': {'color': Colors.GREY.value}},
    TsOtherCols.CO2_EQ.value: {'line': {'color': Colors.DARK_BROWN.value, 'width': 1.0}},
    TsOtherCols.ELEC_PRICE_HH.value: {'line': {'color': Colors.BLUE.value, 'width': 1.0}},
    TsOtherCols.ELEC_PRICE_HEAT.value: {'line': {'color': Colors.RED.value, 'width': 1.0}},
    TsOtherCols.ELEC_PRICE_EMOB.value: {'line': {'color': Colors.NEON.value, 'width': 1.0}},
    TsThCols.TH_DEMAND.value: {'line': {'color': Colors.DARK_RED.value, 'width': 1.0}},
    TsThCols.ROOMHEAT_DEMAND.value: {'line': {'color': Colors.RED.value, 'width': 1.0}},
    TsThCols.HEAT_GAINS.value: {'line': {'color': Colors.DARK_YELLOW.value, 'width': 1.0}},
    TsThCols.HEAT_GAINS_USED.value: {'line': None, 'fill': {'color': Colors.YELLOW.value}},
    TsThCols.SOLAR_THERMAL.value: {'line': None, 'fill': {'color': Colors.NEON.value}},
    TsThCols.QMAX_SOLAR_THERMAL.value: {'line': {'color': Colors.NEON.value, 'width': 1.0}},
    TsThCols.HEATPUMP_EL.value: {'line': None, 'fill': {'color': Colors.LIGHT_BLUE.value}},
    TsThCols.HEATPUMP_AMBIENT_HEAT.value: {'line': None, 'fill': {'color': Colors.GREY.value}},
    TsThCols.DIRECT_EL_HEATER.value: {'line': None, 'fill': {'color': Colors.ORANGE.value}},
    TsThCols.GAS_BOILER.value: {'line': None, 'fill': {'color': Colors.DARK_GREEN.value}},
    TsThCols.DHW_STORAGE_CHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_downward_diagonal',
            'fg_color': Colors.PINK.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsThCols.DHW_STORAGE_DISCHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.PINK.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsThCols.BUFFER_STORAGE_CHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_downward_diagonal',
            'fg_color': Colors.LIGHT_RED.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsThCols.BUFFER_STORAGE_DISCHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.LIGHT_RED.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsThCols.BUFFER_STORAGE_CHARGE_HT.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_downward_diagonal',
            'fg_color': Colors.DARK_RED.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsThCols.BUFFER_STORAGE_DISCHARGE_HT.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.DARK_RED.value,
            'bg_color': Colors.GREY.value,
        }
    },
    TsThCols.TH_INERTIA_CHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_downward_diagonal',
            'fg_color': Colors.BROWN.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },
    TsThCols.TH_INERTIA_DISCHARGE.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.BROWN.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },
    TsOtherCols.TEMP_AIR.value: {'line': {'dash_type': 'dash', 'color': Colors.BLACK, 'width': 1.0}},
    TsElCols.HELPER_EL.value: {'line': None, 'fill': {'color': Colors.WHITE.value}},
    TsThCols.HELPER_TH.value: {'line': None, 'fill': {'color': Colors.WHITE.value}},
}

BARCHART_STYLES_XLSX = {
    # Investments
    SmryCols.TOTAL_COSTS.value: {
        'marker': {
            'type': 'x',
            'size': 10,
            'fill': None,
            'line': {'color': Colors.BLACK.value}
        }
    },
    SmryCols.CAPEX_PV.value: {'line': None, 'fill': {'color': Colors.YELLOW.value}},
    SmryCols.CAPEX_BATT.value: {'line': None, 'fill': {'color': Colors.BLUE.value}},
    SmryCols.CAPEX_HEAT.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.ORANGE.value,
            'bg_color': Colors.RED.value,
        }
    },
    SmryCols.CAPEX_REFURB.value: {'line': None, 'fill': {'color': Colors.BROWN.value}},

    # Yearly costs
    SmryCols.EL_COST_HH.value: {'line': None, 'fill': {'color': Colors.BLUE.value}},
    SmryCols.EL_COST_HEAT.value: {'line': None, 'fill': {'color': Colors.ORANGE.value}},
    SmryCols.EL_COST_EMOB.value: {'line': None, 'fill': {'color': Colors.NEON.value}},
    SmryCols.FUEL_COSTS.value: {'line': None, 'fill': {'color': Colors.DARK_GREEN.value}},
    SmryCols.FEEDIN_REV.value: {'line': None, 'fill': {'color': Colors.YELLOW.value}},
    SmryCols.CO2_COSTS.value: {'line': None, 'fill': {'color': Colors.BLACK.value}},
    SmryCols.CAPEX_ANNUITY.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.GREY.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },

    # Installed capacities
    SmryCols.PV_KWP.value: {'line': None, 'fill': {'color': Colors.YELLOW.value}},
    SmryCols.BATT_KWH.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.BLUE.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },
    SmryCols.HP_KW.value: {'line': None, 'fill': {'color': Colors.ORANGE.value}},
    SmryCols.BOILER_KW.value: {'line': None, 'fill': {'color': Colors.DARK_GREEN.value}},
    SmryCols.DEH_KW.value: {'line': None, 'fill': {'color': Colors.DARK_RED.value}},
    SmryCols.AC_KW.value: {'line': None, 'fill': {'color': Colors.PINK.value}},
    SmryCols.BUFFER_KWH.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.ORANGE.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },
    SmryCols.DHW_KWH.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.DARK_RED.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },
    SmryCols.TH_INERTIA.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.LIGHT_BROWN.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },

    # Energy balances electricity
    SmryCols.PV_USED_HH.value: {'line': None, 'fill': {'color': Colors.YELLOW.value}},
    SmryCols.PV_USED_HEAT.value: {'line': None, 'fill': {'color': Colors.ORANGE.value}},
    SmryCols.PV_USED_EMOB.value: {'line': None, 'fill': {'color': Colors.NEON.value}},
    SmryCols.PV_FEEDIN.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.YELLOW.value,
            'bg_color': Colors.GREY.value,
        }
    },
    SmryCols.EL_IMP_HH.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.BLUE.value,
            'bg_color': Colors.LIGHT_GREY.value,
        }
    },
    SmryCols.EL_IMP_HEAT.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.ORANGE.value,
            'bg_color': Colors.GREY.value,
        }
    },
    SmryCols.EL_IMP_EMOB.value: {
        'line': None,
        'pattern': {
            'pattern': 'dark_upward_diagonal',
            'fg_color': Colors.NEON.value,
            'bg_color': Colors.GREY.value,
        }
    },

    # Energy balances heat
    SmryCols.TH_GEN_BOILER.value: {'line': None, 'fill': {'color': Colors.DARK_GREEN.value}},
    SmryCols.TH_GEN_HP.value: {'line': None, 'fill': {'color': Colors.ORANGE.value}},
    SmryCols.TH_GEN_DEH.value: {'line': None, 'fill': {'color': Colors.DARK_RED.value}},
    SmryCols.TH_GEN_AC.value: {'line': None, 'fill': {'color': Colors.PINK.value}},

    # Self-sufficiency
    SmryCols.SELF_CONS.value: {'line': None, 'fill': {'color': Colors.YELLOW.value}},
    SmryCols.SELF_SUFF.value: {'line': None, 'fill': {'color': Colors.LIGHT_YELLOW.value}},
    SmryCols.SELF_SUFF_HH.value: {'line': None, 'fill': {'color': Colors.BLUE.value}},
    SmryCols.SELF_SUFF_HEAT.value: {'line': None, 'fill': {'color': Colors.ORANGE.value}},
    SmryCols.SELF_SUFF_EMOB.value: {'line': None, 'fill': {'color': Colors.NEON.value}},
}

RES_TS_GRAPHS = [
    {
        'graph_type_1': {'type': 'line'},
        'graph_type_2': {'type': 'line'},
        'graph_type_2_cols': ['Temp. air in °C'],
        'y2_axis': 1,
        'position': "B3",
        'x_y_scale': [2.5, 1.2],
        'cols_to_include': TsOtherCols.to_list(),
        'column_styles': TS_PLOT_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'area', 'subtype': 'stacked'},
        'graph_type_2': {'type': 'line'},
        'graph_type_2_cols': [TsElCols.EL_DEMAND.value, TsElCols.EL_DEM_HEAT.value,
                              TsElCols.EL_DEM_TOT.value],
        'position': "B21",
        'x_y_scale': [2.5, 1.5],
        'cols_to_include': TsElCols.to_list(),
        'column_styles': TS_PLOT_STYLES_XLSX,
        'delete_series_from_legend': [6]
    },
    {
        'graph_type_1': {'type': 'area', 'subtype': 'stacked'},
        'graph_type_2': {'type': 'line'},
        'graph_type_2_cols': ['Th. Demand', 'Roomheat Demand', 'Qmax solar-thermal', 'Heat gains'],
        'position': "B43",
        'x_y_scale': [2.5, 1.7],
        'cols_to_include': TsThCols.to_list(),
        'column_styles': TS_PLOT_STYLES_XLSX,
        'delete_series_from_legend': [14]
    },
]

RES_BARCHART_GRAPHS = [
    {
        'graph_type_1': {'type': 'column', 'subtype': 'stacked'},
        'title': "Investments (present values, incl. reinvests)",
        'y_axis_title': "Investment costs in €",
        'position': "B13",
        'x_y_scale': [1.5, 1.3],
        'cols_to_include': [SmryCols.CAPEX_PV.value, SmryCols.CAPEX_BATT.value, SmryCols.CAPEX_HEAT.value,
                            SmryCols.CAPEX_REFURB.value],
        'column_styles': BARCHART_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column', 'subtype': 'stacked'},
        'graph_type_2': {'type': 'scatter'},
        'graph_type_2_cols': [SmryCols.TOTAL_COSTS.value],
        'title': "Yearly costs (incl. investment annuities)",
        'y_axis_title': "Average yearly costs in €/a",
        'position': "N13",
        'x_y_scale': [1.5, 1.3],
        'cols_to_include': [SmryCols.EL_COST_HH.value, SmryCols.EL_COST_HEAT.value, SmryCols.EL_COST_EMOB.value,
                            SmryCols.FUEL_COSTS.value, SmryCols.FEEDIN_REV.value, SmryCols.CO2_COSTS.value,
                            SmryCols.CAPEX_ANNUITY.value, SmryCols.TOTAL_COSTS],
        'column_styles': BARCHART_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column',},
        'title': "Installed capacities electricity",
        'y_axis_title': "Installed capacities in kW / kWh",
        'position': "B33",
        'x_y_scale': [1.5, 1.3],
        'cols_to_include': [SmryCols.PV_KWP.value, SmryCols.BATT_KWH.value],
        'column_styles': BARCHART_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column', },
        'title': "Installed capacities heat",
        'y_axis_title': "Installed capacities in kW / kWh",
        'position': "N33",
        'x_y_scale': [1.5, 1.3],
        'cols_to_include': [SmryCols.HP_KW.value, SmryCols.BOILER_KW.value, SmryCols.DEH_KW.value, SmryCols.AC_KW.value,
                            SmryCols.DHW_KWH.value, SmryCols.BUFFER_KWH.value, SmryCols.TH_INERTIA.value],
        'column_styles': BARCHART_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column', 'subtype': 'stacked'},
        'title': "Yearly energy balances electricity",
        'y_axis_title': "yearly energy balance in kWh",
        'position': "B53",
        'x_y_scale': [1.5, 1.3],
        'cols_to_include': [SmryCols.PV_USED_HH.value, SmryCols.PV_USED_HEAT.value, SmryCols.PV_USED_EMOB.value,
                            SmryCols.PV_FEEDIN.value, SmryCols.EL_IMP_HH.value, SmryCols.EL_IMP_HEAT.value,
                            SmryCols.EL_IMP_EMOB.value],
        'column_styles': BARCHART_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column', 'subtype': 'stacked'},
        'title': "Yearly energy balances heat",
        'y_axis_title': "yearly energy balance in kWh",
        'position': "N53",
        'x_y_scale': [1.5, 1.3],
        'cols_to_include': [SmryCols.TH_GEN_BOILER.value, SmryCols.TH_GEN_HP.value, SmryCols.TH_GEN_DEH.value,
                            SmryCols.TH_GEN_AC.value],
        'column_styles': BARCHART_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column', },
        'title': "Self-sufficiency",
        'y_axis_title': "ratio 0 - 1",
        'position': "B73",
        'x_y_scale': [1.5, 1.3],
        'cols_to_include': [SmryCols.SELF_CONS.value, SmryCols.SELF_SUFF.value, SmryCols.SELF_SUFF_HH.value,
                            SmryCols.SELF_SUFF_HEAT.value, SmryCols.SELF_SUFF_EMOB.value],
        'column_styles': BARCHART_STYLES_XLSX,
    },
]


def create_keyfacts_summary(res: ScenarioResults, sce_data: ScenarioData, printout=False) -> pd.Series:
    """
    Generates a summary of key facts and metrics for the given scenario results and input data. The function
    aggregates data such as costs, installed capacities, yearly energy balances, and other performance parameters.
    It calculates various energy and cost-related metrics based on the input parameters and scenario data.

    :param res: ScenarioResults
        Object containing all result variables including costs, energy balances, and installed capacities.
    :param sce_data: ScenarioData
        Object consisting of scenario input data like technical and economic parameters.
    :param printout: bool
        Flag indicating whether to print the summary results.
    :return: pd.Series
        A pandas Series containing the summarized key facts and metrics for the given scenario.
    """

    key_facts = pd.Series()

    # Add costs to key_facts
    key_facts["COSTS"] = ""
    key_facts[SmryCols.TOTAL_COSTS.value] = res.get_ti_sum('tot_costs', 'sum')
    key_facts[SmryCols.OPEX_TOTAL.value] = (res.get_ti_sum('var_costs|fix_costs', 'sum')
                                            - res.get_ti_sum('revenues', 'sum'))
    key_facts[SmryCols.CO2_COSTS.value] = res.get_ti_sum('co2_costs', 'sum')
    key_facts[SmryCols.CAPEX_ANNUITY.value] = res.get_ti_sum('inv_costs', 'sum')
    key_facts[SmryCols.CAPEX.value] = calc_present_value(
        key_facts[SmryCols.CAPEX_ANNUITY.value], duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE
    )
    key_facts[SmryCols.CAPEX_PV.value] = calc_present_value(
        res.get_ti_sum('inv_costs', 'PV: '), duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE
    )
    key_facts[SmryCols.CAPEX_BATT.value] = calc_present_value(
        res.get_ti_sum('inv_costs', 'ElStorage'), duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE
    )
    key_facts[SmryCols.CAPEX_REFURB.value] = calc_present_value(
        res.get_ti_sum('inv_costs', 'Refurbishment'), duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE
    )
    key_facts[SmryCols.CAPEX_HEAT.value] = calc_present_value(
        res.get_ti_sum('inv_costs', 'CombustionHeater|Heatpump|DirectElHeater|MultiTempStorage|ThStorage'),
        duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE
    )
    key_facts[SmryCols.EL_COST_HH.value] = res.get_ti_sum('var_costs', 'ElImport: el_std') \
                                           + res.get_ti_sum('fix_costs', 'ElImport: el_std')
    key_facts[SmryCols.EL_COST_HEAT.value] = res.get_ti_sum('var_costs', 'ElImport: el_hp') \
                                             + res.get_ti_sum('fix_costs', 'ElImport: el_hp')
    key_facts[SmryCols.EL_COST_EMOB.value] = res.get_ti_sum('var_costs', 'ElImport: el_emob') \
                                             + res.get_ti_sum('fix_costs', 'ElImport: el_emob')
    key_facts[SmryCols.FUEL_COSTS.value] = res.get_ti_sum('var_costs', 'CombustionHeater')
    key_facts[SmryCols.FEEDIN_REV.value] = - res.get_ti_sum('revenues', 'ElExport')

    # Add installed capacities to key facts
    key_facts["INSTALLED CAPACITIES"] = ""
    key_facts[SmryCols.PV_KWP.value] = res.get_ti_sum('p_inst', 'PV')
    key_facts[SmryCols.HP_KW.value] = res.get_ti_sum('p_inst', 'Heatpump: hp')
    key_facts[SmryCols.AC_KW.value] = res.get_ti_sum('p_inst', 'Heatpump: ac')
    key_facts[SmryCols.DEH_KW.value] = res.get_ti_sum('p_inst', 'DirectElHeater')
    key_facts[SmryCols.BOILER_KW.value] = res.get_ti_sum('p_inst', 'CombustionHeater')
    key_facts[SmryCols.BATT_KWH.value] = res.get_ti_sum('c_inst', 'ElStorage')
    key_facts[SmryCols.BUFFER_L.value] = res.get_ti_sum('c_inst_l', 'buffsto')
    key_facts[SmryCols.BUFFER_KWH.value] = (
            key_facts[SmryCols.BUFFER_L.value] * const.TH_CAP_WATER_Wh_per_kg_K * 10 / 1000
    )
    key_facts[SmryCols.DHW_L.value] = res.get_ti_sum('c_inst_l', 'dhwsto')
    key_facts[SmryCols.DHW_KWH.value] = (
            res.get_ti_sum('c_inst_l', 'dhwsto') * const.TH_CAP_WATER_Wh_per_kg_K * 55 / 1000
    )
    key_facts[SmryCols.TH_INERTIA.value] = res.get_ti_sum('c_inst', 'ThInertia')

    # Add energy balances to key facts
    key_facts["YEARLY ENERGY BALANCES"] = ""
    key_facts[SmryCols.PV_GEN.value] = res.get_en_balance('el', 'src', 'PV')
    key_facts[SmryCols.PV_GEN_CURTAILED.value] = (
        res.find_vals('yearly_sums', 'sums_others', 'pmax_t').sum().sum()
        - key_facts[SmryCols.PV_GEN.value])
    key_facts[SmryCols.PV_GEN_USED.value] = (
            res.get_en_balance('el', 'src', 'el_pv-el_hp|el_pv-el_imp|el_pv-el_emob') -
            res.get_en_balance('el', 'src', 'ElectricLine:.*-el_pv')
    )
    key_facts[SmryCols.PV_FEEDIN.value] = -(key_facts[SmryCols.PV_GEN.value] - key_facts[SmryCols.PV_GEN_USED.value])
    key_facts[SmryCols.EL_IMP_HH.value] = res.get_en_balance('el', 'src', 'ElImport: el_std')
    key_facts[SmryCols.EL_IMP_HEAT.value] = res.get_en_balance('el', 'src', 'ElImport: el_hp')
    key_facts[SmryCols.EL_IMP_EMOB.value] = res.get_en_balance('el', 'src', 'ElImport: el_emob')
    key_facts[SmryCols.EL_DEM_HH.value] = res.get_en_balance('el', 'sink', 'ElDemand: el_hh')
    key_facts[SmryCols.EL_DEM_HP.value] = res.get_en_balance('el', 'sink', 'Heatpump: hp')
    key_facts[SmryCols.EL_DEM_AC.value] = res.get_en_balance('el', 'sink', 'Heatpump: ac')
    key_facts[SmryCols.EL_DEM_DEH.value] = res.get_en_balance('el', 'sink', 'deh')
    key_facts[SmryCols.EL_DEM_HEAT_TOTAL.value] = (
            key_facts[SmryCols.EL_DEM_HP.value] +
            key_facts[SmryCols.EL_DEM_AC.value] +
            key_facts[SmryCols.EL_DEM_DEH.value]
    )
    key_facts[SmryCols.EL_DEM_EMOB.value] = res.get_en_balance('el', 'sink', 'ElDemand: emob')
    key_facts[SmryCols.EL_DEM_TOTAL.value] = (
            res.get_en_balance('el', 'src', 'ElImport|PV') -
            res.get_en_balance('el', 'sink', 'ElExport')
    )
    key_facts[SmryCols.TH_DEM_ROOM.value] = res.get_en_balance('th', 'sink', 'RoomheatDemand')
    key_facts[SmryCols.TH_DEM_DHW.value] = res.get_en_balance('th', 'sink', 'th_dhw')
    key_facts[SmryCols.TH_DEM_TOTAL.value] = (
            key_facts[SmryCols.TH_DEM_ROOM.value] + key_facts[SmryCols.TH_DEM_DHW.value]
    )
    key_facts[SmryCols.TH_GEN_BOILER.value] = res.get_en_balance('th', 'src', 'CombustionHeater: boiler')
    key_facts[SmryCols.TH_GEN_HP.value] = res.get_en_balance('th', 'src', 'Heatpump: hp')
    key_facts[SmryCols.TH_GEN_AC.value] = res.get_en_balance('th', 'src', 'Heatpump: ac')
    key_facts[SmryCols.TH_GEN_DEH.value] = res.get_en_balance('th', 'src', 'deh')

    # Calculating shares of PV generation used for which purpose
    grey_stored_el = res.get_en_balance('el', 'src', 'ElectricLine:.*-el_pv')
    total_pv_es_cons = res.get_en_balance('el', 'sink', 'el_pv-el_imp|el_pv-el_hp|el_pv-el_emob')

    heat_cons_share = res.get_en_balance('el', 'src', 'el_pv-el_hp') / total_pv_es_cons
    ev_cons_share = res.get_en_balance('el', 'src', 'el_pv-el_emob') / total_pv_es_cons

    key_facts[SmryCols.PV_USED_HEAT.value] = heat_cons_share * (total_pv_es_cons - grey_stored_el)
    key_facts[SmryCols.PV_USED_EMOB.value] = ev_cons_share * (total_pv_es_cons - grey_stored_el)
    key_facts[SmryCols.PV_USED_HH.value] = (
            key_facts[SmryCols.PV_GEN_USED.value] - key_facts[SmryCols.PV_USED_HEAT.value]
            - key_facts[SmryCols.PV_USED_EMOB.value]
    )

    # Add other parameters to key facts
    key_facts["OTHER PARAMETERS"] = ""
    key_facts[SmryCols.SELF_SUFF.value] = (
        key_facts[SmryCols.PV_GEN_USED.value] / key_facts[SmryCols.EL_DEM_TOTAL.value]
        if key_facts[SmryCols.EL_DEM_TOTAL.value] > 0 else None
    )
    key_facts[SmryCols.SELF_CONS.value] = (
        key_facts[SmryCols.PV_GEN_USED.value] / key_facts[SmryCols.PV_GEN.value]
        if key_facts[SmryCols.PV_GEN.value] > 0 else None
    )
    key_facts[SmryCols.SELF_SUFF_HEAT.value] = (
        key_facts[SmryCols.PV_USED_HEAT.value] / key_facts[SmryCols.EL_DEM_HEAT_TOTAL.value]
        if key_facts[SmryCols.EL_DEM_HEAT_TOTAL.value] > 0 else None
    )
    key_facts[SmryCols.SELF_SUFF_EMOB.value] = (
        key_facts[SmryCols.PV_USED_EMOB.value] / key_facts[SmryCols.EL_DEM_EMOB.value]
        if key_facts[SmryCols.EL_DEM_EMOB.value] > 0 else None
    )
    key_facts[SmryCols.SELF_SUFF_HH.value] = (
        key_facts[SmryCols.PV_USED_HH.value] / key_facts[SmryCols.EL_DEM_HH.value]
        if key_facts[SmryCols.EL_DEM_HH.value] > 0 else None
    )
    key_facts[SmryCols.GREEN_CHARGE_SHARE.value] = 1 - (grey_stored_el/res.get_en_balance('el', 'sink', 'ElStorage'))

    if res.get_en_balance('el', 'sink', 'Heatpump: hp') > 0:
        key_facts[SmryCols.COP.value] = (
                res.get_en_balance('th', 'src', 'Heatpump: hp') /
                res.get_en_balance('el', 'sink', 'Heatpump: hp')
        )
        key_facts[SmryCols.SCOP_WINTER.value] = (
                res.get_en_balance('th', 'src', 'Heatpump: hp', season=const.Season.Winter) /
                res.get_en_balance('el', 'sink', 'Heatpump: hp', season=const.Season.Winter)
        )
    else:
        key_facts[SmryCols.COP.value] = None
        key_facts[SmryCols.SCOP_WINTER.value] = None

    if key_facts[SmryCols.BATT_KWH.value] > 0:
        key_facts[SmryCols.BATT_CYCLES.value] = (
                res.get_en_balance('el', 'src', 'ElStorage: batt1') *
                sce_data.tech_data.batt['batt1']['lifetime'] /
                res.get_ti_sum('c_inst', 'ElStorage: batt1')
        )
    else:
        key_facts[SmryCols.BATT_CYCLES.value] = None

    key_facts[SmryCols.SPEC_PV_GEN.value] = (
        (key_facts[SmryCols.PV_GEN.value] + key_facts[SmryCols.PV_GEN_CURTAILED.value])
        / key_facts[SmryCols.PV_KWP.value]
        if key_facts[SmryCols.PV_KWP.value] > 0 else None
    )
    key_facts[SmryCols.LCOE_PV.value] = (
        res.get_ti_sum('tot_costs', 'PV') / key_facts[SmryCols.PV_GEN.value]
        if key_facts[SmryCols.PV_GEN.value] > 0 else None
    )
    key_facts[SmryCols.LCOE_SELF.value] = (
        (
                res.get_ti_sum('tot_costs', 'PV') +
                res.get_ti_sum('tot_costs', 'ElStorage') -
                res.get_ti_sum('revenues', 'ElExport')
        ) / key_facts[SmryCols.PV_GEN_USED.value]
        if key_facts[SmryCols.PV_GEN_USED.value] > 0 else None
    )
    key_facts[SmryCols.LCOE_TOTAL.value] = (
        (
                res.get_ti_sum('tot_costs', 'PV|ElStorage|ElImport') -
                res.get_ti_sum('revenues', 'ElExport')
        ) / key_facts[SmryCols.EL_DEM_TOTAL.value]
        if key_facts[SmryCols.EL_DEM_TOTAL.value] > 0 else None
    )
    if res.get_en_balance('th', 'src', 'Heatpump|deh|boiler') > 0:
        key_facts[SmryCols.LCOH.value] = (
            res.get_ti_sum('tot_costs', 'Heatpump|CombustionHeater|DirectElHeater|dhwsto|buffsto') +
            (key_facts[SmryCols.LCOE_SELF.value] * key_facts[SmryCols.PV_USED_HEAT.value]
             if key_facts[SmryCols.PV_USED_HEAT.value] > 0 else 0) +
            res.get_ti_sum('tot_costs', 'ElImport: el_hp')
            ) / res.get_en_balance('th', 'src', 'Heatpump|deh|boiler')
    else:
        key_facts[SmryCols.LCOH.value] = None

    if printout:
        print(key_facts.to_string())

    return key_facts


def define_ts_plots(res: ScenarioResults):
    """
    Defines time series data for plotting based on different seasons for electrical, thermal,
    and other parameters. Combines multiple components of the dataset and summarizes them
    for seasonal analysis.

    This function iterates over the defined seasons, calculates various balances for electrical
    and thermal energy, and collects other time series data (e.g., air temperature, CO2
    equivalents, and electricity prices). The processed time series data for each season is
    stored as an attribute of the input object `res`.

    :param res: An instance of ScenarioResults containing time series data.
    :type res: ScenarioResults
    :return: None
    :rtype: NoneType
    """
    for season in create_ts_multiindex().levels[0]:
        try:
            el_balance = {
                TsElCols.HELPER_EL.value: -res.get_ts(
                    'el', 'sink', 'ElExport|ElStorage', season=season).sum(axis=1),
                TsElCols.FEEDIN.value: res.get_ts(
                    'el', 'sink', 'ElExport', season=season).sum(axis=1),
                TsElCols.BATTERY_CHARGE.value: res.get_ts(
                    'el', 'sink', 'ElStorage', season=season).sum(axis=1),
                TsElCols.BATTERY_DISCHARGE.value: res.get_ts(
                    'el', 'src', 'ElStorage', season=season).sum(axis=1),
                TsElCols.PV.value: res.get_ts(
                    'el', 'src', 'PV', season=season).sum(axis=1),
                TsElCols.PV_CURTAILED.value: (
                        res.ts_others.filter(regex='pmax_t').loc[season].sum(axis=1)
                        -  res.get_ts('el', 'src', 'PV', season=season).sum(axis=1)),
                TsElCols.GRID_DEMAND.value: res.get_ts(
                    'el', 'src', 'ElImport: el_std', season=season).sum(axis=1),
                TsElCols.GRID_HEATPUMP.value: res.get_ts(
                    'el', 'src', 'ElImport: el_hp', season=season).sum(axis=1),
                TsElCols.GRID_EMOB.value: res.get_ts(
                    'el', 'src', 'ElImport: el_emob', season=season).sum(axis=1),
                TsElCols.EL_DEMAND.value: res.get_ts(
                    'el', 'sink', 'ElDemand: el_hh', season=season).sum(axis=1),
                TsElCols.EL_DEM_HEAT.value:
                    res.get_ts(
                        'el', 'sink', 'ElDemand: el_hh|Heatpump|DirectElHeater', season=season).sum(axis=1),
                TsElCols.EL_DEM_TOT.value:
                    res.get_ts(
                        'el', 'sink', 'ElDemand|Heatpump|DirectElHeater', season=season).sum(axis=1),
            }
        except Exception as e:
            print("Error in el_balance timeseries plot definition:", e)
            el_balance = {}

        try:
            th_balance = {
                TsThCols.HELPER_TH.value: -res.get_ts(
                    'th', 'sink', 'ThInertia|buffsto|dhwsto', season=season).sum(axis=1),
                TsThCols.DHW_STORAGE_CHARGE.value: res.get_ts(
                    'th', 'sink', 'MultiTempStorage: dhwsto', season=season).sum(
                    axis=1),
                TsThCols.BUFFER_STORAGE_CHARGE.value:
                    res.get_ts(
                        'th', 'sink', 'MultiTempStorage: buffsto.*ht_flow', season=season).iloc[:, 1],
                TsThCols.BUFFER_STORAGE_CHARGE_HT.value:
                    res.get_ts(
                        'th', 'sink', 'MultiTempStorage: buffsto.*ht_flow', season=season).iloc[:, 0],
                TsThCols.TH_INERTIA_CHARGE.value: res.get_ts(
                    'th', 'sink', 'ThInertia', season=season).sum(axis=1),
                TsThCols.HEAT_GAINS_USED.value: (
                        res.get_ts('th', 'src', 'HeatGains', season=season).sum(axis=1)
                        - res.get_ts('th', 'sink', 'excess_heat', season=season).sum(axis=1)
                ),
                TsThCols.SOLAR_THERMAL.value: res.get_ts(
                    'th', 'src', 'SolarThermal', season=season).sum(axis=1),
                TsThCols.HEATPUMP_EL.value: res.get_ts(
                    'el', 'sink', 'Heatpump', season=season).sum(axis=1),
                TsThCols.HEATPUMP_AMBIENT_HEAT.value: res.get_ts(
                    'th', 'sink', 'Heatpump', season=season).sum(axis=1),
                TsThCols.GAS_BOILER.value: res.get_ts(
                    'th', 'src', 'boiler', season=season).sum(axis=1),
                TsThCols.DIRECT_EL_HEATER.value: res.get_ts(
                    'th', 'src', 'DirectElHeater', season=season).sum(axis=1),
                TsThCols.DHW_STORAGE_DISCHARGE.value:
                    res.get_ts(
                        'th', 'src', 'MultiTempStorage: dhwsto', season=season).sum(axis=1),
                TsThCols.BUFFER_STORAGE_DISCHARGE.value:
                    res.get_ts(
                        'th', 'src', 'MultiTempStorage: buffsto.*ht_flow', season=season).iloc[:, 1],
                TsThCols.BUFFER_STORAGE_DISCHARGE_HT.value:
                    res.get_ts(
                        'th', 'src', 'MultiTempStorage: buffsto.*ht_flow', season=season).iloc[:, 0],
                TsThCols.TH_INERTIA_DISCHARGE.value: res.get_ts(
                    'th', 'src', 'ThInertia', season=season).sum(axis=1),
                TsThCols.ROOMHEAT_DEMAND.value: res.get_ts(
                    'th', 'sink', 'RoomheatDemand', season=season).sum(axis=1),
                TsThCols.TH_DEMAND.value: (
                        res.get_ts('th', 'sink', 'RoomheatDemand', season=season).sum(axis=1)
                        + res.get_ts('th', 'sink', 'ThDemand', season=season).sum(axis=1)
                ),
                TsThCols.HEAT_GAINS.value: res.get_ts(
                    'th', 'src', 'HeatGains', season=season).sum(axis=1),
                TsThCols.QMAX_SOLAR_THERMAL.value:
                    res.ts_others.filter(regex='(qmax_t.*ht_flow)(?!.*high)').loc[season].sum(axis=1),
            }
        except Exception as e:
            print("Error in th_balance timeseries plot definition:", e)
            th_balance = {}

        try:
            other_ts = {
                TsOtherCols.TEMP_AIR.value: res.ts_inputs[PvgisDataCols.Air_temp].loc[season],
                TsOtherCols.CO2_EQ.value: res.ts_inputs[CO2ElecCols.Co2eq_lca].loc[season],
                TsOtherCols.ELEC_PRICE_HH.value: res.find_vals(
                    'ts_others', 'elec. price: el_std').sum(axis=1).loc[season],
                TsOtherCols.ELEC_PRICE_HEAT.value: res.find_vals(
                    'ts_others', 'elec. price: el_hp').sum(axis=1).loc[season],
                TsOtherCols.ELEC_PRICE_EMOB.value: res.find_vals(
                    'ts_others', 'elec. price: el_emob').sum(axis=1).loc[season],
            }
        except Exception as e:
            print("Error in ts_others timeseries plot definition:", e)
            other_ts = {}

        setattr(
            res,
            f'ts_plot_{season.value if hasattr(season, "value") else season}',
            pd.DataFrame({**other_ts, **el_balance, **th_balance})
        )


def write_scenario_results(
        sce_data: ScenarioData, m: ConcreteModel, ts_red_inputs: pd.DataFrame, results_folder: str
) -> ScenarioResults:
    """
    This function writes results of a scenario to the specified output folder. It processes the given
    scenario data, predictive model, and time-series reduction inputs to generate and save output files.
    Additionally, it prepares a summary of key facts for the scenario results and defines time-series
    plots for graphical representation.

    :param sce_data: Scenario data containing configuration and input parameters.
    :type sce_data: ScenarioData
    :param m: Optimization model containing variables, parameters, constraints, and results.
    :type m: ConcreteModel
    :param ts_red_inputs: Dataframe containing time-series reduction inputs used for scenario analysis.
    :type ts_red_inputs: pd.DataFrame
    :param results_folder: Directory path where the results files will be saved.
    :type results_folder: str
    :return: An instance of ScenarioResults containing the processed results and additional outputs.
    :rtype: ScenarioResults
    """

    # dump pickle for manual result postprocessing
    dump_pickle('last_scenario_results', [sce_data, m, ts_red_inputs])

    res = ScenarioResults(ts_red_inputs, m)
    res.key_facts = create_keyfacts_summary(res, sce_data)

    define_ts_plots(res)

    save_xlsx_wb(
        f"{results_folder}/OutputDetailed_{sce_data.scenario_name}.xlsx",
        res.to_dict(),
        graphs={
            f'ts_plot_{season.value if hasattr(season, "value") else season}':
                RES_TS_GRAPHS for season in create_ts_multiindex().levels[0]
        }
    )

    return res


def write_results_summary(results: list, results_folder: str, print_summary=True):
    """
    Generates a summary of results for all given scenarios, saves the summary and associated
    graph outputs to an Excel file, and optionally prints the summary to the console.

    :param results: A list containing results for different scenarios, where each result is
        expected to have specific attributes, such as key facts and time-series plot data.
    :type results: list
    :param results_folder: The folder path where the output summary file should be saved.
    :type results_folder: str
    :param print_summary: A flag indicating whether the summary should also be printed to
        the console. Defaults to True.
    :type print_summary: bool
    :return: None
    """

    res_summary = pd.DataFrame({sce: results[sce].key_facts for sce in results})
    res_output = {
        'summary': res_summary.T,
        **{
            f"{sce}_{res_key}": getattr(results[sce], res_key)
            for sce in results
            for res_key in results[sce].__dict__.keys() if "ts_plot" in res_key
        }
    }
    save_xlsx_wb(
        results_folder + "/OutputSummary.xlsx",
        res_output,
        graphs={
            'summary': RES_BARCHART_GRAPHS,
            **{
                res_key: RES_TS_GRAPHS
                for res_key in res_output.keys() if "ts_plot" in res_key
            }
        }
    )

    if print_summary:
        print(res_summary.to_string())



class ClimateDataCols(str, Enum):
    TEMP_AVG = "Avg. temperature"
    TEMP_MAX = "Max. temperature"
    TEMP_MIN = "Min. temperature"
    P_SOLAR_SUM = "Solar yield (kWh/kWp)"
    P_SOLAR_SUM_REL = "Solar yield rel. to avg. (%)"
    P_SOLAR_SUM_WEEKLY = "Solar yield per week (kWh/kWp)"
    HDD_SUM = "Heating degree days (°C*d)"
    HDD_SUM_REL = "HDD rel. to avg. (%)"
    HDD_WEEKLY = "HDD per week"
    CDD_SUM = "Cooling degree days (°C*d)"
    CDD_SUM_REL = "CDD rel. to avg. (%)"
    CDD_WEEKLY = "CDD per week"


CLIMATEDATA_STYLES_XLSX = {
    # Investments
    ClimateDataCols.TEMP_AVG.value: {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.BLACK.value}
        },
        'line': {'color': Colors.BLACK.value, 'width': 1.0}
    },
    ('AVG.', ClimateDataCols.TEMP_AVG): {
        'line': {'color': Colors.BLACK.value, 'width': 1.0,}
    },
    ('MIN.', ClimateDataCols.TEMP_AVG): {
        'line': {'color': Colors.BLUE.value, 'width': 1.0, 'dash_type': 'dash',}
    },
    ('MAX.', ClimateDataCols.TEMP_AVG): {
        'line': {'color': Colors.RED.value, 'width': 1.0, 'dash_type': 'dash', }
    },
    ClimateDataCols.TEMP_MIN.value: {
        'marker': {
            'type': 'circle',
            'size': 5,
            'fill': {'color': Colors.BLUE.value},
            'line': {'color': Colors.BLUE.value},
        },
        'line': {'color': Colors.BLUE.value, 'width': 1.0}
    },
    ClimateDataCols.TEMP_MAX.value: {
        'marker': {
            'type': 'circle',
            'size': 5,
            'fill': {'color': Colors.RED.value},
            'line': {'color': Colors.RED.value},
        },
        'line': {'color': Colors.RED.value, 'width': 1.0}
    },
    ClimateDataCols.P_SOLAR_SUM_REL.value: {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.YELLOW.value}
        },
        'line': {'color': Colors.YELLOW.value, 'width': 1.0}
    },
    ('AVG.', ClimateDataCols.P_SOLAR_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.YELLOW.value}
        },
        'line': {'color': Colors.YELLOW.value, 'width': 1.0}
    },
    ('MIN.', ClimateDataCols.P_SOLAR_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.LIGHT_YELLOW.value}
        },
        'line': {'color': Colors.LIGHT_YELLOW.value, 'width': 1.0, 'dash_type': 'dash',}
    },
    ('MAX.', ClimateDataCols.P_SOLAR_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.DARK_YELLOW.value}
        },
        'line': {'color': Colors.DARK_YELLOW.value, 'width': 1.0, 'dash_type': 'dash',}
    },
    ('AVG.', ClimateDataCols.P_SOLAR_SUM_WEEKLY.value): {'line': None, 'fill': {'color': Colors.YELLOW.value}},
    ('MIN.', ClimateDataCols.P_SOLAR_SUM_WEEKLY.value): {'line': None, 'fill': {'color': Colors.LIGHT_YELLOW.value}},
    ('MAX.', ClimateDataCols.P_SOLAR_SUM_WEEKLY.value): {'line': None, 'fill': {'color': Colors.DARK_YELLOW.value}},
    ('SEL. WEEK', ClimateDataCols.P_SOLAR_SUM.value): {'line': None, 'fill': {'color': Colors.DARK_GREY.value}},
    ClimateDataCols.HDD_SUM_REL.value: {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.RED.value}
        },
        'line': {'color': Colors.RED.value, 'width': 1.0}
    },
    ('AVG.', ClimateDataCols.HDD_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.RED.value}
        },
        'line': {'color': Colors.RED.value, 'width': 1.0}
    },
    ('MIN.', ClimateDataCols.HDD_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.LIGHT_RED.value}
        },
        'line': {'color': Colors.LIGHT_RED.value, 'width': 1.0, 'dash_type': 'dash', }
    },
    ('MAX.', ClimateDataCols.HDD_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.DARK_RED.value}
        },
        'line': {'color': Colors.DARK_RED.value, 'width': 1.0, 'dash_type': 'dash', }
    },
    ('AVG.', ClimateDataCols.HDD_WEEKLY.value): {'line': None, 'fill': {'color': Colors.RED.value}},
    ('MIN.', ClimateDataCols.HDD_WEEKLY.value): {'line': None, 'fill': {'color': Colors.LIGHT_RED.value}},
    ('MAX.', ClimateDataCols.HDD_WEEKLY.value): {'line': None, 'fill': {'color': Colors.DARK_RED.value}},
    ('SEL. WEEK', ClimateDataCols.HDD_SUM.value): {'line': None, 'fill': {'color': Colors.DARK_GREY.value}},
    ClimateDataCols.CDD_SUM_REL.value: {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.BLUE.value}
        },
        'line': {'color': Colors.BLUE.value, 'width': 1.0}
    },
    ('AVG.', ClimateDataCols.CDD_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.BLUE.value}
        },
        'line': {'color': Colors.BLUE.value, 'width': 1.0}
    },
    ('MIN.', ClimateDataCols.CDD_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.LIGHT_BLUE.value}
        },
        'line': {'color': Colors.LIGHT_BLUE.value, 'width': 1.0, 'dash_type': 'dash', }
    },
    ('MAX.', ClimateDataCols.CDD_SUM.value): {
        'marker': {
            'type': 'x',
            'size': 5,
            'fill': None,
            'line': {'color': Colors.DARK_BLUE.value}
        },
        'line': {'color': Colors.DARK_BLUE.value, 'width': 1.0, 'dash_type': 'dash', }
    },
    ('AVG.', ClimateDataCols.CDD_WEEKLY.value): {'line': None, 'fill': {'color': Colors.BLUE.value}},
    ('MIN.', ClimateDataCols.CDD_WEEKLY.value): {'line': None, 'fill': {'color': Colors.LIGHT_BLUE.value}},
    ('MAX.', ClimateDataCols.CDD_WEEKLY.value): {'line': None, 'fill': {'color': Colors.DARK_BLUE.value}},
    ('SEL. WEEK', ClimateDataCols.CDD_SUM.value): {'line': None, 'fill': {'color': Colors.DARK_GREY.value}},
}

CLIMATE_DATA_GRAPHS = [
    {
        'graph_type_1': {'type': 'scatter'},
        'title': "Development of yearly temperatures: avg. / min. / max.",
        'y_axis_title': "Temperature in °C",
        'position': "L3",
        'x_y_scale': [1.5, 1.5],
        'cols_to_include': [ClimateDataCols.TEMP_AVG, ClimateDataCols.TEMP_MAX, ClimateDataCols.TEMP_MIN],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
        'skip_last_n_rows': 4,
        'trendline': {ClimateDataCols.TEMP_AVG: 'linear', ClimateDataCols.TEMP_MIN: 'linear',
                      ClimateDataCols.TEMP_MAX: 'linear'}
    },
    {
        'graph_type_1': {'type': 'scatter'},
        'title': "Development of yearly solar yield, heating demand and cooling demand (relative)",
        'y_axis_title': "1 = 100%",
        'position': "L26",
        'x_y_scale': [1.5, 1.5],
        'cols_to_include': [ClimateDataCols.P_SOLAR_SUM_REL, ClimateDataCols.HDD_SUM_REL, ClimateDataCols.CDD_SUM_REL],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
        'skip_last_n_rows': 4,
        'trendline': {ClimateDataCols.P_SOLAR_SUM_REL: 'linear', ClimateDataCols.HDD_SUM_REL: 'linear',
                      ClimateDataCols.CDD_SUM_REL: 'linear'}
    },
    {
        'graph_type_1': {'type': 'line'},
        'title': "Average weekly temperatures: AVG. / MIN. / MAX. of years",
        'y_axis_title': "Temperature in °C",
        'position': "B3",
        'x_y_scale': [1.5, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.TEMP_AVG), ('MIN.',ClimateDataCols.TEMP_AVG),
                            ('MAX.',ClimateDataCols.TEMP_AVG),],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'line'},
        'title': "Average weekly temperatures: AVG. / MIN. / MAX. of years",
        'y_axis_title': "Temperature in °C",
        'position': "B3",
        'x_y_scale': [1.5, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.TEMP_AVG), ('MIN.', ClimateDataCols.TEMP_AVG),
                            ('MAX.', ClimateDataCols.TEMP_AVG), ],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'line'},
        'title': "Average weekly solar yield: AVG. / MIN. / MAX. of years",
        'y_axis_title': "kWh/kWp",
        'position': "N3",
        'x_y_scale': [1.5, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.P_SOLAR_SUM), ('MIN.', ClimateDataCols.P_SOLAR_SUM),
                            ('MAX.', ClimateDataCols.P_SOLAR_SUM), ],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'line'},
        'title': "Average weekly heating demand (HDD): AVG. / MIN. / MAX. of years",
        'y_axis_title': "HDD in °C*d",
        'position': "B26",
        'x_y_scale': [1.5, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.HDD_SUM), ('MIN.', ClimateDataCols.HDD_SUM),
                            ('MAX.', ClimateDataCols.HDD_SUM), ],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'line'},
        'title': "Average weekly cooling demand (CDD): AVG. / MIN. / MAX. of years",
        'y_axis_title': "CDD in °C*d",
        'position': "N26",
        'x_y_scale': [1.5, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.CDD_SUM), ('MIN.', ClimateDataCols.CDD_SUM),
                            ('MAX.', ClimateDataCols.CDD_SUM), ],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column'},
        'title': "Average weekly solar yield in seasons: AVG. / MIN. / MAX. of years",
        'y_axis_title': "kWh/kWp per week",
        'position': "B6",
        'x_y_scale': [1.0, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.P_SOLAR_SUM_WEEKLY), ('MIN.', ClimateDataCols.P_SOLAR_SUM_WEEKLY),
                            ('MAX.', ClimateDataCols.P_SOLAR_SUM_WEEKLY),
                            ('SEL. WEEK', ClimateDataCols.P_SOLAR_SUM.value),],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column'},
        'title': "Average weekly heating demand (HDD) in seasons: AVG. / MIN. / MAX. of years",
        'y_axis_title': "HDD in °C*d per week",
        'position': "J6",
        'x_y_scale': [1.0, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.HDD_WEEKLY), ('MIN.', ClimateDataCols.HDD_WEEKLY),
                            ('MAX.', ClimateDataCols.HDD_WEEKLY), ('SEL. WEEK', ClimateDataCols.HDD_SUM.value),],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },
    {
        'graph_type_1': {'type': 'column'},
        'title': "Average weekly cooling demand (CDD) in seasons: AVG. / MIN. / MAX. of years",
        'y_axis_title': "CDD in °C*d per week",
        'position': "R6",
        'x_y_scale': [1.0, 1.5],
        'cols_to_include': [('AVG.', ClimateDataCols.CDD_WEEKLY), ('MIN.', ClimateDataCols.CDD_WEEKLY),
                            ('MAX.', ClimateDataCols.CDD_WEEKLY), ('SEL. WEEK', ClimateDataCols.CDD_SUM.value),],
        'column_styles': CLIMATEDATA_STYLES_XLSX,
    },

]

def write_climate_data_xlsx_analysis(
        wb_dict: dict[str, pd.DataFrame] = None, results_folder: str=None, overwrite: bool = False,
):
    """
    Writes or updates an Excel file containing climate data analysis based on the provided
    dataframes. The function either creates a new workbook or modifies an existing one
    depending on the overwrite parameter. It visualizes the data in various sheets according
    to the provided graphs configuration.

    The climate data is saved in a standardized xlsx format, and the file location is
    determined by the results_folder parameter combined with a naming convention
    based on location constants LATITUDE and LONGITUDE.

    :param wb_dict: A dictionary where the keys represent sheet names and the values are the
        corresponding pandas DataFrames containing climate data. If not provided,
        it defaults to loaded pickle data.
    :param results_folder: A string specifying the folder where the xlsx file will be saved.
        If None, the default folder is determined by the calling code.
    :param overwrite: A boolean indicating whether the existing workbook should be overwritten.
        If True, the workbook is updated or created as necessary. If False, the workbook is
        only created if it does not already exist.
    :return: None
    """
    # read prepared data from pickle
    if wb_dict is None:
        wb_dict = load_pickle(create_climate_data_dump_id())

    # check if xlsx already exists and write / update
    fpath = results_folder + f"/ClimateDataAnalysis_{const.LATITUDE}_{const.LONGITUDE}.xlsx"
    if overwrite:
        save_xlsx_wb(
            fpath,
            wb_dict,
            graphs={
                'years': CLIMATE_DATA_GRAPHS[:2],
                'months summary': CLIMATE_DATA_GRAPHS[2:7],
                'weeks summary': CLIMATE_DATA_GRAPHS[2:7],
                'seasons summary': CLIMATE_DATA_GRAPHS[7:10],
            }
        )
        print(f"{'updated' if os.path.exists(fpath) else 'wrote'} {fpath}")
    else:
        if not os.path.exists(fpath):
            save_xlsx_wb(
                fpath,
                wb_dict,
                graphs={
                    'years': CLIMATE_DATA_GRAPHS[:2],
                    'months summary': CLIMATE_DATA_GRAPHS[2:7],
                    'weeks summary': CLIMATE_DATA_GRAPHS[2:7],
                    'seasons summary': CLIMATE_DATA_GRAPHS[7:10],
                }
            )
            print(f"wrote {fpath}")



if __name__ == '__main__':

    # write_climate_data_xlsx_analysis(results_folder=const.PATH_TO_WD + "/data/output/")

    # debug write_results_summary
    results = load_pickle('last_results')
    write_results_summary(results, results_folder=create_result_folder('last_results'))

    # # debug write_scenario_results
    # sce_data, m, ts_red_inputs = load_pickle('last_scenario_results')
    # write_scenario_results(sce_data, m, ts_red_inputs, results_folder=create_result_folder('last_results'))
