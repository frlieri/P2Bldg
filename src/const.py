"""
src/const.py

This module centralizes project-wide constants, enumerations and configuration
values used across the building and battery optimization codebase.

Contents and responsibilities:

- Environment and paths:
    - Sets essential environment variables (e.g. `NEOS_EMAIL`) and defines
      precomputed data paths derived from `PATH_TO_WD`.
- Enumerations:
    - Defines typed enums for input data keys, timeseries step sizes, column
      names, German states, seasons and typeday categories.
- Timeseries and modeling defaults:
    - Default resolution and modeled year, timeseries reduction settings,
      hours in a year and TS per hour constants.
- Economic and lifetime parameters:
    - Nominal and real interest rates, inflation, project duration.
- Limits and location defaults:
    - Large numeric caps for sizing and geographic defaults (latitude,
      longitude, modeled temperature).
- Building and household base data:
    - Default household count, reference temperatures, heating area/volume,
      ventilation and internal gains.
- Technology specifications:
    - Preloads technology lookup tables (e.g. `COP_TABLE`) and solar thermal
      collector performance specs used by model components.

Notes:

- Values here are intended as project defaults and may be overridden by
  scenario inputs or external configuration files.
- The module imports CSV resources from the repository; ensure `PATH_TO_WD`
  is set correctly and resource files exist.
"""

import os
import pandas as pd
from enum import Enum

from src.config import PATH_TO_WD, PATH_TO_SRC_DIR

PRECALC_FOLDER = PATH_TO_WD + "/data/resources/precalc/"


class InputData(str, Enum):
    PvgisData = 'pvgis_data'
    DhwDdemand = 'dhw_load'
    ElDemand = 'el_load'
    StatEmobDemand = 'ev_load_fix'
    FlexEmobDemand = 'ev_load_flex'
    CO2perKwhEl = 'CO2_per_kwh_el'
    ElPrice = 'el_price_EUR_per_kWh'
    CostFactors = 'cost_factors'


class TSStepSize(str, Enum):
    HOURLY = '1h'
    QUARTER_HOURLY = '15min'
    DAILY = '24h'


class TimeseriesCols(str, Enum):
    Step = 'timestep'
    TSDay = 'day_of_ts'
    DTime = 'datetime'
    Hour = 'hour'
    Yday = 'day_of_year'
    Week = 'week'
    Month = 'month'
    Year = 'year'
    Typeday = 'typeday'
    Season = 'season'

class State(str, Enum):
    BAYERN = 'BY'
    BADEN_WUERTTEMBERG = 'BW'
    BERLIN = 'BE'
    BRANDENBURG = 'BB'
    BYP = 'BYP'
    BREMEN = 'HB'
    HESSEN = 'HE'
    HAMBURG = 'HH'
    MECKLENBURG_VORPOMMERN = 'MV'
    NIEDERSACHSEN = 'NI'
    NORDRHEIN_WESTFALEN = 'NW'
    RHEINLAND_PFALZ = 'RP'
    SCHLESWIG_HOLSTEIN = 'SH'
    SAARLAND = 'SL'
    SACHSEN = 'SN'
    SACHSEN_ANHALT = 'ST'
    THUERINGEN = 'TH'


class Season(str, Enum):
    Summer = 'summer'
    Transition = 'transition'
    Winter = 'winter'

SEASON_PERIODS = {
    # format: [(start, end), ...] -- 'mm-dd'
    Season.Summer.value: [('05-15', '09-14')],
    Season.Winter.value: [('11-01', '03-20')],
    Season.Transition.value: [('09-15', '10-31'), ('03-21', '05-14')]
}

class Typeday(str, Enum):
    Workday = 'workday'
    Saturday = 'saturday'
    Sunday = 'sunday'


class TS_RES(int, Enum):
    HOURLY = 1
    QUARTER_HOURLY = 4


COUNTRY = 'DE'
STATE = State.BAYERN

HOURS_IN_YEAR: int = 8760
TS_PER_HOUR: int = TS_RES.HOURLY.value
MODELED_YEAR: int = 2023

# Timeseries reduction
CLIMATE_DATA_YEARS: list[int] = [2005, 2023]
APPLY_TIMESERIES_REDUCTION: bool = True


# KALKULATIONSZINS
NOMINAL_INTEREST_RATE = 0.04
INFLATION_RATE = 0.02
REAL_INTEREST_RATE = (1 + NOMINAL_INTEREST_RATE) / (1 + INFLATION_RATE) -1

# Laufzeit in Jahren
DURATION = 30

MAX_AREA = 10**6
MAX_P = 10**6
MAX_C = 10**6
MAX_C_in_l = 10**9

# location
LATITUDE, LONGITUDE = 48.163, 11.510
TEMP_EARTH = 10


# HOUSEHOLD DATA
NR_OF_HOUSEHOLDS = 1
T_OUT_REF = -14
NOM_TEMP_INDOOR = 20
NIGHT_TEMP_INDOOR = 16
HEATING_THRESHOLD = 16
YEARLY_EL_CONSUMPTION_kWh = 5500

# BUILDING DATA
HEATING_AREA_m2 = 231
# V_e
HEATING_VOLUME_m3 = 611
#  V_L = (0.76 EFH, 0.8 MFH) * V_e
VENT_RATE = 0.8 * HEATING_VOLUME_m3
#  n_L = n_Lüft (0.5, 0.4 with ventilator) * (1 - WRG (75-90%)) + n_infiltration (0,04 @n_Lüft=0.5)
AIR_CHANGE = 0.54
# Q_I = t_HP (185 - 225 d/a) * Qdot_I (EFH 50Wh/(m2d), MFH 100) * A_EB (heated area)
INTERNAL_GAINS = 0.1/24 * HEATING_AREA_m2

TH_CAP_WATER_Wh_per_kg_K = 1.163


# CO2 emissions of imported electricity in Germany
CO2_g_per_kWh_EL_LCA = 395.1  # g/kWh in 2023
CO2_g_per_kWh_EL_DIR = 329.1  # g/kWh in 2023


# Optional parameters if peak load tariffs are applied (Hochlastzeitfenster atypische Netznutzung)
HLTF_PEAK_LOAD_LIMIT = 0.8


# TECHNOLOGY DATA

# Heat pump (HP)
COP_TABLE = {
    'air-water': pd.read_csv(PATH_TO_WD + '/data/resources/HP_Specs/vitocal_250A_specs.csv',
                             sep='\t', decimal=',', index_col=[0,1]),
    'air-air': pd.read_csv(PATH_TO_WD + '/data/resources/HP_Specs/bosch_ClimateClass8000i.csv',
                           sep=';', decimal=',', index_col=[0,1]),
}

# Solar thermal
# according to Quaschning
class STCollectorType(str, Enum):
    ABSORBER = 'absorber'
    FLATPLATE_SINGLECOVER_NONSEL = 'flatplate_singlecover_nonsel'
    FLATPLATE_SINGLECOVER_SEL = 'flatplate_singlecover_sel'
    FLATPLATE_DOUBLECOVER_SEL = 'flatplate_singlecover_nonsel'
    VACUUMTUBE = 'vacuumtube'


STCollectorSpecs = {
    STCollectorType.ABSORBER: {
        'eta0': 0.91,
        'a1': 12.0,
        'a2': 0,
    },
    STCollectorType.FLATPLATE_SINGLECOVER_NONSEL: {
        'eta0': 0.86,
        'a1': 6.1,
        'a2': 0.025,
    },
    STCollectorType.FLATPLATE_SINGLECOVER_SEL: {
        'eta0': 0.81,
        'a1': 3.8,
        'a2': 0.009,
    },
    STCollectorType.FLATPLATE_DOUBLECOVER_SEL: {
        'eta0': 0.73,
        'a1': 1.7,
        'a2': 0.016,
    },
    STCollectorType.VACUUMTUBE: {
        'eta0': 0.80,
        'a1': 1.1,
        'a2': 0.008,
    },
}




