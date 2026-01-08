import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.data_preparation.demand_data import (
    get_elec_co2_intensity, get_var_elec_prices, get_dhwcalc_series,
    get_synpro_el_hh_profiles, calc_emob_demands,
    CO2ElecCols, ElPriceCols
)
from src.data_preparation.data_formats import Timeseries
from src import const


class TestDemandData:

    @patch('pandas.read_csv')
    def test_get_elec_co2_intensity_default(self, mock_read_csv):
        # Force FileNotFoundError to test default values
        mock_read_csv.side_effect = FileNotFoundError
        ts = get_elec_co2_intensity(year=2023)
        assert isinstance(ts, Timeseries)
        assert ts[CO2ElecCols.Co2eq_lca].iloc[0] == 395.1 / 1000

    @patch('pandas.read_csv')
    def test_get_var_elec_prices(self, mock_read_csv):
        # Mock energy charts data
        mock_df = pd.DataFrame({
            'Preis (EUR/MWh, EUR/tCO2)': [50.0] * 35040
        })
        # Mocking the MultiIndex/Header structure if necessary, or just simple df
        mock_read_csv.return_value = mock_df
        
        # Test without variable grid fees
        ts = get_var_elec_prices(grid_fees_st=0.1, other_taxes_and_charges=0.05, vat=0.19)
        assert ElPriceCols.Wholesale in ts.data_columns
        # 50 EUR/MWh * 0.001 = 0.05 EUR/kWh
        assert ts[ElPriceCols.Wholesale].iloc[0] == pytest.approx(0.05)

    @patch('pandas.read_csv')
    def test_get_dhwcalc_series(self, mock_read_csv):
        # 1 liter per step (assume hourly dummy)
        mock_read_csv.return_value = pd.DataFrame([1.0] * 8760)
        
        # Energy = mass * cp * dT. 
        # const.TH_CAP_WATER_Wh_per_kg_K is approx 1.16
        temp_dhw = 60
        expected_val = const.TH_CAP_WATER_Wh_per_kg_K / 1000 * (temp_dhw - const.TEMP_EARTH)
        
        ts = get_dhwcalc_series(nr_of_households=1, l_per_person_day=30, temp_dhw=temp_dhw)
        assert ts['DHW'].iloc[0] == pytest.approx(expected_val)

    @patch('os.listdir')
    @patch('pandas.read_csv')
    def test_get_synpro_el_hh_profiles(self, mock_read_csv, mock_listdir):
        mock_listdir.return_value = ['profile1.csv']
        mock_df = pd.DataFrame({'P_el': np.ones(35040)}) # 15 min for a year
        mock_read_csv.return_value = mock_df
        
        ts = get_synpro_el_hh_profiles(hourly=True)
        assert 'profile1' in ts.data_columns
        assert ts.no_of_timesteps() == 8760

    def test_calc_emob_demands_flat(self):
        ev_data = {
            'ev1': {
                'max_p': 11,
                'period': '18-22', # 4 hours
                'drivedist': 100,
                'kwhper100km': 20,
                'daysinweek': "1,2,3,4,5,6,7",
                'chargetype': 'flat'
            }
        }
        # Energy = 20 kWh. Distributed over 4 hours = 5 kW load.
        stat, flex = calc_emob_demands(ev_data)
        # Check a specific hour in the charging period (e.g., hour 19)
        # Filter for hour 19
        load_at_19 = stat.filter(hour_periods=[(19, 20)])['ev1_load']
        assert (load_at_19 == 5.0).all()

    def test_calc_emob_demands_optimized(self):
        ev_data = {
            'ev1': {
                'max_p': 11,
                'period': '18-22',
                'drivedist': 100,
                'kwhper100km': 20,
                'daysinweek': "1,2,4",
                'chargetype': 'optimized'
            }
        }
        stat, flex = calc_emob_demands(ev_data)
        assert 'flex_ev1_delivered_energy' in flex.data_columns
        # Energy should be at the last index of the block
        total_energy = flex['flex_ev1_delivered_energy'].sum()
        # 100 km * 20 kWh/100km = 20 kWh. Sum over 365 days = 7300 kWh.
        assert total_energy == pytest.approx(3120)
