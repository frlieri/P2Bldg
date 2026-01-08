import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

import src.helper
from src.data_preparation.timeseries_reduction import climate_data_analysis, prepare_ts_profiles
from src.data_preparation.data_formats import Timeseries
from src.const import Season, InputData
from src.data_preparation.pvgis_climate_data import PvgisDataCols
from src.data_preparation.demand_data import ElLoadCols, ElPriceCols, DHWCols, CO2ElecCols
from src import const


class TestTimeseriesReduction:

    @pytest.fixture
    def mock_hourly_pvgis_ts(self):
        # Create a mock Timeseries object for a full year (8760 hours)
        df = pd.DataFrame({
            PvgisDataCols.Air_temp.value: np.random.uniform(-10, 30, 8760),
            PvgisDataCols.P_el_t.value: np.random.uniform(0.01, 1, 8760)
        })
        return Timeseries(df, start_year=2023, end_year=2023)

    @patch('src.data_preparation.timeseries_reduction.get_pvgis_hourly_data')
    @patch('src.data_preparation.timeseries_reduction.load_pickle')
    @patch('src.data_preparation.timeseries_reduction.dump_pickle')
    def test_climate_data_analysis(self, mock_dump, mock_load, mock_get_pvgis, mock_hourly_pvgis_ts):
        # Force FileNotFoundError to trigger calculation
        mock_load.side_effect = FileNotFoundError
        mock_get_pvgis.return_value = mock_hourly_pvgis_ts
        
        # Mock constants for a single year range
        with patch.object(const, 'CLIMATE_DATA_YEARS', [2023, 2023]):
            result = climate_data_analysis(write_xlsx=False)
            
            assert 'years' in result
            assert 'weeks' in result
            assert 'seasons' in result
            assert mock_dump.called

    def test_prepare_ts_profiles_no_reduction(self):
        # Setup inputs
        t_len = 8760
        ts_inputs = {
            InputData.PvgisData: Timeseries(pd.DataFrame({PvgisDataCols.P_el_t.value: [0.5]*t_len, 
                                                          PvgisDataCols.Air_temp.value: [15.0]*t_len})),
            InputData.ElPrice: Timeseries(pd.DataFrame({ElPriceCols.Wholesale.value: [0.1]*t_len})),
            InputData.ElDemand: Timeseries(pd.DataFrame({'load1': [1.0]*t_len})),
            InputData.StatEmobDemand: Timeseries(pd.DataFrame({ElLoadCols.TOTAL_EV_STAT.value: [0.0]*t_len})),
            InputData.FlexEmobDemand: Timeseries(pd.DataFrame({'flex_load': [0.0]*t_len})),
            InputData.DhwDdemand: Timeseries(pd.DataFrame({DHWCols.Load.value: [0.2]*t_len})),
            InputData.CO2perKwhEl: Timeseries(pd.DataFrame({CO2ElecCols.Co2eq_lca.value: [0.4]*t_len})),
        }

        ts_red_inputs, sel_weeks = prepare_ts_profiles(ts_inputs, apply_ts_reduction=False)
        
        assert ts_red_inputs.shape[0] == t_len
        assert 'cost_factors' in ts_red_inputs.columns
        assert (ts_red_inputs['cost_factors'] == 1.0).all()

    @patch('src.data_preparation.timeseries_reduction.dump_pickle')
    @patch('src.data_preparation.timeseries_reduction.create_ts_multiindex')
    def test_prepare_ts_profiles_with_reduction(self, mock_multiindex, mock_dump):
        # Mocking reduction logic is complex, so we provide pre-selected weeks
        t_idx = src.helper.create_ts_multiindex()
        mock_multiindex.return_value = t_idx
        
        # 168 hours * 52 weeks = 8736 (close enough for mock year)
        dummy_df = pd.DataFrame(1.0, index=range(8760), columns=['P', 'temp_air'])
        ts_pvgis = Timeseries(dummy_df)
        
        ts_inputs = {
            InputData.PvgisData: ts_pvgis,
            InputData.ElPrice: Timeseries(pd.DataFrame({ElPriceCols.Wholesale.value: [0.1]*8760})),
            InputData.ElDemand: Timeseries(pd.DataFrame({'load': [1.0]*8760})),
            InputData.StatEmobDemand: Timeseries(pd.DataFrame({ElLoadCols.TOTAL_EV_STAT.value: [0.0]*8760})),
            InputData.FlexEmobDemand: Timeseries(pd.DataFrame({'flex': [0.0]*8760})),
            InputData.DhwDdemand: Timeseries(pd.DataFrame({DHWCols.Load.value: [0.2]*8760})),
            InputData.CO2perKwhEl: Timeseries(pd.DataFrame({CO2ElecCols.Co2eq_lca.value: [0.4]*8760})),
        }
        
        sel_weeks = pd.DataFrame({
            'nr': {Season.Winter: 1, Season.Transition: 14, Season.Summer: 27,},
            'cost_factor': {Season.Winter: 20, Season.Transition: 15, Season.Summer: 17}
        })

        ts_red_inputs, result_sel_weeks = prepare_ts_profiles(
            ts_inputs, sel_weeks=sel_weeks, apply_ts_reduction=True
        )
        
        # Reduced size should be 4 seasons * 168 hours = 672
        assert ts_red_inputs.shape[0] == 3*168
        assert 'cost_factors' in ts_red_inputs.columns
        assert ts_red_inputs.loc[Season.Winter.value, 'cost_factors'].iloc[0] == 20
