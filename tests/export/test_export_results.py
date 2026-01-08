import pandas as pd
from unittest.mock import patch, MagicMock

from src.data_preparation.pvgis_climate_data import PvgisDataCols
from src.data_preparation.demand_data import CO2ElecCols
from src.export.export_results import (
    write_scenario_results, write_results_summary, write_climate_data_xlsx_analysis
)
from src.data_preparation.data_formats import ScenarioResults
from src.helper import create_ts_multiindex

class TestExportResults:

    @patch('src.export.export_results.save_xlsx_wb')
    @patch('src.export.export_results.ScenarioResults')
    @patch('src.export.export_results.dump_pickle')
    def test_write_scenario_results(self, mock_pickle, mock_results_cls, mock_save_xlsx):
        # setup mock for ScenarioResults
        mock_sce_res = MagicMock(spec=ScenarioResults)
        mock_sce_res.to_dict.return_value = {'ts_el_sources': pd.DataFrame()}
        mock_sce_res.get_ti_sum.return_value = 1.0
        mock_sce_res.find_vals.return_value = pd.DataFrame(1.0, index=create_ts_multiindex(), columns=['value'])
        mock_sce_res.get_en_balance.return_value = 10

        def mock_get_ts(arg1, arg2, arg3, season=None):
            full_ts = pd.DataFrame(1.0, index=create_ts_multiindex(), columns=['val1', 'val2', 'val3'])
            if season is None:
                return full_ts
            return full_ts.loc[season]
        mock_sce_res.get_ts.side_effect = mock_get_ts
        mock_sce_res.ts_others = pd.DataFrame(1.0, index=create_ts_multiindex(), columns=['pmax_t'])
        mock_sce_res.ts_inputs = pd.DataFrame(1.0, index=create_ts_multiindex(),
                                              columns=[PvgisDataCols.Air_temp, CO2ElecCols.Co2eq_lca])

        mock_results_cls.return_value = mock_sce_res

        mock_sce_data = MagicMock()
        mock_sce_data.scenario_name = "TestScenario"
        
        m = MagicMock() # Pyomo model
        ts_inputs = pd.DataFrame()
        results_folder = "results/test"

        res = write_scenario_results(mock_sce_data, m, ts_inputs, results_folder)

        assert res == mock_sce_res
        assert mock_save_xlsx.called
        # Check if the filename contains the scenario name
        args, _ = mock_save_xlsx.call_args
        assert "TestScenario" in args[0]

    @patch('src.export.export_results.save_xlsx_wb')
    def test_write_results_summary(self, mock_save_xlsx):
        # Mock results dictionary
        mock_res1 = MagicMock()
        mock_res1.ti_components = pd.DataFrame({'val': [1, 2]}, index=['comp1', 'sum'])
        mock_res1.yearly_sums = pd.DataFrame({'sum': [10, 20]})
        mock_res1.key_facts = pd.Series({'COP': 2.0, 'self-sufficiency': 0.5})
        
        results = {'Sce1': mock_res1}
        results_folder = "results/summary"

        write_results_summary(results, results_folder)

        assert mock_save_xlsx.called
        args, _ = mock_save_xlsx.call_args

    @patch('src.export.export_results.save_xlsx_wb')
    def test_write_climate_data_xlsx_analysis(self, mock_save_xlsx):
        climate_data = {
            'years': pd.DataFrame({'temp': [10]}),
            'months': pd.DataFrame({'temp': [10]})
        }
        results_folder = "results/climate"

        write_climate_data_xlsx_analysis(climate_data, results_folder)

        assert mock_save_xlsx.called
        args, _ = mock_save_xlsx.call_args
        assert "ClimateDataAnalysis" in args[0]
