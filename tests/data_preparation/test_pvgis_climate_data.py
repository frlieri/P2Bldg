import pandas as pd
from unittest.mock import patch
from src.data_preparation.pvgis_climate_data import (
    get_pvgis_hourly_data, get_pvgis_tmy_data, PvgisDataCols
)
from src.data_preparation.data_formats import Timeseries


class TestPvgisClimateData:

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    @patch('src.data_preparation.data_formats.Timeseries.plot_typeday')
    def test_get_pvgis_hourly_data_cached(self, mock_plot, mock_read_csv, mock_exists):
        # Setup: file exists in cache
        mock_exists.return_value = True
        
        # Create a mock dataframe with required columns
        cols = [e.value for e in PvgisDataCols]
        mock_df = pd.DataFrame(100.0, index=range(8760), columns=cols)
        mock_read_csv.return_value = mock_df

        ts = get_pvgis_hourly_data(latitude=50, longitude=10, start_year=2023, end_year=2023)
        
        assert isinstance(ts, Timeseries)
        # PV power P is divided by 1000 in the function
        assert ts[PvgisDataCols.P_el_t].iloc[0] == 0.1
        assert mock_read_csv.called

    @patch('pvlib.iotools.pvgis.get_pvgis_hourly')
    @patch('os.path.exists')
    @patch('pandas.DataFrame.to_csv')
    @patch('src.data_preparation.data_formats.Timeseries.plot_typeday')
    def test_get_pvgis_hourly_data_api(self, mock_plot, mock_to_csv, mock_exists, mock_get_pvgis):
        # Setup: file does NOT exist in cache
        mock_exists.return_value = False
        
        cols = [e.value for e in PvgisDataCols]
        mock_df = pd.DataFrame(1000.0, index=pd.date_range('2023-01-01', periods=8760, freq='h'), columns=cols)
        mock_get_pvgis.return_value = (mock_df, "")

        ts = get_pvgis_hourly_data(latitude=50, longitude=10, start_year=2023, end_year=2023)
        
        assert ts[PvgisDataCols.P_el_t].iloc[0] == 1.0
        assert mock_get_pvgis.called
        assert mock_to_csv.called

    @patch('pvlib.iotools.pvgis.get_pvgis_tmy')
    @patch('os.path.exists')
    @patch('pandas.DataFrame.to_csv')
    def test_get_pvgis_tmy_data_api(self, mock_to_csv, mock_exists, mock_get_tmy):
        mock_exists.return_value = False
        
        # TMY data usually has specific columns like 'T2m', 'G(h)', etc.
        mock_df = pd.DataFrame({'T2m': [15.0]*8760, 'G(h)': [200.0]*8760})
        mock_get_tmy.return_value = (mock_df, None, None, None)

        ts = get_pvgis_tmy_data(latitude=50, longitude=10)
        
        assert isinstance(ts, Timeseries)
        assert ts['T2m'].iloc[0] == 15.0
        assert mock_get_tmy.called
