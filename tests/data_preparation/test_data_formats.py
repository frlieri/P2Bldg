import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data_preparation.data_formats import Timeseries, SheetData, BuildingModelInput, prepare_dummy_ts
from src.const import (TimeseriesCols, TSStepSize, Season)


class TestTimeseries:
    @pytest.fixture
    def sample_df(self):
        # Create a simple 24-hour dataframe for testing
        data = {'col1': np.random.rand(8760), 'col2': np.random.rand(8760)}
        return pd.DataFrame(data)

    def test_init_empty(self):
        ts = Timeseries(start_year=2023)
        assert ts.freq == TSStepSize.HOURLY
        assert ts.no_of_timesteps() == 8760  # Default non-leap year hourly

    def test_init_with_df(self, sample_df):
        ts = Timeseries(df=sample_df, start_year=2023, end_year=2023)
        # Timeseries init populates full year if not matched? 
        # Actually, the implementation forces index match: self.df.index = self[TimeseriesCols.Step]
        # And it initializes index cols for the full range of years.
        assert ts.no_of_timesteps() == 8760 
        assert 'col1' in ts.data_columns
        assert ts['col1'].iloc[0] == sample_df['col1'].iloc[0]

    def test_set_ts_per_day(self):
        ts_h = Timeseries(freq=TSStepSize.HOURLY)
        assert ts_h.ts_per_day == 24
        ts_q = Timeseries(freq=TSStepSize.QUARTER_HOURLY)
        assert ts_q.ts_per_day == 96

    def test_basic_stats(self, sample_df):
        ts = Timeseries(df=sample_df)
        ts['col1'] = 1.0
        assert ts.sum()['col1'] == ts.no_of_timesteps()
        assert ts.max()['col1'] == 1.0
        assert ts.min()['col1'] == 1.0
        assert ts.mean()['col1'] == 1.0

    def test_normalize(self):
        ts = Timeseries()
        ts['data'] = [1.0, 3.0] + [0.0]*8758
        ts.normalize(columns=['data'])
        assert ts['data'].sum() == pytest.approx(1.0)

    def test_filter_season(self):
        ts = Timeseries(start_year=2023)
        # January 1st should be Winter
        winter_data = ts.filter(season=Season.Winter.value)
        assert not winter_data.empty
        assert (winter_data[TimeseriesCols.Season] == Season.Winter.value).all()

    def test_convert_to_daily(self):
        ts = Timeseries(freq=TSStepSize.HOURLY, start_year=2023)
        ts['data'] = 1.0
        ts.convert_to_daily()
        assert ts.freq == TSStepSize.DAILY
        assert ts.no_of_timesteps() == 365
        assert ts['data'].iloc[0] == 1.0  # Mean of 1.0 is 1.0

    def test_daysums(self):
        ts = Timeseries(start_year=2023)
        ts['data'] = 1.0
        sums = ts.daysums('data')
        assert sums.loc[1, 'data'] == 24.0
        assert len(sums) == 365

class TestSheetData:
    def test_init_components(self):
        params = {
            'pv1_p_inst': 10.0,
            'pv1_azimuth': 180,
            'batt1_capacity': 5.0,
            'other_param': 123
        }
        sd = SheetData(**params)
        assert hasattr(sd, 'pv')
        assert 'pv1' in sd.pv
        assert sd.pv['pv1']['p_inst'] == 10.0
        assert sd.other_param == 123
        assert 'batt' in dir(sd)

class TestBuildingModelInput:
    @patch('pandas.read_excel')
    def test_init_and_scenario_parsing(self, mock_read_excel):
        # Create a mock Excel structure
        mock_df = pd.DataFrame(
            {
                ('PARAMETERS', 'datatype'): ['float'],
                ('SCENARIO VALUES', 'Base'): [50.0]
            },
            index=['latitude']
        )
        # We need to mock all expected sheets
        sheets = ['Location', 'Consumption Data', 'Building Data', 
                  'Technical Component Data', 'Economic Data', 'Model Settings', 'HP Specifications']
        
        mock_data = {s: mock_df.copy() for s in sheets}
        # Special case for HP Specifications which is handled differently in _set_hp_specs
        # Define the multi-level columns
        columns = pd.MultiIndex.from_tuples([
            ('INDEX', 'param'),
            ('INDEX', 'flow_temp'),
            ('VALUES', -20),
            ('VALUES', -15),
            ('VALUES', -10),
            ('VALUES', -7),
            ('VALUES', 2),
            ('VALUES', 7),
            ('VALUES', 10),
            ('VALUES', 20),
            ('VALUES', 30),
            ('VALUES', 35)
        ])
        data = [
            ['COP', 25, 2.6, 2.8, 3.2, 3.5, 4.8, 5.8, 5.8, 8.4, 8.7, 8.7],
            ['COP', 35, 2.3, 2.5, 2.8, 3.0, 4.0, 5.2, 5.2, 8.2, 8.7, 8.7],
            ['COP', 45, 2.0, 2.2, 2.4, 2.6, 3.3, 4.1, 4.8, 5.4, 8.5, 8.5]
        ]

        # Create the DataFrame
        hp_df = pd.DataFrame(data, columns=columns, index=['vitocal_250A', 'vitocal_250A', 'vitocal_250A'])
        mock_data['HP Specifications'] = hp_df

        mock_read_excel.return_value = mock_data
        
        with patch('src.const.PATH_TO_WD', ''):
            bmi = BuildingModelInput('dummy.xlsx')
            assert 'Base' in bmi.scenarios
            assert 'Base' in bmi.scenario_data
            assert bmi.scenario_data['Base'].location.latitude == 50.0

def test_prepare_dummy_ts():
    ts = prepare_dummy_ts(['test_col'], value=5.0, freq=TSStepSize.HOURLY)
    assert isinstance(ts, Timeseries)
    assert (ts['test_col'] == 5.0).all()
    assert ts.freq == TSStepSize.HOURLY
