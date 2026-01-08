import pytest
import pandas as pd
from datetime import datetime
from src.helper import (
    get_season, get_typeday, normalize_wrt_sum, hour_within_period,
    calc_hours_in_period, identify_blocks, calc_for_blocks_of_n,
    create_ts_multiindex, tuple_list_from_hour_periods, Season
)
from src import const


class TestHelper:

    def test_get_season(self):
        # Winter: Dec, Jan, Feb
        assert get_season(datetime(2023, 1, 15)) == Season.Winter.value
        # Summer: Jun, Jul, Aug
        assert get_season(datetime(2023, 7, 15)) == Season.Summer.value
        # Spring: Mar, Apr, May
        assert get_season(datetime(2023, 4, 15)) == Season.Transition.value
        # Autumn: Sep, Oct, Nov
        assert get_season(datetime(2023, 10, 15)) == Season.Transition.value

    def test_get_typeday(self):
        # 2023-01-02 is a Monday (Workday)
        assert get_typeday(datetime(2023, 1, 2)) == const.Typeday.Workday.value
        # 2023-01-01 is a Sunday (Sunday)
        assert get_typeday(datetime(2023, 1, 1)) == const.Typeday.Sunday.value
        # 2023-01-07 is a Saturday (Saturday)
        assert get_typeday(datetime(2023, 1, 7)) == const.Typeday.Saturday.value

    def test_normalize_wrt_sum(self):
        data = pd.Series([10, 20, 30, 40])
        normalized = normalize_wrt_sum(data)
        assert normalized.sum() == pytest.approx(1.0)
        assert normalized.iloc[0] == 0.1

    def test_hour_within_period(self):
        period = (18, 7) # Evening to morning
        assert hour_within_period(period, 20) is True
        assert hour_within_period(period, 3) is True
        assert hour_within_period(period, 12) is False
        
        period_normal = (8, 17)
        assert hour_within_period(period_normal, 10) is True
        assert hour_within_period(period_normal, 20) is False

    def test_calc_hours_in_period(self):
        assert calc_hours_in_period((18, 7)) == 13
        assert calc_hours_in_period((8, 17)) == 9
        assert calc_hours_in_period((10, 10)) == 0

    def test_identify_blocks(self):
        # 1 represent "active" status
        data = pd.Series([0, 1, 1, 0, 0, 1, 0])
        blocks = identify_blocks(data)
        # Expected: {1: [1, 2], 2: [5]}
        assert len(blocks) == 2
        assert blocks[1] == [1, 2]
        assert blocks[2] == [5]

    def test_calc_for_blocks_of_n(self):
        # Default n=24 (daily)
        data = pd.Series([1.0] * 24 + [2.0] * 24)
        result = calc_for_blocks_of_n(data, block_size=24, func='mean')
        assert result.iloc[0] == 1.0
        assert result.iloc[24] == 2.0
        assert len(result) == 48

    def test_create_ts_multiindex(self):
        # Mock constants for a small range
        with pytest.MonkeyPatch.context() as m:
            m.setattr(const, 'APPLY_TIMESERIES_REDUCTION', False)
            m.setattr(const, 'TS_PER_HOUR', 1)
            idx = create_ts_multiindex()
            # Non-leap year hourly: 8760
            assert len(idx) == 8760
            assert isinstance(idx, pd.MultiIndex)

    def test_tuple_list_from_hour_periods(self):
        # Input format from Excel is often "18-7, 8-12"
        input_str = "18-7, 8-12"
        result = tuple_list_from_hour_periods(input_str)
        assert result == [(18, 7), (8, 12)]
        
        # Handle nan
        assert tuple_list_from_hour_periods("nan") == []
