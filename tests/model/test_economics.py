import pytest
import pandas as pd
from src.model.economics import (
    calc_annuity, 
    calc_inv_cost_with_residuals_and_reinvests, 
    calc_constant_escalation_row
)

class TestEconomics:

    def test_calc_annuity(self):
        """
        Tests the basic annuity calculation.
        Example: 1000 Euro, 10 years, 5% interest rate.
        Formula: A = K * (i * (1+i)^n) / ((1+i)^n - 1)
        """
        k = 1000
        n = 10
        i = 0.05
        expected = k * (i * (1 + i)**n) / ((1 + i)**n - 1)
        
        result = calc_annuity(k, n, i)
        assert result == pytest.approx(expected)

    def test_calc_annuity_zero_interest(self):
        """Tests annuity with 0% interest (should be linear division)."""
        result = calc_annuity(1000, 10, 0.0)
        assert result == 100.0

    def test_calc_inv_cost_with_residuals_and_reinvests(self):
        """
        Tests calculation of total present value including reinvestments 
        and residual values at the end of the observation period.
        """
        inv = 1000
        lifetime = 10
        duration = 15 # Observation period longer than lifetime
        i_eff = 0.05
        
        # This should trigger one reinvestment at year 10
        # and a residual value for the second unit at year 15
        result = calc_inv_cost_with_residuals_and_reinvests(
            inv, lifetime, duration=duration, i_eff=i_eff
        )
        
        # PV = 1000 (now) + 1000 / (1.05^10) (reinvest) - residual
        reinvest_pv = inv / (1 + i_eff)**10
        # Residual: unit has 5 years left out of 10. Value at year 15 is 500.
        # Residual PV = (500) / (1.05^15)
        residual_pv = (inv * 5 / 10) / (1 + i_eff)**15
        expected = inv + reinvest_pv - residual_pv
        
        assert result == pytest.approx(expected)

    def test_calc_constant_escalation_row(self):
        """
        Tests the calculation of levelled costs for a variable that escalates yearly.
        """
        val_start = 0.10 # e.g. 10 cents/kWh
        esc_rate = 0.02  # 2% price increase p.a.
        duration = 20
        i_eff = 0.04
        
        result = calc_constant_escalation_row(val_start, esc_rate, duration, i_eff)
        
        # If escalation rate < interest rate, there's a specific formula
        # result should be between val_start and val_start * (1.02^20)
        assert val_start < result < val_start * (1 + esc_rate)**duration

    def test_calc_constant_escalation_series(self):
        """Tests levelization when the input is a pandas Series (e.g. hourly prices)."""
        val_start = pd.Series([0.1, 0.2, 0.3])
        esc_rate = 0.0
        duration = 10
        i_eff = 0.05
        
        result = calc_constant_escalation_row(val_start, esc_rate, duration, i_eff)
        
        assert isinstance(result, pd.Series)
        # With 0% escalation and 5% interest, levelled value is exactly the input 
        # (assuming constant annuity calculation logic inside)
        # Actually, levelled costs are often defined such that the NPV is preserved.
        assert result.iloc[0] == pytest.approx(0.1)
