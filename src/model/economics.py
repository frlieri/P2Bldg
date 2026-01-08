"""
src/model/economics.py

Financial utilities for the optimisation framework. This module implements
common discounted‑cashflow and levelised cost calculations used when converting
capital and operational costs into annualised or present‑value figures for
scenario analysis.

Main functions:

- crf(lifetime, i_eff=const.REAL_INTEREST_RATE) -> float
    Compute the capital recovery factor (CRF). Handles i_eff == 0 as a special
    case; asserts that lifetime is non‑negative and raises for negative rates.

- calc_annuity(inv_cost, duration=None, i_eff=None) -> float
    Annualise an investment cost using the CRF over the given duration.

- calc_present_value(annuity, duration=None, i_eff=None) -> float
    Convert an annuity to its present value using the inverse CRF.

- celf(escalation_rate, duration=None, i_eff=None) -> float
    Compute the constant escalation levelization factor (CELF) for recurring
    variable costs subject to an escalation rate.

- calc_constant_escalation_row(var_cost, escalation_rate, duration=None, i_eff=None) -> float
    Apply CELF to a variable cost row to obtain a levelised annual figure.

- calc_residual_value(inv_cost, lifetime, duration=None, i_eff=None) -> float
    Estimate residual book value when an asset outlives or is retired before
    the analysis duration.

- calc_reinvest(inv_cost, lifetime, escalation_rate=0.0, i_eff=None) -> float
    Compute the escalated cost of a reinvestment occurring after a given lifetime.

- calc_inv_cost_with_residuals_and_reinvests(inv_cost, lifetime, escalation_rate=0.0,
    duration=None, i_eff=None) -> float
    Aggregate initial investment, necessary reinvestments and residual values
    across the analysis duration to obtain an effective total investment.

- inflation_adjustment(annuity, i_nom, duration=None, i_eff=None, inflation_rate=None) -> float
    Adjust an annuity value to account for inflation using nominal and real
    interest conventions.

Behavior and notes:

- Functions use project constants from `src.const` (e.g. DURATION, REAL_INTEREST_RATE,
  INFLATION_RATE) when optional arguments are not supplied.
- Inputs and outputs are floats representing yearly amounts or factors; durations
  are interpreted in years.
- Functions are pure (no file I/O or network access) and raise ValueError for
  invalid financial inputs (negative interest rates, inconsistent parameters).
- Numerical edge cases (e.g. zero interest) are handled explicitly to avoid
  division-by-zero errors.

Dependencies:

- math
- `src.const` (for default economic parameters)

Example:

    ann = calc_annuity(10000.0, duration=20)
    pv = calc_present_value(ann, duration=20)
"""

import math
from src import const


def crf(lifetime: int, i_eff=const.REAL_INTEREST_RATE) -> float:
    """calculate the capital recovery factor (CRF)"""

    assert lifetime >= 0, "lifetime must be > 0 years"

    if i_eff > 0:
        return (i_eff * (1+i_eff) ** lifetime) / ((1 + i_eff) ** lifetime - 1)
    elif i_eff == 0:
        return 1/lifetime
    else:
        raise ValueError("i_eff must be >= 0")


def calc_annuity(inv_cost: float, duration=None, i_eff=None) -> float:
    """calculate annuities using the capital recovery factor (CRF)"""
    if duration is None:
        duration = const.DURATION
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE

    return inv_cost * crf(duration, i_eff)


def calc_present_value(annuity: float, duration=None, i_eff=None) -> float:
    """calculate total investment"""
    if duration is None:
        duration = const.DURATION
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE
    pvf = 1/crf(duration, i_eff)
    return annuity * pvf


def celf(escalation_rate: float, duration=None, i_eff=None) -> float:
    """calculate constant escalation levelization factor (CELF)"""
    if duration is None:
        duration = const.DURATION
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE

    f = (1+escalation_rate)/(1+i_eff)

    if f == 1.0:
        return 1.0
    else:
        return f*(1-f**duration)/(1-f)*crf(duration, i_eff)


def calc_constant_escalation_row(var_cost: float, escalation_rate: float, duration=None, i_eff=None) -> float:
    """calculate leveled costs for variable costs row using the CELF"""
    if duration is None:
        duration = const.DURATION
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE

    return var_cost*celf(escalation_rate, duration, i_eff)


def calc_residual_value(inv_cost: float, lifetime: int, duration=None, i_eff=None) -> float:
    """calculate the residual lifetime if lifetime > duration"""
    if duration is None:
        duration = const.DURATION
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE

    return inv_cost * (lifetime - duration)/(lifetime*(1+i_eff)**duration)


def calc_reinvest(inv_cost: float, lifetime: int, escalation_rate=0.0, i_eff=None) -> float:
    """calculate reinvest needed if duration > lifetime """
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE

    return inv_cost * ((1 + escalation_rate)/(1 + i_eff))**lifetime


def calc_inv_cost_with_residuals_and_reinvests(inv_cost: float, lifetime: int, escalation_rate=0.0,
                                               duration=None, i_eff=None) -> float:
    """
    Calculates the total investment cost considering reinvestments and residual values over
    a given duration, while accounting for escalation rates and effective interest rates.
    This function is particularly useful for cost analysis in long-term projects where
    investments are required periodically or partially recovered before the end of the project.

    :param inv_cost: Initial investment cost.
    :param lifetime: Lifetime of the investment in years.
    :param escalation_rate: Annual escalation rate of the investment cost. Defaults to 0.0.
    :param duration: Total duration in years for which the investment is being considered.
        Defaults to a predefined constant value.
    :param i_eff: Effective interest rate. Defaults to a predefined constant value.
    :return: Final adjusted investment cost considering reinvestment, residual value,
        or both as appropriate.
    """
    if duration is None:
        duration = const.DURATION
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE

    if duration > lifetime:
        nr_of_investments = math.ceil(duration/lifetime)
        reinv_cost = [calc_reinvest(inv_cost, lifetime*i, escalation_rate, i_eff) for i in range(1, nr_of_investments)]
        if (lifetime*nr_of_investments - duration) > 0:
            residual_value = calc_residual_value(
                reinv_cost[-1], lifetime, duration - (nr_of_investments -1) * lifetime, i_eff)
        elif duration == lifetime*nr_of_investments:
            residual_value = 0
        else:
            raise ValueError
        inv_cost = inv_cost + sum(reinv_cost) - residual_value
    elif duration < lifetime:
        inv_cost = inv_cost - calc_residual_value(inv_cost, lifetime, duration, i_eff)
    else:
        inv_cost = inv_cost

    return inv_cost


def inflation_adjustment(annuity: float, i_nom: float, duration=None, i_eff=None, inflation_rate: float=None) -> float:
    """
    Calculates the adjusted annuity to account for the effects of inflation using the nominal
    interest rate, effective interest rate, duration of annuity, and inflation rate.

    This function leverages cash flow ratios (CRF) to compute the adjustment based on given
    parameters or default constants when values are not provided. The inflation adjustment is
    considered to help accurately determine the real value of the annuity over its duration.

    :param annuity: Initial annuity value to be inflation-adjusted.
    :type annuity: float
    :param i_nom: Nominal interest rate.
    :type i_nom: float
    :param duration: Annuity duration in years. Optional; uses a default value if not provided.
    :type duration: int, optional
    :param i_eff: Effective annual interest rate. Optional; uses a default value if not provided.
    :type i_eff: float, optional
    :param inflation_rate: Annual inflation rate. Optional; uses a default value if not provided.
    :type inflation_rate: float, optional
    :return: Inflation-adjusted annuity value.
    :rtype: float
    """
    if duration is None:
        duration = const.DURATION
    if i_eff is None:
        i_eff = const.REAL_INTEREST_RATE
    if inflation_rate is None:
        inflation_rate = const.INFLATION_RATE

    return (annuity * crf(lifetime=duration, i_eff=i_eff)/
            (crf(lifetime=duration, i_eff=i_nom)*(1+inflation_rate)**duration))



