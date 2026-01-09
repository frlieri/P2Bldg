"""
src/model/components.py

Core component and device model classes for the energy system optimisation framework.

Purpose:

- Define an extensible object model for system components (sources, sinks, nodes, links)
  and common technologies (PV, solar thermal, heat pumps, storages, heaters, imports/exports).
- Provide initialization, validation and derived-value calculations used by the optimisation model.

Main responsibilities:

- Component base: encapsulate costs, lifetimes, emissions and automatic levelled-cost calculation.
- Temperature levels: represent static or time-varying temperatures and heating-curve logic.
- Solar potential & technologies: prepare PV/solar time series (normalised generation, irradiance)
  and wire potentials into PV and solar-thermal technology classes.
- Thermal and electrical nodes: group sources and sinks and enforce type consistency.
- Storage technologies: model electrical and thermal storages, capacities, efficiencies and degradation.
- Device-specific calculations: heating-curve parameter solving, COP/Qmax interpolation, thermal capacity
  computations and simple solar thermal performance estimation.
- Utility components: demand, flexible demand, heat gains, refurbishment and import/export wrappers.

Behavior and side effects:

- Many classes compute and expose time-series attributes (pandas Series/DataFrame) and summary values.
- SolarPotential may fetch and cache PVGIS data and call reduction utilities; some constructors call
  pickle helpers for caching (file I/O and network access possible).
- Classes are primarily data containers with derived-value helpers; they do not perform optimisation
  themselves but prepare inputs consumed by the optimisation model.

Dependencies:

- pandas, numpy
- project modules: `src.const`, `src.data_preparation.pvgis_climate_data`,
  `src.data_preparation.timeseries_reduction`, `src.model.economics`, `src.helper`

Intended use:

- Instantiate and configure system components for scenario setup, then pass configured objects into
  the optimisation pipeline which constructs models, constraints and objectives from these instances.
"""

import pandas as pd
import numpy as np

import src.const as const
from src.const import InputData
from src.data_preparation.pvgis_climate_data import get_pvgis_hourly_data, PvgisDataCols
from src.data_preparation.timeseries_reduction import prepare_ts_profiles
from src.model.economics import calc_inv_cost_with_residuals_and_reinvests, calc_annuity, \
    calc_constant_escalation_row
from src.helper import calc_for_blocks_of_n, load_pickle, dump_pickle, create_ts_data_dump_id


class Component:
    """
    Base class for all energy system components.

    Encapsulates common attributes for capacity, operational bounds, costs, lifetime and emissions.
    Performs levelled-cost calculations on initialization and assigns any additional keyword
    parameters as instance attributes.

    :param name: Component identifier.
    :param p_max: Maximum power capacity.
    :param p_max_rel_t: Relative maximum power over time (pd.Series or float).
    :param p_min: Minimum power capacity.
    :param p_min_rel_t: Relative minimum power over time (pd.Series or float).
    :param t_full_min: Minimum full-load hours.
    :param t_full_max: Maximum full-load hours.
    :param is_built: Boolean indicating whether the component is already installed.
    :param inv_cost: Investment cost (also applied at reinvestment).
    :param inv_cost_escalation_rate: Annual escalation rate for investment cost.
    :param fix_cost_inv: Fixed investment cost (also at reinvestment).
    :param fix_cost_inv_escalation_rate: Annual escalation rate for fixed investment cost.
    :param fix_cost_one_time: One-time fixed cost (incurred if built).
    :param fix_cost_one_time_escalation_rate: Annual escalation rate for one-time cost.
    :param fix_cost_yearly: Yearly fixed cost.
    :param fix_cost_yearly_escalation_rate: Annual escalation rate for yearly fixed cost.
    :param var_cost_t: Variable operational cost (pd.Series or float).
    :param var_cost_t_escalation_rate: Annual escalation rate for variable cost.
    :param lifetime: Operational lifetime in years.
    :param co2eq_per_cap: CO₂-equivalent per unit of capacity.
    :param co2eq_per_kwh_t: CO₂-equivalent per kWh over time (pd.Series or float).
    :param params: Additional dynamic attributes assigned to the instance.

    :ivar name: Component name.
    :ivar p_max: Maximum power capacity.
    :ivar p_min: Minimum power capacity.
    :ivar t_full_min: Minimum full-load hours.
    :ivar t_full_max: Maximum full-load hours.
    :ivar is_built: Installation state.
    :ivar inv_cost_levelled: Levelled investment cost per year.
    :ivar fix_cost_levelled: Levelled fixed yearly cost.
    :ivar fix_cost_one_time_levelled: Levelled one-time cost.
    :ivar var_cost_t_levelled: Levelled variable costs (series or scalar).
    :ivar lifetime: Lifetime in years.
    :ivar co2eq_per_cap: CO₂-equivalent per capacity.
    :ivar co2eq_per_kwh_t: Time series or scalar of CO₂ per kWh.
    """
    def __init__(self, name,
                 p_max=const.MAX_P, p_max_rel_t=1.0, p_min=0.0, p_min_rel_t=0.0,
                 t_full_min=0, t_full_max=8760,
                 is_built=None,
                 inv_cost=0.000000001, inv_cost_escalation_rate=0.0,
                 fix_cost_inv=0.0, fix_cost_inv_escalation_rate=0.0,
                 fix_cost_one_time=0.0, fix_cost_one_time_escalation_rate=0.0,
                 fix_cost_yearly=0.0, fix_cost_yearly_escalation_rate=0.0,
                 var_cost_t=0.000000001, var_cost_t_escalation_rate=0.0,
                 lifetime=20, co2eq_per_cap=0.0, co2eq_per_kwh_t=0.0,
                 **params):

        self.name = name
        self.p_max = p_max
        self.p_max_rel_t: pd.Series | float = p_max_rel_t
        self.p_min = p_min
        self.p_min_rel_t: pd.Series | float = p_min_rel_t
        self.t_full_min: int = t_full_min
        self.t_full_max: int = t_full_max
        self.is_built: bool = is_built
        self.inv_cost = inv_cost  # also paid at every reinvest
        self.inv_cost_escalation_rate = inv_cost_escalation_rate
        self.fix_cost_inv = fix_cost_inv  # also paid at every reinvest
        self.fix_cost_inv_escalation_rate = fix_cost_inv_escalation_rate
        self.fix_cost_one_time = fix_cost_one_time  # paid only once if built
        self.fix_cost_one_time_escalation_rate = fix_cost_one_time_escalation_rate
        self.fix_cost_yearly = fix_cost_yearly
        self.fix_cost_yearly_escalation_rate = fix_cost_yearly_escalation_rate
        self.var_cost_t: pd.Series | float = var_cost_t
        self.var_cost_t_escalation_rate = var_cost_t_escalation_rate
        self.lifetime = lifetime
        self.co2eq_per_cap = co2eq_per_cap
        self.co2eq_per_kwh_t: pd.Series | float = co2eq_per_kwh_t

        self._calc_levelled_costs()

        for key, val in params.items():
            setattr(self, key, val)

    def __repr__(self):
        class_name = str(self.__class__).split('.')[-1][:-2]
        return f"{class_name}: {self.name}"

    def _calc_levelled_costs(self):
        self._inv_cost_for_duration = calc_inv_cost_with_residuals_and_reinvests(
            self.inv_cost, self.lifetime, escalation_rate=self.inv_cost_escalation_rate)
        self.inv_cost_levelled = calc_annuity(self._inv_cost_for_duration,
                                              duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE)
        self._fix_cost_inv_for_duration = calc_inv_cost_with_residuals_and_reinvests(
            self.fix_cost_inv, self.lifetime, escalation_rate=self.fix_cost_inv_escalation_rate,
            duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE)
        self.fix_cost_inv_levelled = calc_annuity(
            self._fix_cost_inv_for_duration, duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE)
        self.fix_cost_levelled = calc_constant_escalation_row(
            self.fix_cost_yearly, escalation_rate=self.fix_cost_yearly_escalation_rate,
            duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE)
        self.fix_cost_one_time_levelled = calc_annuity(self.fix_cost_one_time,
                                                       duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE)
        self.var_cost_t_levelled = calc_constant_escalation_row(
            self.var_cost_t, escalation_rate=self.var_cost_t_escalation_rate,
            duration=const.DURATION, i_eff=const.REAL_INTEREST_RATE)


class TempLevel:
    """
    Represents a temperature level with associated information.

    This class is used to store and manage temperature levels. It holds the
    name of the level, the main temperature value, and optionally a series
    of temperature values. If the optional temperature series is not provided,
    the class will default it to the main temperature value.

    :ivar name: The name of the temperature level.
    :type name: str
    :ivar temp: The main temperature value associated with the level.
    :type temp: Any
    :ivar temp_t: The temperature series associated with the level if provided,
        otherwise defaults to the main temperature value.
    :type temp_t: pd.Series | int
    """
    def __init__(self, name, temp, temp_t: pd.Series=None, **params):
        self.name = name
        self.temp = temp

        if temp_t is not None:
            self.temp_t: pd.Series | int = temp_t
        else:
            self.temp_t: pd.Series | int = self.temp

        super().__init__(**params)


class FlowTempLevel(TempLevel):
    """
    Class responsible for managing and calculating flow temperature levels, including the
    application of a heating curve for dynamic adjustments based on environmental and indoor
    temperature parameters.

    This class supports the configuration of a heating curve that dynamically updates flow
    temperature values based on input data like outdoor temperatures and heating thresholds.
    It also calculates reference and adjustable temperature values for flow and return
    temperatures, considering optional raised storage temperature values.

    :ivar flow_temp_ref: Reference value for the flow temperature.
    :ivar return_temp_ref: Reference value for the return temperature.
    :ivar temp_air_t: Time series data for outdoor air temperature when a heating curve is applied.
    :type temp_air_t: pandas.Series | None
    :ivar flow_temp_min: Minimum flow temperature when a heating curve is applied.
    :ivar outtemp_min_ref: Reference outdoor temperature for minimum heating conditions.
    :ivar indoortemp_ref: Reference indoor temperature for the heating curve.
    :ivar indoortemp_t: Time series data for indoor temperature adjustments. Only used when a
        heating curve is applied.
    :type indoortemp_t: pandas.Series | None
    :ivar heating_threshold: Outdoor temperature threshold for enabling the heating curve.
    :ivar flow_temp_t: Calculated flow temperature over time or as a constant depending on
        configuration.
    :type flow_temp_t: pandas.Series | int
    :ivar return_temp_t: Calculated return temperature over time or as a constant depending
        on configuration.
    :type return_temp_t: pandas.Series | int
    :ivar delta_temp_t: Difference between calculated flow and return temperatures.
    :ivar ret_flow_ratio: Ratio of return temperature to flow temperature when using the
        heating curve.
    :ivar _heating_curve_level: Level parameter of the calculated heating curve.
    :ivar _heating_curve_slope: Slope parameter of the calculated heating curve.
    """
    def __init__(self, name, flow_temp: int, return_temp: int,
                 apply_heating_curve=False, temp_air_t: pd.Series = None, raised_storage_temp: int = None,
                 outtemp_min_ref=-14, indoortemp_ref=const.NOM_TEMP_INDOOR, indoortemp_t: pd.Series = None,
                 heating_threshold=const.HEATING_THRESHOLD, flow_temp_min=35, **params):

        self.flow_temp_ref = flow_temp
        self.return_temp_ref = return_temp

        if apply_heating_curve:
            assert temp_air_t is not None, "temp_air_t needs to be given if apply_heating_curve=True"
            self.temp_air_t = temp_air_t
            self.flow_temp_min = flow_temp_min
            self.outtemp_min_ref = outtemp_min_ref
            self.indoortemp_ref = indoortemp_ref
            self.indoortemp_t: pd.Series = indoortemp_t
            self.heating_threshold = heating_threshold
            self._heating_curve_level, self._heating_curve_slope = self._calc_heating_curve_params()
            self.flow_temp_t: pd.Series | int = self._calc_flowtemp_t()
            self.ret_flow_ratio = self.return_temp_ref/self.flow_temp_ref
            self.return_temp_t = self.ret_flow_ratio * self.flow_temp_t
            if raised_storage_temp is not None:
                self.flow_temp_t = raised_storage_temp
            self.delta_temp_t = self.flow_temp_t - self.return_temp_t
        else:
            self.flow_temp_t: pd.Series | int = self.flow_temp_ref
            self.return_temp_t: pd.Series | int = self.return_temp_ref
            self.delta_temp_t = self.flow_temp_t - self.return_temp_t

        super().__init__(name, temp=self.flow_temp_ref, temp_t=self.flow_temp_t, **params)

    def _calc_heating_curve_params(self):
        # solve param values level and slope for min and max flowtemp

        d_out_in_ref = self.outtemp_min_ref - self.indoortemp_ref
        d_out_in_min = self.heating_threshold - self.indoortemp_ref

        # Define the non-linear parts of the equations
        f_ref = d_out_in_ref * (1.4347 + 0.021 * d_out_in_ref + 247.9 * 10 ** -6 * d_out_in_ref ** 2)
        f_min = d_out_in_min * (1.4347 + 0.021 * d_out_in_min + 247.9 * 10 ** -6 * d_out_in_min ** 2)

        # Set up the system of equations in matrix form
        # Form the coefficient matrix
        A = np.array([
            [1, -f_ref],  # Coefficients for level and slope in equation 1
            [1, -f_min]  # Coefficients for level and slope in equation 2
        ])

        # Right-hand side values (adjusting for RTSoll)
        b = np.array([self.flow_temp_ref - self.indoortemp_ref, self.flow_temp_min - self.indoortemp_ref])

        level, slope = np.linalg.solve(A, b)

        return level, slope

    def _calc_flowtemp_t(self):
        # flow temperature is calculated for daily min. temperature
        ts_per_day = const.TS_PER_HOUR * 24
        if self.indoortemp_t is None:
            # d_out_in_t = self.outtemp_t - self.indoortemp_ref  # use hourly temperatures
            d_out_in_t = calc_for_blocks_of_n(self.temp_air_t, func='min', block_size=ts_per_day) - self.indoortemp_ref  # use daily mean temp.
        else:
            print(f"{self}: Nightly reduction of flow temperature activated.")
            # d_out_in_t = self.outtemp_t - self.indoortemp_t  # use hourly temperatures
            d_out_in_t = calc_for_blocks_of_n(self.temp_air_t, block_size=ts_per_day) - self.indoortemp_t  # use daily mean temp.

        return d_out_in_t.apply(lambda x: self._calc_t_vl(x, self._heating_curve_level, self._heating_curve_slope,
                                                          self.indoortemp_ref, self.flow_temp_min))

    def _calc_t_vl(self, t_diff, level=None, slope=None, t_in_ref=None, t_vl_min=None):
        if level is None:
            level = self._heating_curve_level
        if slope is None:
            slope = self._heating_curve_slope
        if t_in_ref is None:
            t_in_ref = self.indoortemp_ref
        if t_vl_min is None:
            t_vl_min = self.flow_temp_min

        return max(int(t_in_ref + level - slope * t_diff * (
                1.4347 + 0.021 * t_diff + 247.9 * 10 ** -6 * t_diff ** 2)), t_vl_min)


class SolarPotential(Component):
    """
    Represents the calculation and management of solar potential for a specific surface orientation
    and setup. This class facilitates the preparation of time series for solar generation profiles based
    on input parameters such as surface tilt, azimuth angle, and horizon factors. It is used to calculate PV generation,
    Solar thermal generation and solar gains.

    This class is designed to calculate solar energy potential using data from PVGIS (Photovoltaic Geographical
    Information System). It allows the user to specify surface properties and other parameters affecting solar
    potential calculation. It processes and normalizes the time series data for efficient usage in further analysis.

    :ivar surface_tilt: The tilt angle of the surface for solar energy calculation.
    :ivar surface_azimuth: The azimuth (compass direction) angle of the surface for solar energy calculation.
    :ivar gen_ts_normalized: Time series of normalized energy generation potential.
    :type surface_tilt: float
    :type surface_azimuth: float
    :type gen_ts_normalized: pd.Series
    """
    surface_tilt: float = None
    surface_azimuth: float = None
    gen_ts_normalized: pd.Series = None

    def __init__(self, name, surface_tilt, surface_azimuth, sel_weeks, horizon=None, window_area_m2=const.MAX_AREA,
                 window_g_val=0.75, solar_pot_area_m2=const.MAX_AREA, rad_corr_factor=0.41, **params):
        super().__init__(name, **params)
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.horizon = horizon
        self._get_pvgis_ts(sel_weeks)
        self.window_area_m2 = window_area_m2
        self.g_val = window_g_val
        self.solar_pot_area_m2 = solar_pot_area_m2
        # r = product of correction factors: frame 0.7, incidence 0.85, dirt 0.95, shade 0.75 = 0.42
        self.rad_corr_factor = rad_corr_factor

    def _get_pvgis_ts(self, sel_weeks):
        try:
            red_ts = load_pickle(
                f"{create_ts_data_dump_id()}_solar_pot_{self.surface_tilt}_{self.surface_azimuth}"
                f"_{self.horizon.replace(',', '') if self.horizon is not None else ''}")
        except FileNotFoundError:
            pv_gis_data = get_pvgis_hourly_data(surface_tilt=self.surface_tilt, surface_azimuth=self.surface_azimuth,
                                                horizon=self.horizon)
            print(f"PV potential: {self.name}, tilt: {self.surface_tilt}°, azimuth {self.surface_azimuth}°: "
                  f"{pv_gis_data[PvgisDataCols.P_el_t].sum()} kWh/kWp")

            if const.TS_PER_HOUR == const.TS_RES.QUARTER_HOURLY.value:
                pv_gis_data.convert_to_quarter_hourly()

            red_ts, _ = prepare_ts_profiles(
                {
                    InputData.PvgisData: pv_gis_data,
                },
                sel_weeks=sel_weeks
            )
            dump_pickle(
                f"{create_ts_data_dump_id()}_solar_pot_{self.surface_tilt}_{self.surface_azimuth}_"
                f"{self.horizon.replace(',', '') if self.horizon is not None else ''}", red_ts)

        self.gen_ts_normalized = red_ts[PvgisDataCols.P_el_t]
        self.direct_rad_t = red_ts[PvgisDataCols.Direct_rad]
        self.total_rad_t = red_ts[PvgisDataCols.Direct_rad] + red_ts[PvgisDataCols.Diffuse_rad_sky] + \
                           red_ts[PvgisDataCols.Diffuse_rad_ground]


class SolarPotentialDummy(Component):
    def __init__(self, name, gen_ts_normalized: pd.Series, **params):
        super().__init__(name, **params)
        self.gen_ts_normalized = gen_ts_normalized
        self.solar_pot_area_m2 = const.MAX_AREA





class Source(Component):
    """Base class a Source component. The model consists of sources, sinks and nodes."""
    def __init__(self, name, **params):
        super().__init__(name, **params)


class Sink(Component):
    """Base class a Sink component. The model consists of sources, sinks and nodes."""
    def __init__(self, name, **params):
        super().__init__(name, **params)


class ElSource(Source):
    """Electrical source."""
    def __init__(self, name, **params):
        super().__init__(name, **params)


class ElSink(Sink):
    """Electrical sink."""
    def __init__(self, name, **params):
        super().__init__(name, **params)


class ThSource(Source):
    """
    Thermal source.

    Can have one or more temperature levels, provided via
    the `temp_levels_feed` parameter. It ensures that the provided objects are valid
    and handles initialization in conjunction with its parent `Source` class.

    :ivar temp_levels_feed: A list containing `TempLevel` objects to be managed by this class.
    :type temp_levels_feed: list
    """
    def __init__(self, name, temp_levels_feed:list=None, **params):
        for temp in temp_levels_feed:
            assert isinstance(temp, TempLevel), "temp_levels_feed must be list of TempLevel objects"
        self.temp_levels_feed = temp_levels_feed
        super().__init__(name, **params)
        assert self.temp_levels_feed is not None, "ThSink attribute temp_levels_feed must not be None"


class ThSink(Sink):
    """
    Thermal sink.

    Can have one or more temperature levels, provided via
    the `temp_levels_feed` parameter. It ensures that the provided objects are valid
    and handles initialization in conjunction with its parent `Source` class.

    :ivar temp_levels_feed: A list containing `TempLevel` objects to be managed by this class.
    :type temp_levels_feed: list
    """

    def __init__(self, name, temp_levels_drain: list=None, **params):
        for temp in temp_levels_drain:
            assert isinstance(temp, TempLevel), "temp_levels_drain must be list of TempLevel objects"
        self.temp_levels_drain = temp_levels_drain
        super().__init__(name, **params)
        assert self.temp_levels_drain is not None, "ThSink attribute temp_levels_drain must not be None"


class Node(Component):
    """Nodes are points for which energy balances are solved in the model, e.g. a temperature level"""
    sources: [Source] = None
    sinks: [Sink] = None

    def __init__(self, name, sources, sinks, **params):
        self.sources = sources
        self.sinks = sinks
        super().__init__(name, **params)


class ThNode(Node):
    """
    Thermal node attributed with one specified temperature level.
    It is ensured that all associated sources and sinks are of the correct thermal types (ThSource and ThSink).

    :ivar temp_level: Temperature level associated with this node.
    :type temp_level: TempLevel
    """

    def __init__(self, name, sources, sinks, temp_level:TempLevel, **params):
        self.temp_level = temp_level
        super().__init__(name, sources, sinks, **params)
        self._check_source_sink_types()

    def _check_source_sink_types(self):
        are_th_sinks = [isinstance(sink, ThSink) for sink in self.sinks]
        are_th_sources = [isinstance(source, ThSource) for source in self.sources]

        return (all(are_th_sinks) & all(are_th_sources))


class RoomheatNode(ThNode):
    """
    Specilized thermal node for room heating.

    :ivar temperature: Represents the temperature of the room heating node,
        initialized to None by default.
    :type temperature: None or float or int
    """
    temperature = None

    def __init__(self, name, sources, sinks, **params):
        super().__init__(name, sources, sinks, **params)


class ElNode(Node):
    """
    Electrical node.

    It is validated that all sources are instances of ElSource and all sinks are instances of ElSink.

    :ivar name: The name of the node.
    :type name: str
    :ivar sources: A list of source components connected to this node.
    :type sources: list
    :ivar sinks: A list of sink components connected to this node.
    :type sinks: list
    """

    def __init__(self, name, sources, sinks, **params):
        super().__init__(name, sources, sinks, **params)
        self._check_source_sink_types()

    def _check_source_sink_types(self):
        are_el_sinks = [isinstance(sink, ElSink) for sink in self.sinks]
        are_el_sources = [isinstance(source, ElSource) for source in self.sources]

        return (all(are_el_sinks) & all(are_el_sources))


class Link(Source, Sink):
    """
    Connection between two nodes.

    A class representing a connection between two nodes with an associated
    efficiency value. It integrates functionalities of both Source and Sink,
    and it can hold additional parameters provided during initialization.

    :ivar eff: Efficiency of the connection between the nodes.
    :type eff: float
    """
    eff: float = 1.0

    def __init__(self, name, eff=1.0, **params):
        self.eff = eff
        super().__init__(name, **params)


class HeatExchanger(ThSink, ThSource, Link):
    """
    Represents a heat exchanger connecting two thermal nodes.

    A HeatExchanger facilitates the transfer of thermal energy between two
    systems. It can serve as both a thermodynamic source and sink. The primary
    usage of the HeatExchanger is defined by its feed and drain temperature
    levels.

    :ivar temp_levels_feed: Temperature levels of the feed fluid. Must contain exactly one temperature level.
    :type temp_levels_feed: list
    :ivar temp_levels_drain: Temperature levels of the drain fluid. Must contain exactly one temperature level.
    :type temp_levels_drain: list
    """

    def __init__(self, name, temp_levels_feed: list = None, temp_levels_drain: list = None, **params):
        super().__init__(name, temp_levels_feed=temp_levels_feed, temp_levels_drain=temp_levels_drain, **params)
        assert len(self.temp_levels_feed) == 1 and len(self.temp_levels_drain) == 1, \
            "HeatExchanger can only have one feed and drain TempLevel"


class ElectricLine(Link, ElSink, ElSource):
    """
    Represents an electric line used to link two electrical nodes.

    The ElectricLine class connects electrical components, serving as both a sink
    and a source of electricity. It inherits from the Link, ElSink, and ElSource
    classes to handle its complex functionalities. It is primarily used in systems
    requiring bidirectional electrical flow or monitoring of connections.

    :ivar name: Name of the electric line.
    :type name: str
    """
    def __init__(self, name, **params):
        super().__init__(name, **params)


class SolarTechnology(Component):
    """
    Represents a general technology utilizing solar potential to generate power.
    Used as base class for PV and SolarThermal.

    This class is designed for solar energy technology, leveraging a defined
    potential (such as SolarPotential or SolarPotentialDummy). It includes
    attributes to define the surface tilt and azimuth for solar panel
    orientation and allows additional configuration through its constructor.

    :ivar potential: The solar potential (e.g., SolarPotential or
        SolarPotentialDummy) used by this technology.
    :type potential: SolarPotential | SolarPotentialDummy
    :ivar surface_tilt: The tilt angle of the solar panel surface, relative
        to the ground.
    :type surface_tilt: float
    :ivar surface_azimuth: The azimuthal angle of the solar panel, describing
        its orientation relative to true north.
    :type surface_azimuth: float
    """
    potential: SolarPotential | SolarPotentialDummy = None
    surface_tilt: float = None
    surface_azimuth: float = None

    def __init__(self, name, potential: SolarPotential|SolarPotentialDummy=None, co2eq_per_cap=0,
                 kwp_per_m2=0.23,
                 **params):
        assert potential is not None, "SolarTechnology needs attribute potential"
        self.potential = potential
        self.kwp_per_m2 = kwp_per_m2
        if hasattr(potential, 'surface_tilt'):
            self.surface_tilt = potential.surface_tilt
        elif 'surface_tilt' in params:
            self.surface_tilt = params['surface_tilt']
        elif isinstance(potential, SolarPotentialDummy):
            pass
        else:
            raise AssertionError('no param surface_tilt given')
        if hasattr(potential, 'surface_azimuth'):
            self.surface_azimuth = potential.surface_azimuth
        elif 'surface_azimuth' in params:
            self.surface_tilt = params['surface_azimuth']
        elif isinstance(potential, SolarPotentialDummy):
            pass
        else:
            raise AssertionError('no param surface_azimuth given')
        super().__init__(name, co2eq_per_cap=co2eq_per_cap, **params)


class PV(SolarTechnology, ElSource):
    """
    Represents a photovoltaic (PV) system. This class provides
    properties and parameters necessary to model and evaluate the performance
    of a PV system over its lifecycle. The possible generation is given with the attributed SolarPotential.

    :ivar pmax_at_lifeend: The maximum power output as a fraction of initial
                           power at the end of the PV system's lifecycle.
    :type pmax_at_lifeend: float
    """

    def __init__(self, name, potential: SolarPotential | SolarPotentialDummy=None,
                 co2eq_per_cap=800, pmax_at_lifeend=0.9, kwp_per_m2=0.23,
                 **params):
        self.pmax_at_lifeend = pmax_at_lifeend
        super().__init__(name, potential=potential, co2eq_per_cap=co2eq_per_cap, kwp_per_m2=kwp_per_m2, **params)


class SolarThermal(SolarTechnology, ThSource):
    """
    Represents a solar thermal system. This class serves as an extension of existing solar
    technology to focus on capturing and utilizing heat energy. It calculates the
    maximum possible thermal output based on specified parameters, environmental
    conditions, and collector efficiency.

    :ivar temp_levels_feed: List containing temperature levels of the feed for
        which the thermal energy output is calculated.
    :type temp_levels_feed: list
    :ivar temp_air_t: Time series data of ambient air temperature used for thermal
        calculations.
    :type temp_air_t: pd.Series
    :ivar eta0: Optical efficiency of the solar collector.
    :type eta0: float
    :ivar a1: Heat loss coefficient related to temperature difference (linear term).
    :type a1: float
    :ivar a2: Heat loss coefficient related to temperature difference (quadratic term).
    :type a2: float
    :ivar qmax_t: Dictionary mapping feed temperature levels to their maximum
        thermal output over time.
    :type qmax_t: dict
    """

    def __init__(self, name, potential: SolarPotential=None, temp_levels_feed:list=None, temp_air_t:pd.Series=None,
                 co2eq_per_cap=100, collector_type=const.STCollectorType.FLATPLATE_SINGLECOVER_SEL,
                 kwp_per_m2=1,
                 **params):
        self.temp_levels_feed = temp_levels_feed
        assert temp_air_t is not None, "SolarThermal needs temp_air_t attribute"
        self.temp_air_t = temp_air_t
        self.eta0 = const.STCollectorSpecs[collector_type]['eta0']
        self.a1 = const.STCollectorSpecs[collector_type]['a1']
        self.a2 = const.STCollectorSpecs[collector_type]['a2']
        super().__init__(name, potential=potential, temp_levels_feed=temp_levels_feed, co2eq_per_cap=co2eq_per_cap,
                         kwp_per_m2=kwp_per_m2, **params)
        self.qmax_t = self.calc_qmax_t()

    def calc_qmax_t(self) -> dict:
        """from quaschning"""
        qmax_t = {}
        for feed_temp in self.temp_levels_feed:
            delta_temp = feed_temp.temp_t - self.temp_air_t
            qmax_t[feed_temp] = (
                    (self.potential.total_rad_t * self.eta0 - (self.a1*delta_temp + self.a2*delta_temp**2)) / 1000
            ).apply(lambda x: x if x > 0 else 0)

        return qmax_t


class Heatpump(ThSink, ThSource, ElSink):
    """
    Class representing a heat pump system.

    This class allows users to specify the heat pump's temperature levels for both the source and the
    heat sink, along with the corresponding specifications for the coefficients of
    performance (COP) and maximum heat generation (Qmax) values. It also includes methods to interpolate
    and calculate these values based on given temperature levels. The class ensures
    that the heat pump can only have one drain temperature level.

    :ivar name: Name of the heatpump.
    :type name: str
    :ivar temp_levels_drain: List of temperature levels of the drain.
    :type temp_levels_drain: list
    :ivar temp_levels_feed: List of temperature levels of the feed.
    :type temp_levels_feed: list
    :ivar specs_table: DataFrame containing specifications for COP and Qmax.
    :type specs_table: pandas.DataFrame
    :ivar heating_threshold: Heating threshold value for the heat pump.
    :type heating_threshold: Any
    :ivar src_temp_t: Source temperature at the drain side.
    :type src_temp_t: Any
    :ivar cop_t: Computed coefficients of performance for the heat pump over time.
    :type cop_t: dict
    :ivar qmax_t: Computed maximum heating capacity for the heat pump over time.
    :type qmax_t: dict
    """
    def __init__(self, name, temp_levels_drain:list=None, temp_levels_feed:list=None,
                 specs_table: pd.DataFrame= const.COP_TABLE['air-water'],
                 heating_threshold=const.HEATING_THRESHOLD,
                 **params):
        super().__init__(name, temp_levels_drain=temp_levels_drain, temp_levels_feed=temp_levels_feed, **params)
        assert len(self.temp_levels_drain) == 1, "Heatpump can only have one drain-TempLevel"
        self.specs_table = specs_table
        self.src_temp_t = self.temp_levels_drain[0].temp_t
        self.heating_threshold = heating_threshold
        self._add_values_for_flow_temps_to_cop_table()
        self.cop_t = self._calc_cop_series()
        self.qmax_t = self._calc_qmax_series()

    def _add_values_for_flow_temps_to_cop_table(self):
        for temp_level in self.temp_levels_feed:
            flow_temps_in_spec = self.specs_table.loc['COP'].index
            if isinstance(temp_level.temp_t, pd.Series):
                flow_temps = temp_level.temp_t.dropna().unique().tolist()
            else:
                flow_temps = [temp_level.temp_t]
            for flow_temp in flow_temps:
                if flow_temp not in flow_temps_in_spec:
                    self.specs_table.loc[('COP', flow_temp), :] = np.nan
                    self.specs_table.loc[('Qmax', flow_temp), :] = np.nan

            self.specs_table = self.specs_table.sort_index(level=[0,1]).interpolate(method='linear')

            too_low_cond = self.specs_table.index.get_level_values(1) < flow_temps_in_spec[0]
            if not self.specs_table.loc[too_low_cond,:].empty:
                print(f"Warning: COP and Qmax for flow temperatures < {flow_temps_in_spec[0]}°C not in specs table of"
                      f"'{self.name}'. "
                      f"Values for {flow_temps_in_spec[0]}°C are used.")
                self.specs_table.loc[('COP', too_low_cond), :] = \
                    self.specs_table.loc[('COP', flow_temps_in_spec[0]), :].values
                self.specs_table.loc[('Qmax', too_low_cond), :] = \
                    self.specs_table.loc[('Qmax', flow_temps_in_spec[0]), :].values

            too_high_cond = self.specs_table.index.get_level_values(1) > flow_temps_in_spec[-1]
            if not self.specs_table.loc[too_high_cond,:].empty:
                print(f"Warning: COP and Qmax for flow temperatures > {flow_temps_in_spec[-1]}°C not in specs table of "
                      f"'{self.name}'. "
                      f"Values for higher temperatures are set to 0 (Heatpump cannot deliver flow temperature).")
                self.specs_table.loc[('COP', too_high_cond), :] = 0
                self.specs_table.loc[('Qmax', too_high_cond), :] = 0

    def _calc_cop_series(self):
        cop_t = {}
        for temp_level in self.temp_levels_feed:
            src_n_flow_temp_t = pd.DataFrame(
                {
                    'flow': temp_level.temp_t,
                    'src': self.src_temp_t,
                }, dtype=float
            )
            cop_t[temp_level] = src_n_flow_temp_t.apply(lambda x: np.interp(
                x['src'], self.specs_table.columns, self.specs_table.loc[('COP', x['flow'])]), axis=1)

        return cop_t

    def _calc_qmax_series(self):
        qmax_t = {}
        for temp_level in self.temp_levels_feed:
            src_n_flow_temp_t = pd.DataFrame(
                {
                    'flow': temp_level.temp_t,
                    'src': self.src_temp_t,
                }, dtype=float
            )
            qmax_t[temp_level] = src_n_flow_temp_t.apply(lambda x: np.interp(
                x['src'], self.specs_table.columns, self.specs_table.loc[('Qmax', x['flow'])]), axis=1)

        return qmax_t


class CombustionHeater(ThSource):
    """
    Simple representation for a combustion heater, e.g. boiler (wood, oil, gas).

    It is initialized with a name and a list of temperature levels for the feed.
    Fuel costs need to be given with the var_cost_t parameter. The efficiency of the combustion process can be
    considered by applying a correction factor to the var_cost_t parameter, e.g. 1/0.9 for 90% combustion efficiency.

    Additional parameters can also be supplied as keyword arguments.

    :ivar name: The name of the combustion heater.
    :type name: str
    :ivar temp_levels_feed: Optional list of temperature levels for the feed.
    :type temp_levels_feed: list
    """
    def __init__(self, name, temp_levels_feed:list=None, **params):
        super().__init__(name, temp_levels_feed=temp_levels_feed, **params)


# direct electric heater
class DirectElHeater(ThSource, ElSink):
    """
    Direct electric heater / heating rod.

    Represents the operation of a heater with a specified constant
    efficiency, accepting certain temperature levels. The class can be
    customized through various parameters passed during initialization.

    :ivar eff: Efficiency of the direct electric heater, representing the
        ratio of useful heat output to the electric energy consumed.
    :type eff: float
    """

    def __init__(self, name, temp_levels_feed:list=None, eff=0.9, **params):
        self.eff = eff
        super().__init__(name, temp_levels_feed=temp_levels_feed, **params)


class StorageTechnology(Sink, Source):
    """
    Represents a storage technology that can act as both a sink and a source.

    This class encapsulates the functionality for a storage technology, including charging,
    discharging, and storage efficiency, as well as constraints on power and capacity. It is
    designed to model energy storage systems with various characteristics and constraints.
    It is the base class for ThermalStorage and ElectricalStorage.

    :ivar eff_in: Charging efficiency of the storage technology.
    :type eff_in: float
    :ivar eff_out: Discharging efficiency of the storage technology.
    :type eff_out: float
    :ivar eff_store: Storage efficiency retained from one timestep to the next.
    :type eff_store: float
    :ivar p_max_in: Maximum allowable charging power for the storage technology. UNUSED so far
    :type p_max_in: float
    :ivar p_max_out: Maximum allowable discharging power for the storage technology. UNUSED so far
    :type p_max_out: float
    :ivar c_max: Maximum energy capacity of the storage technology.
    :type c_max: float
    :ivar c_max_rel: Relative maximum charge state of the storage technology, expressed as
        a fraction of its maximum capacity.
    :type c_max_rel: float
    :ivar c_min: Minimum energy capacity of the storage technology.
    :type c_min: float
    :ivar c_min_rel: Depth of discharge, expressed as a fraction of maximum capacity.
    :type c_min_rel: float
    :ivar c_rate_max: Maximum charge/discharge rate relative to the capacity.
    :type c_rate_max: float
    """
    eff_in: float = None   # charging efficiency
    eff_out: float = None  # discharging efficiency
    eff_store: float = None    # storage efficiency from one timestep to the next
    p_max_in: float = None
    p_max_out: float = None
    c_max: float = None
    c_max_rel: float = None   # max charge state
    c_min: float = None
    c_min_rel: float = None   # depth of discharge
    c_rate_max: float = None

    def __init__(self, name, c_max=const.MAX_C, c_min=0.0, eff_in=1.0, eff_out=1.0, eff_store=0.999,
                 p_max_in: float=const.MAX_P, p_max_out: float=const.MAX_P, c_rate_max=1.0, c_max_rel=1.0,
                 c_min_rel=0.0, c_init_rel=0.0, **params):
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.eff_store = eff_store
        self.p_max_in = p_max_in
        self.p_max_out = p_max_out
        self.c_max = c_max
        self.c_max_rel = c_max_rel
        self.c_min = c_min
        self.c_min_rel = c_min_rel
        self.c_rate_max = c_rate_max
        self.c_init_rel = c_init_rel
        super().__init__(name, **params)


class ElStorage(StorageTechnology, ElSink, ElSource):
    """
    Represents an electric storage unit that functions as both an electricity sink and source.

    This class models the behavior of an energy storage system with parameters for efficiency, capacity,
    and operational constraints. It supports features such as degradation handling, adjustable maximum
    charge/discharge rates, and constant voltage charge point configurations.

    :ivar max_cycles: The maximum number of charge/discharge cycles the storage unit can handle before its
        end-of-life capacity is reached.
    :type max_cycles: int
    :ivar c_max_at_lifeend: The relative maximum capacity of the storage unit at its end of life, expressed
        as a fraction of the initial maximum capacity.
    :type c_max_at_lifeend: float
    :ivar endofcharge_eq_slope_offset: The slope and offset values for the linear equation defining the
        transition between charging modes (constant current to constant voltage) and
        the end-of-charge condition.
    :type endofcharge_eq_slope_offset: tuple[float, float]
    """
    def __init__(self, name, c_max=const.MAX_C, c_min=0.0, eff_in=0.98, eff_out=0.95, eff_store=0.9999,
                 p_max_in: float=None, p_max_out: float=None, c_rate_max=0.5, max_cycles=3000, c_max_at_lifeend=0.8,
                 c_max_rel=0.95, c_min_rel=0.2, co2eq_per_cap=230, const_voltage_charge_points: list = None,
                 c_init_rel=0.0, **params):
        """
        :param degradation_rate: capacity loss per cycle
        """
        self.max_cycles = max_cycles
        self.c_max_at_lifeend = c_max_at_lifeend
        if const_voltage_charge_points is None:
            # first point determines SOC at which charging mode switches from constant current to constant voltage
            # second point is cut-off point for charging
            const_voltage_charge_points = [0.8, 0.98]
        self.endofcharge_eq_slope_offset = self._calc_endofcharge_eurrent_eq(const_voltage_charge_points)
        super().__init__(name, c_max=c_max, c_min=c_min, eff_in=eff_in, eff_out=eff_out, eff_store=eff_store,
                         p_max_in=p_max_in, p_max_out=p_max_out, c_rate_max=c_rate_max, c_max_rel=c_max_rel,
                         c_min_rel=c_min_rel, co2eq_per_cap=co2eq_per_cap, c_init_rel=c_init_rel, **params)

    @staticmethod
    def _calc_endofcharge_eurrent_eq(const_voltage_charge_points):
        # Extract x and y values
        x = const_voltage_charge_points
        y = [1.0, 0.0]

        # Solve for coefficients (slope and intercept)
        A = np.vstack([x, np.ones(len(x))]).T  # Create the design matrix
        slope, offset = np.linalg.lstsq(A, y, rcond=None)[0]  # Solve for slope (m) and intercept (b)

        return slope, offset


class ThStorage(StorageTechnology, ThSink, ThSource):
    """
    Thermal Energy Storage class.

    This class represents a thermal energy storage system, integrating functionality
    for defining temperature levels, storage capacities, efficiencies, and other operational
    constraints. It extends multiple base classes, enabling it to act as both a thermal
    energy source and sink. The class is designed to model thermal energy storage systems
    and their behavior in various simulation or real-world setups.

    :ivar temp_levels: List of temperature levels that define the thermal storage system.
    :type temp_levels: list
    """
    def __init__(self, name, temp_levels:list=None,
                 c_max=const.MAX_C, c_min=0.0, eff_in=0.95, eff_out=0.95, eff_store=0.995, p_max_in: float=None,
                 p_max_out: float=None, c_rate_max=1.0, c_init_rel=0.0, **params):
        self.temp_levels = temp_levels
        super().__init__(name, temp_levels_drain=temp_levels, temp_levels_feed=temp_levels,
            c_max=c_max, c_min=c_min, eff_in=eff_in, eff_out=eff_out, eff_store=eff_store,
            p_max_in=p_max_in, p_max_out=p_max_out, c_rate_max=c_rate_max, c_init_rel=c_init_rel, **params)
        # ThSink.__init__(self, name, temp_levels_drain=temp_levels, **params)
        # ThSource.__init__(self, name, temp_levels_feed=temp_levels, **params)


class ThInertia(ThStorage):
    """
    Represents a thermal inertia model which calculates the effective storage attributes
    for a thermal energy storage system based on given parameters. Inherits from the
    base class ThStorage to provide functionality related to thermal energy storage.

    This class computes the maximum capacity (`c_max`) and storage efficiency (`eff_store`)
    based on the specific parameters provided, and ensures the thermal storage model
    adheres to specified constraints.

    :ivar name: Name of the thermal inertia instance.
    :type name: str
    :ivar temp_levels: List of required temperature levels for operation.
    :type temp_levels: list
    :ivar comp_volume: Storage volume in cubic meters.
    :type comp_volume: float
    :ivar transm_loss: Heat transmission loss coefficient.
    :type transm_loss: float
    :ivar delta_temp_max: Maximum temperature difference in Kelvin.
    :type delta_temp_max: float
    :ivar th_cap: Specific heat capacity in Joules per (kilogram * Kelvin).
    :type th_cap: float
    :ivar density: Material density in kilograms per cubic meter.
    :type density: float
    :ivar c_min: Minimum storage capacity.
    :type c_min: float
    :ivar eff_in: Charging efficiency coefficient.
    :type eff_in: float
    :ivar eff_out: Discharging efficiency coefficient.
    :type eff_out: float
    :ivar eff_store: Storage efficiency, determined by property parameters.
    :type eff_store: float
    :ivar c_init_rel: Initial relative capacity as a fraction of maximum capacity.
    :type c_init_rel: float
    :ivar p_max_in: Maximum input power in watts.
    :type p_max_in: float or None
    :ivar p_max_out: Maximum output power in watts.
    :type p_max_out: float or None
    :ivar c_rate_max: Maximum charge rate as a proportion of maximum capacity.
    :type c_rate_max: float
    :ivar params: Additional optional parameters for the thermal storage system.
    :type params: dict
    """
    def __init__(self, name, temp_levels:list=None,
                 comp_volume=None, transm_loss=None, delta_temp_max=2, th_cap=800, density=1500,
                 c_min=0.0, eff_in=0.9, eff_out=0.9, eff_store=0.9, c_init_rel=1.0,
                 p_max_in: float=None, p_max_out: float=None,
                 c_rate_max=None,
                 **params):
        """
        :param comp_volume: in m^3
        :param delta_temp_max: in K
        :param th_cap: cp in J/(kg*K)
        :param density: in kg/m^3
        """
        assert comp_volume is not None and transm_loss is not None, \
            "ThInertia needs arguments comp_value and transm_loss"
        c_max = comp_volume*density*th_cap/(3.6*10**6)*delta_temp_max # in kWh
        if c_max != 0:
            eff_store = 1 - (transm_loss * delta_temp_max)/c_max

        if p_max_out is None:
            # maximum discharge is assumend to be transmission loss at 30K difference between in and out
            p_max_out = transm_loss * 30
        if p_max_in is None:
            # maximum charge rate is assumed to be same as max discharge rate
            p_max_in = p_max_out
        if c_rate_max is None:
            c_rate_max = p_max_out/c_max

        super().__init__(name, temp_levels=temp_levels,
                         c_max=c_max, c_min=c_min, eff_in=eff_in, eff_out=eff_out, eff_store=eff_store,
                         p_max_in=p_max_in, p_max_out=p_max_out, c_rate_max=c_rate_max, c_init_rel=c_init_rel,
                         **params)


class MultiTempStorage(ThStorage):
    """
    Represents a storage system with multiple temperature levels.

    This class extends the `ThStorage` class to enable the usage of multiple temperature levels for thermal storage.
    This is needed to represent the possibility of e.g. raising the temperature in a buffer storage to increase its
    storage capacity. It calculates the maximum and minimum temperature for the given temperature levels,
    as well as the maximum temperature differential. The class also
    includes methods to compute thermal capacity based on the temperature
    differential and thermal storage properties.

    :ivar c_max_l: Maximum thermal capacity in liters.
    :type c_max_l: float
    :ivar c_min_l: Minimum thermal capacity in liters.
    :type c_min_l: float
    :ivar max_temp: Maximum temperature across all temperature levels.
    :type max_temp: float
    :ivar min_ret_temp: Minimum return temperature across all temperature levels.
    :type min_ret_temp: float
    :ivar max_delta: Maximum temperature differential derived from the
                    temperature levels.
    :type max_delta: float
    """
    def __init__(self, name, temp_levels:list=None, c_max_l=const.MAX_C_in_l, c_min_l=0, **params):
        for temp in temp_levels:
            assert isinstance(temp, FlowTempLevel), "temp_levels needs to be list of FlowTempLevel"
        self.c_max_l = c_max_l
        self.c_min_l = c_min_l
        self.max_temp = max([temp.temp_t.max() if isinstance(temp.temp_t, pd.Series)
                             else temp.temp_t
                             for temp in temp_levels])
        self.min_ret_temp = min([temp.return_temp_t.min() if isinstance(temp.return_temp_t, pd.Series)
                                 else temp.return_temp_t
                                 for temp in temp_levels])
        self.max_delta = self.max_temp - self.min_ret_temp
        c_max = self.calc_th_capacity(self.max_delta)
        super().__init__(name, c_max=c_max, temp_levels=temp_levels, **params)

    def calc_th_capacity(self, delta_temp):
        return const.TH_CAP_WATER_Wh_per_kg_K/10**3 * delta_temp * self.c_max_l


class Export(Sink):
    revenue_per_kwh: float = None

    def __init__(self, name, revenue_per_kwh=0.0, **params):
        self.revenue_per_kwh = revenue_per_kwh
        super().__init__(name, **params)


class ElExport(Export, ElSink):
    def __init__(self, name, revenue_per_kwh=0.082, **params):
        super().__init__(name, revenue_per_kwh=revenue_per_kwh, **params)


class ThExport(Export, ThSink):
    def __init__(self, name, revenue_per_kwh=0.0, **params):
        super().__init__(name, revenue_per_kwh=revenue_per_kwh, **params)


class Import(Source):
    costs_per_kwh: float = None

    def __init__(self, name, costs_per_kwh=0.0, **params):
        self.costs_per_kWh = costs_per_kwh
        super().__init__(name, **params)


class ElImport(Import, ElSource):
    def __init__(self, name, costs_per_kwh=0.40, atypical_consumption=False, peak_load_timesteps:pd.Series = None,
                 **params):
        self.peak_load_timesteps: pd.Series = peak_load_timesteps
        self.atypical_consumption: bool = atypical_consumption
        super().__init__(name, costs_per_kwh=costs_per_kwh, **params)


class ThImport(Import, ThSource):
    def __init__(self, name, costs_per_kwh=0.12, **params):
        super().__init__(name, costs_per_kwh=costs_per_kwh, **params)


class Demand(Sink):
    """Demand inititalized with timesieres p_t."""
    p_t: pd.Series = None

    def __init__(self, name, p_t: pd.Series, **params):
        super().__init__(name, **params)
        self.p_t = p_t


class ElDemand(Demand, ElSink):
    """Electrical demand inititalized with timesieres p_t."""
    def __init__(self, name, p_t: pd.Series, **params):
        super().__init__(name, p_t, **params)


class FlexElDemand(ElExport):
    """
    A flexible demand is a demand which can be flexibly satisfied within a period.
    They need to be defined in an input dictionary like
    flex_demands = {
        '1' : {
            'period': list <timeseries indices giving the period>
            'demand': float <sum that needs to be delivered within period>
        }
    }
    """
    def __init__(self, name, flex_demands: dict, revenue_per_kwh=0.0, **params):
        self.flex_demands: dict = flex_demands
        super().__init__(name, revenue_per_kwh=revenue_per_kwh, **params)


class ThDemand(Demand, ThSink):
    """Thermal demand inititalized with timesieres p_t."""
    def __init__(self, name, p_t: pd.Series, **params):
        """
        :param p_t: demand series in kW
        :param temp_out_t: Outside temperature series in °C
        """
        super().__init__(name, p_t, **params)


class RoomheatDemand(ThDemand):
    """
    Represents the room heat demand model, which calculates the heat demand for maintaining
    indoor temperature based on several factors such as outdoor temperature, transmission loss,
    ventilation loss, and configurable settings for indoor and nighttime temperatures.

    This class is a specialized extension of the `ThDemand` class. It determines the
    time-series of indoor air temperature and calculates temperature differences for heat-demand
    determination. It also allows custom configuration of parameters like air change rate, ventilation
    rate, and heating thresholds.

    :ivar temp_air_t: Time-series of outside air temperatures in °C.
    :type temp_air_t: pd.Series
    :ivar indoor_temp: Target indoor temperature during the day in °C.
    :type indoor_temp: float
    :ivar indoor_temp_night: Target indoor temperature during the night in °C.
    :type indoor_temp_night: float
    :ivar night_interval: Interval of nighttime hours, specified as a two-element list
                          indicating the start and end hours (24-hour format).
    :type night_interval: list
    :ivar use_dayavg_heating_threshold: Whether to use daily average temperature to determine
                                        heating thresholds.
    :type use_dayavg_heating_threshold: bool
    :ivar indoor_temp_t: Time-series of indoor temperature in °C, varying based on day/night settings.
    :type indoor_temp_t: pd.Series
    :ivar transm_loss: Transmission heat loss in kW/K.
    :type transm_loss: float
    :ivar vent_loss: Calculated ventilation heat loss in kW/K.
    :type vent_loss: float
    :ivar delta_temp_t: Time-series of temperature difference (delta T) between indoor and outdoor
                        temperatures, considering heating thresholds.
    :type delta_temp_t: pd.Series
    """
    def __init__(self, name, temp_air_t: pd.Series, transm_loss: float, air_change=const.AIR_CHANGE,
                 vent_rate=const.VENT_RATE, indoor_temp=const.NOM_TEMP_INDOOR,
                 indoor_temp_night=const.NIGHT_TEMP_INDOOR, night_interval: list=None,
                 use_dayavg_heating_threshold=False,
                 **params):
        """
        :param transm_loss: transmission heat loss in kW/K
        :param temp_air_t: Outside temperature series in °C
        """
        if night_interval is None:
            night_interval = [1,5]
        self.temp_air_t = temp_air_t
        self.indoor_temp = indoor_temp
        self.indoor_temp_night = indoor_temp_night
        self.night_interval = night_interval
        self.use_dayavg_heating_threshold = use_dayavg_heating_threshold
        self.indoor_temp_t = self._set_indoor_temp_t()
        self.transm_loss = transm_loss
        self.vent_loss = self._calc_vent_loss(air_change, vent_rate)
        self.delta_temp_t = self._calc_delta_temp_t()
        super().__init__(name, p_t=pd.Series(), **params)

    def _calc_vent_loss(self, air_change, vent_rate):
        """
        H_V = n_L * V_L * 0,34 Wh/(m3*K)
        :param air_change: n_L without unit
        :param vent_rate: V_L air flow in m^3/h
        :return: ventilation loss H_V in kW/K
        """
        return air_change * vent_rate * 0.34 / 1000

    def _set_indoor_temp_t(self) -> pd.Series:
        indoor_temp_t = pd.Series(self.indoor_temp, index=self.temp_air_t.index)
        night_inds = [i for i in self.temp_air_t.index.levels[1] if
                      (i % 24 * const.TS_PER_HOUR >= self.night_interval[0] and i % 24 * const.TS_PER_HOUR
                       <= self.night_interval[1])]
        indoor_temp_t.loc[(indoor_temp_t.index.levels[0].unique().tolist(), night_inds)] = \
            self.indoor_temp_night
        return indoor_temp_t

    def _calc_delta_temp_t(self):
        temp_df = pd.DataFrame(
            {
                'temp_air': self.temp_air_t,
                'daily_avg_temp': calc_for_blocks_of_n(self.temp_air_t),
                'indoor_temp': self.indoor_temp_t,
            }
        )
        if self.use_dayavg_heating_threshold:
            delta_t = temp_df.apply(
                lambda x:
                0 if x['daily_avg_temp'] >= const.HEATING_THRESHOLD
                else abs(x['indoor_temp'] - x['temp_air']),
                axis=1
            )
        else:
            delta_t = temp_df.apply(
                lambda x:
                x['indoor_temp'] - x['temp_air'] if x['indoor_temp'] - x['temp_air'] > 0 else 0,
                axis=1
            )
        return delta_t


class HeatGains(ThImport):
    """
    Represents the heat gains of a room by accounting for various factors such as
    temperature differences, internal gains, and solar potentials.

    This class models the heat balance in a room environment by calculating the
    temperature difference, transmission and ventilation losses, as well as the
    internal and solar heat gains. It is particularly useful for energy modeling
    and understanding heat demand dynamics.

    The purpose of this class is to compute these values based on the provided
    demand and solar potentials, enabling users to assess the energy behavior in
    varied conditions.

    :ivar delta_temp_t: Temperature difference between outdoor and indoor air
        temperature, calculated only if outdoor temperature is higher than
        indoor temperature.
    :type delta_temp_t: pandas.Series
    :ivar transm_loss: The transmission heat loss derived from the given room heat
        demand.
    :type transm_loss: float
    :ivar vent_loss: The ventilation heat loss derived from the given room heat
        demand.
    :type vent_loss: float
    :ivar internal_gains_t: The internal heat gains over time, based on predefined
        constants or values provided.
    :type internal_gains_t: pandas.Series
    :ivar solar_gains_t: The solar heat gains over time, calculated based on the
        provided solar potentials.
    :type solar_gains_t: pandas.Series
    """
    def __init__(self, name, demand: RoomheatDemand, solar_potentials: list = None,
                 internal_gains=const.INTERNAL_GAINS, costs_per_kwh=0.0, **params):

        self.delta_temp_t = self._calc_delta_temp_t(demand)
        self.transm_loss = demand.transm_loss
        self.vent_loss = demand.vent_loss
        self.internal_gains_t = self._calc_internal_gains(internal_gains)
        self.solar_gains_t = self._calc_solar_gains(solar_potentials)

        super().__init__(name, costs_per_kwh=costs_per_kwh, **params)

    @staticmethod
    def _calc_delta_temp_t(demand: RoomheatDemand):
        return demand.temp_air_t.apply(lambda x: x - demand.indoor_temp if x > demand.indoor_temp else 0)

    def _calc_internal_gains(self, internal_gains):
        return pd.Series(internal_gains, index=self.delta_temp_t.index)

    def _calc_solar_gains(self, solar_potentials):
        """ Q_S = sum(r*g_i*A_W,i*G_solar,i) """
        if solar_potentials is not None:
            gains_i = pd.DataFrame(
                {pot.name: pot.rad_corr_factor * pot.g_val * pot.window_area_m2 * pot.total_rad_t / 1000
                 for pot in solar_potentials}
            )
            gains_tot = gains_i.sum(axis=1)
        else:
            gains_tot = 0

        return gains_tot


class Refurbishment(Component):
    """
    Represents a refurbishment component that is capable of reducing demand
    by a specified factor.

    :ivar dem_red_factor: Demand reduction factor achieved through refurbishment.
    :type dem_red_factor: float
    """
    def __init__(self, name, dem_red_factor: float, **params):
        """
        :param dem_red_factor: demand reduction achieved through refurbishment
        """
        self.dem_red_factor = dem_red_factor
        super().__init__(name, **params)


if __name__ == "__main__":
    pass
    # pv = PV()
    # rad_profile = prep_climate_and_rad_data()
    # pv.calc_p_el_t(rad_profile)
    # print("")
