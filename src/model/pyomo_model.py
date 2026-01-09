"""
src/model/pyomo_model.py

Constructs a Pyomo ConcreteModel for the project's energy-system optimisation.
This module provides constraint rule functions and helpers to validate component
sets and to initialize a full optimisation model containing sets, variables,
an objective and a comprehensive set of constraints that represent electrical
and thermal balances, device limits, storage dynamics and cost/emission accounting.

Main callables:

- check_components(components)
    Validate component connectivity, types and thermal level consistency.
    Raises AssertionError or ValueError for structural or data issues.

- init_pyomo_model(components, cost_weight_factors: pd.Series, co2_price=1) -> ConcreteModel
    Build and return a configured Pyomo ConcreteModel. Creates Sets (components,
    timesteps, seasons, device classes), Vars (power, capacity, costs, flags),
    an Objective (minimise system costs) and many Constraints implemented by
    rule functions (e.g. el_node_balance_rule, th_node_balance_rule, pv_gen_rule,
    storage state equations, heatpump rules, etc.).

Key behaviours and expectations:

- Expects component objects defined in `src.model.components` (Source, Sink,
  Node, StorageTechnology, Heatpump, PV, SolarThermal, Import/Export, Demand, ...).
- `cost_weight_factors` must be a pandas Series indexed by (season, timestep)
  used to scale variable costs and annualised metrics when timeseries reduction
  is employed.
- The module assembles model structure only; it does not solve the model or
  perform file I/O. The initializer prints a short status line and attaches
  cost weights / CO2 price to the model instance.
- Validation helpers raise informative assertions for misconfigured networks or
  out-of-range demand data.

Dependencies:

- pyomo.environ (ConcreteModel, Var, Set, Constraint, Objective, minimize, ...)
- pandas (for cost weight indexing)
- project modules: `src.model.components`, `src.helper`, `src.const`

Intended use:

- Programmatic construction of optimisation models for scenario runs, unit
  testing of constraint logic, or as an input stage before invoking a Pyomo
  solver and post-processing results.
"""

from pyomo.environ import ConcreteModel, Var, Set, Constraint, Objective, minimize, summation,\
    Reals, NonNegativeReals, Binary

from src.model.components import *
from src.helper import check_if_values_outside_range, get_var_t


def el_node_balance_rule(m, node, t, s):
    balance = 0
    for src in node.sources:
        balance += m.p_el_t_feed[src, t, s]
    for sink in node.sinks:
        balance -= m.p_el_t_drain[sink, t, s]
    return balance == 0


def th_node_balance_rule(m, node, t, s):
    balance = 0
    for src in node.sources:
        balance += m.p_th_t_feed[src, node.temp_level, t, s]
    for sink in node.sinks:
        balance -= m.p_th_t_drain[sink, node.temp_level, t, s]
    return balance == 0


def objective_function(m):
    costs = (
        summation(m.inv_costs) + summation(m.fix_costs) + summation(m.var_costs)
        - summation(m.revenues)
        + summation(m.co2_costs)
    )

    return costs


def var_costs_rule(m, source):
    """ variable costs have to be weighted with factors to calculate var costs for a whole year solely based
    on reference years """
    var_costs = sum(m.p_t_feed[source, t, s] * m.cost_weights[(s, t)] * get_var_t(source.var_cost_t_levelled, (s, t))
                    for t in m.t for s in m.season) / const.TS_PER_HOUR

    return m.var_costs[source] == var_costs


def fix_costs_rule(m, comp):
    return m.fix_costs[comp] == comp.fix_cost_levelled


def inv_costs_rule(m, comp):
    if isinstance(comp, StorageTechnology):
        if isinstance(comp, MultiTempStorage):
            relevant_cap = m.c_inst_l[comp]
        else:
            relevant_cap = m.c_inst[comp]
    elif isinstance(comp, ElImport):
        if not comp.atypical_consumption:
            relevant_cap = m.p_inst[comp]
        else:
            relevant_cap = m.p_peak_hltf[comp]
    else:
        relevant_cap = m.p_inst[comp]

    return m.inv_costs[comp] == (
            comp.fix_cost_one_time_levelled * m.is_built[comp]
            + comp.fix_cost_inv_levelled * m.is_built[comp]
            + comp.inv_cost_levelled * relevant_cap
    )


def revenues_rule(m, export):
    """revenues also have to be weighted (see var costs)"""
    return (m.revenues[export] == sum(
            -get_var_t(export.var_cost_t, (s, t)) * m.p_t_drain[export, t, s] * m.cost_weights[(s, t)]
            for t in m.t for s in m.season) / const.TS_PER_HOUR)


def co2eq_rule(m, comp):
    if isinstance(comp, StorageTechnology):
        balance = comp.co2eq_per_cap * m.c_inst[comp] / comp.lifetime
    else:
        balance = comp.co2eq_per_cap * m.p_inst[comp] / comp.lifetime

    if isinstance(comp, Source):
        balance += sum(m.p_t_feed[comp, t, s] * m.cost_weights[(s, t)] * get_var_t(comp.co2eq_per_kwh_t, (s, t))
                       for t in m.t for s in m.season) / const.TS_PER_HOUR

    return m.co2eq_balance[comp] == balance


def co2_costs_rule(m, comp):
    return m.co2_costs[comp] == m.co2eq_balance[comp] * m.co2_price


def p_inst_min_rule(m, comp):
    return comp.p_min * m.is_built[comp] <= m.p_inst[comp]


def p_inst_max_rule(m, comp):
    return m.p_inst[comp] <= comp.p_max * m.is_built[comp]


def is_built_rule(m, comp):
    if comp.is_built is not None:
        return m.is_built[comp] == comp.is_built
    else:
        return m.is_built[comp] == m.is_built[comp]


def p_t_feed_max_rule(m, source, t, s):
    return m.p_t_feed[source, t, s] <= get_var_t(source.p_max_rel_t, (s, t))*m.p_inst[source]


def p_t_feed_min_rule(m, source, t, s):
    return m.p_t_feed[source, t, s] >= get_var_t(source.p_min_rel_t, (s, t))*m.p_inst[source]


def p_t_drain_max_rule(m, sink, t, s):
    return m.p_t_drain[sink, t, s] <= get_var_t(sink.p_max_rel_t, (s, t))*m.p_inst[sink]


def p_t_drain_min_rule(m, sink, t, s):
    return m.p_t_drain[sink, t, s] >= get_var_t(sink.p_min_rel_t, (s, t))*m.p_inst[sink]


def t_full_min_rule(m, comp):
    """for storages the energy going in counts"""
    if isinstance(comp, Sink):
        return (sum(m.p_t_drain[comp, t, s] for t in m.t for s in m.season) / const.TS_PER_HOUR
                >= comp.t_full_min * m.p_inst[comp])
    else:
        return (sum(m.p_t_feed[comp, t, s] for t in m.t for s in m.season) / const.TS_PER_HOUR
                >= comp.t_full_min * m.p_inst[comp])


def t_full_max_rule(m, comp):
    """for storages the energy going in counts"""
    if isinstance(comp, Sink):
        return (sum(m.p_t_drain[comp, t, s] for t in m.t for s in m.season) / const.TS_PER_HOUR
                <= comp.t_full_max * m.p_inst[comp])
    else:
        return (sum(m.p_t_feed[comp, t, s] for t in m.t for s in m.season) / const.TS_PER_HOUR
                <= comp.t_full_max * m.p_inst[comp])


def trans_rule(m, link, t, s):
    return m.p_t_feed[link, t, s] == link.eff * m.p_t_drain[link, t, s]


def trans_th_rule(m, heat_ex, temp, t, s):
    if temp == heat_ex.temp_levels_feed[0]:
        return m.p_th_t_feed[heat_ex, temp, t, s] == (
                heat_ex.eff * m.p_th_t_drain[heat_ex, heat_ex.temp_levels_drain[0], t, s])
    else:
        return m.p_th_t_feed[heat_ex, temp, t, s] == 0


def pv_gen_rule(m, pv, t, s):
    """calculation of PV generation (includes module degeneration)"""
    return m.p_t_feed[pv, t, s] <= m.p_inst[pv] * pv.potential.gen_ts_normalized[(s, t)] * (1.0 + pv.pmax_at_lifeend)/2


def st_gen_rule(m, st, temp, t, s):
    if temp in st.temp_levels_feed:
        return m.p_th_t_feed[st, temp, t, s] <= m.p_inst[st] * st.qmax_t[temp][(s, t)]
    else:
        return m.p_th_t_feed[st, temp, t, s] == 0


def tot_st_gen_calc(m, st, t, s):
    return m.p_t_feed[st, t, s] == sum(m.p_th_t_feed[st, temp, t, s] for temp in st.temp_levels_feed)


def tot_st_gen_rule(m, st, t, s):
    return m.p_t_feed[st, t, s] <= max(st.qmax_t[temp][(s, t)] for temp in st.temp_levels_feed) * m.p_inst[st]


def solarpotential_area_rule(m, pot):
    pot_user = []
    for comp in m.solar_technologies:
        if comp.potential == pot:
            pot_user.append(comp)

    return sum(m.p_inst[comp] / comp.kwp_per_m2 for comp in pot_user) <= pot.solar_pot_area_m2


def p_t_demand_rule(m, dem, t, s):
    return m.p_t_drain[dem, t, s] == dem.p_t[(s, t)]


def w_flex_demand_rule(m, flex_dem, flex_period):
    return (sum(m.p_t_drain[flex_dem, t, s] for (s,t) in flex_dem.flex_demands[flex_period]['period'])
            == flex_dem.flex_demands[flex_period]['demand'])


def w_flex_demand_outside_period_rule(m, flex_dem, t, s):
    all_flex_periods = [period for flex_demand in flex_dem.flex_demands.values() for period in flex_demand['period']]
    if (s,t) not in all_flex_periods:
        return m.p_t_drain[flex_dem, t, s] == 0
    else:
        return m.p_t_drain[flex_dem, t, s] >= 0


def p_t_roomheat_demand_rule(m, dem, t, s):

    heat_loss_wout_refurb = dem.delta_temp_t[(s, t)] * (dem.transm_loss + dem.vent_loss)

    heat_loss_w_refurb = heat_loss_wout_refurb * (
            1 - sum(m.is_built[ref] * ref.dem_red_factor for ref in m.refurbishments))

    return m.p_t_drain[dem, t, s] == heat_loss_w_refurb


def p_t_heat_gains_rule(m, imp, t, s):

    heat_gains_wout_refurb = (imp.delta_temp_t[(s, t)] * (imp.transm_loss + imp.vent_loss) +
        imp.internal_gains_t[(s, t)] + imp.solar_gains_t[(s, t)])

    heat_gains_w_refurb = heat_gains_wout_refurb * (
            1 - sum(m.is_built[ref] * ref.dem_red_factor for ref in m.refurbishments))

    return m.p_t_feed[imp, t, s] == heat_gains_w_refurb


# storages
def c_inst_min_rule(m, sto):
    return sto.c_min * m.is_built[sto] <= m.c_inst[sto]


def c_inst_max_rule(m, sto):
    return m.c_inst[sto] <= sto.c_max * m.is_built[sto]


def c_t_rule(m, sto, t, s):
    if t > 0:
        current_state = sto.eff_store * m.c_t[sto, t-1, s]
    else:
        current_state = sto.c_init_rel * m.c_inst[sto]
    return (m.c_t[sto, t, s] == current_state + m.p_t_drain[sto, t, s] / const.TS_PER_HOUR
            * sto.eff_in - 1 / sto.eff_out * m.p_t_feed[sto, t, s] / const.TS_PER_HOUR)


def c_t_max_rule(m, sto, t, s):
    return m.c_t[sto, t, s] <= sto.c_max_rel * m.c_inst[sto]


def c_t_min_rule(m, sto, t, s):
    return m.c_t[sto, t, s] >= sto.c_min_rel * sto.c_min


def max_charge_rule(m, sto, t, s):
    return m.p_t_drain[sto, t, s] / const.TS_PER_HOUR <= sto.c_rate_max * m.c_inst[sto]


def max_discharge_rule(m, sto, t, s):
    return m.p_t_feed[sto, t, s] / const.TS_PER_HOUR <= sto.c_rate_max * m.c_inst[sto]


# battery storages
def max_cycle_rule(m, sto):
    return (sto.lifetime *
            sum(m.cost_weights[(s, t)] * m.p_t_drain[sto, t, s] / const.TS_PER_HOUR for t in m.t for s in m.season)
            <= m.c_inst[sto] * sto.max_cycles)


def max_endofchargecurrent_rule(m, sto, t, s):
    return (m.p_t_drain[sto, t, s] / const.TS_PER_HOUR / sto.c_rate_max <=
            sto.endofcharge_eq_slope_offset[0] * m.c_t[sto, t, s] + sto.endofcharge_eq_slope_offset[1] * m.c_inst[sto])


def max_cap_w_degradation_rule(m, sto, t, s):
    return m.c_t[sto, t, s] <= m.c_inst[sto] * sto.c_max_rel * (1.0 + sto.c_max_at_lifeend)/2


# thermal multi-temp storages
def c_th_t_temp_rule(m, sto, temp, t, s):
    if temp in sto.temp_levels_feed:
        if t > 0:
            current_state = sto.eff_store * m.c_th_t[sto, temp, t-1, s]
        else:
            current_state = sto.c_init_rel * m.c_inst[sto]
        return (m.c_th_t[sto, temp, t, s] ==
                current_state + m.p_th_t_drain[sto, temp, t, s] / const.TS_PER_HOUR * sto.eff_in
                - m.p_th_t_feed[sto, temp, t, s] / const.TS_PER_HOUR / sto.eff_out)
    else:
        return m.c_th_t[sto, temp, t, s] == 0


def c_th_t_tot_rule(m, sto, t, s):
    return m.c_t[sto, t, s] == sum(m.c_th_t[sto, temp, t, s] for temp in sto.temp_levels_feed)


def c_th_t_in_l_rule(m, sto, t, s):
    return sum(m.c_th_t[sto, temp, t, s]/(
                const.TH_CAP_WATER_Wh_per_kg_K / 10 ** 3 * get_var_t(temp.delta_temp_t, (s, t)))
               for temp in sto.temp_levels_feed) <= sto.c_max_rel * m.c_inst_l[sto]


def c_th_t_max_rule(m, sto):
    return m.c_inst_l[sto] <= sto.c_max_l


def c_th_t_min_rule(m, sto):
    return m.c_inst_l[sto] >= sto.c_min_l


# general feed and drain rules
def p_th_t_drain_rule(m, th_sink, t, s):
    return sum(m.p_th_t_drain[th_sink, temp, t, s] for temp in th_sink.temp_levels_drain) == m.p_t_drain[th_sink, t, s]


def p_th_t_feed_rule(m, th_source, t, s):
    return sum(m.p_th_t_feed[th_source, temp, t, s] for temp in th_source.temp_levels_feed) == m.p_t_feed[th_source, t, s]


def p_el_t_drain_rule(m, el_sink, t, s):
    return m.p_el_t_drain[el_sink, t, s] == m.p_t_drain[el_sink, t, s]


def p_el_t_feed_rule(m, el_source, t, s):
    return m.p_el_t_feed[el_source, t, s] == m.p_t_feed[el_source, t, s]


def p_inst_min_node_rule(m, node):
    return sum([m.p_inst[src] for src in node.sources if not isinstance(src, StorageTechnology)]) >= node.p_min


# heatppump
def hp_p_th_t_feed_rule(m, hp, temp, t, s):
    if temp in hp.temp_levels_feed:
        return m.p_th_t_feed[hp, temp, t, s] == hp.cop_t[temp][(s, t)] * m.p_el_t_drain_temp_level[hp, temp, t, s]
    else:
        return m.p_th_t_feed[hp, temp, t, s] == 0


def hp_p_th_t_drain_rule(m, hp, t, s):
    return m.p_th_t_drain[hp, hp.temp_levels_drain[0], t, s] == sum(
        max(0, hp.cop_t[temp][(s, t)] - 1) * m.p_el_t_drain_temp_level[hp, temp, t, s] for temp in hp.temp_levels_feed)


def p_el_t_drain_temp_levels_rule(m, comp, t, s):
    return m.p_el_t_drain[comp, t, s] == sum(m.p_el_t_drain_temp_level[comp, temp, t, s]
                                             for temp in comp.temp_levels_feed)


def hp_p_th_feed_max_rule(m, hp, t, s):
    return sum(m.p_th_t_feed[hp, temp, t, s]/hp.qmax_t[temp][(s, t)] for temp in hp.temp_levels_feed
               if hp.qmax_t[temp][(s, t)] > 0 ) <= m.p_inst[hp]


def deh_p_th_t_feed_rule(m, deh, temp, t, s):
    return m.p_th_t_feed[deh, temp, t, s] == deh.eff * m.p_el_t_drain_temp_level[deh, temp, t, s]


def set_refurb_cap_rule(m, ref):
    return m.p_inst[ref] == m.is_built[ref]


def peak_load_rule(m, el_imp, t, s):
    if el_imp.atypical_consumption and el_imp.peak_load_timesteps[(s, t)] == 1:
        return m.p_t_feed[el_imp, t, s] <= m.p_peak_hltf[el_imp]
    else:
        return m.p_t_feed[el_imp, t, s] <= m.p_inst[el_imp]


def hltf_pmax_rule(m, el_imp):
    return m.p_peak_hltf[el_imp] <= const.HLTF_PEAK_LOAD_LIMIT * m.p_inst[el_imp]


# Helper constraints
# def force_high_temp_storage_rule(m, sto, temp):
#     if temp in sto.temp_levels_feed:
#         return sum(m.p_th_t_feed[sto, temp, t, s] for t in m.t for s in m.season) >= 1
#     else:
#         return sum(m.p_th_t_feed[sto, temp, t, s] for t in m.t for s in m.season) == 0


# def limit_import_rule(m, t, s):
#     return (sum(m.p_el_t_feed[src, t, s] for src in m.el_sources if src not in m.links) <=
#             sum(m.p_el_t_drain[sink, t, s] for sink in m.el_sinks if sink not in (m.el_exports|m.links)))




def check_components(components):
    """
    Validates the structure, relationships, and data types within a components dictionary, ensuring
    correct connectivity between nodes, sinks, sources, and other elements.

    :param components: A dictionary containing various system components categorized into groups such as
        'nodes', 'links', 'demand', etc., which may include `Sink`, `Source`, `Node`, `SolarPotential`,
        `SolarPotentialDummy`, `Refurbishment`, `ThNode`, and `Demand` objects.
    :type components: dict

    :return: None

    :raises AssertionError: If links in the 'links' group reference undefined node names.
    :raises AssertionError: If components within any group are not instances of expected classes
        (`Sink`, `Source`, `Node`, `SolarPotential`, `SolarPotentialDummy`, or `Refurbishment`).
    :raises AssertionError: If temperature levels of thermal-related nodes do not match those of the
        connected sinks or sources within the 'nodes' group.
    :raises AssertionError: If any sink or source is not connected to a node.
    :raises ValueError: If any `Demand` object contains values outside the acceptable range.
    """

    node_names = [components['nodes'][node].name for node in components['nodes']]
    all_sinks = []
    all_srcs = []

    # check if links connect nodes, i.e. names defined correctly
    for link in components['links']:
        lname = components['links'][link].name
        start_node = lname.split('-')[0]
        end_node = lname.split('-')[1]
        assert start_node in node_names, f"{start_node} not in {node_names}"
        assert end_node in node_names, f"{end_node} not in {node_names}"

    # check data types
    for comp_group in components:
        for comp in components[comp_group]:
            if not (isinstance(components[comp_group][comp], Sink)
                    or isinstance(components[comp_group][comp], Source)
                    or isinstance(components[comp_group][comp], Node)
                    or isinstance(components[comp_group][comp], SolarPotential)
                    or isinstance(components[comp_group][comp], SolarPotentialDummy)
                    or isinstance(components[comp_group][comp], Refurbishment)):
                raise AssertionError(
                    f"{components[comp_group][comp]} is not Source, Sink, Node, SolarPotential or Refurbishment")
            if isinstance(components[comp_group][comp], Sink):
                all_sinks.append(components[comp_group][comp])
            if isinstance(components[comp_group][comp], Source):
                all_srcs.append(components[comp_group][comp])

    for demand in components['demand'].values():
        if isinstance(demand, Demand):
            check_if_values_outside_range(demand.p_t)

    # check temperature levels of thermal components match
    for node in components['nodes'].values():
        if isinstance(node, ThNode):
            assert node.sinks != [], f"node {node.name} has no sinks!"
            assert node.sources != [], f"node {node.name} has no sources!"
            for sink in node.sinks:
                assert node.temp_level in sink.temp_levels_drain, \
                    f"TempLevel of node {node.name}: {node.temp_level.name} " \
                    f"not in sinks: {[t.name for t in sink.temp_levels_drain]} of component {sink.name}"
            for source in node.sources:
                assert node.temp_level in source.temp_levels_feed, \
                    f"TempLevel of node {node.name}: {node.temp_level.name} " \
                    f"not in sources: {[t.name for t in source.temp_levels_feed]} of component {source.name}"

    # check if all sinks and sources are connected to node
    conn_sinks = []
    conn_srcs = []
    for node in components['nodes'].values():
        for sink in node.sinks:
            conn_sinks.append(sink)
        for src in node.sources:
            conn_srcs.append(src)
    for sink in all_sinks:
        assert sink in conn_sinks, f"{sink} not connected to node"
    for src in all_srcs:
        assert src in conn_srcs, f"{src} not connected to node"



def init_pyomo_model(components, cost_weight_factors: pd.Series, co2_price=1) -> ConcreteModel:
    """
    Initializes a Pyomo ConcreteModel representing a framework for energy system
    optimization. This model includes various components and their properties,
    timesteps, seasons, and associated costs and emissions. Sets and variables
    are created to model the relationships and behaviors of these elements within
    an energy system while allowing for flexible parameterization and optimization.

    :param components: Dictionary categorizing various energy system components
        into groups. Nested dictionaries specify the components belonging to
        each group.
    :type components: dict
    :param cost_weight_factors: Pandas Series specifying weight factors for variable costs,
    indexed by season and timestep. They are needed to caluclate yearly operational costs when
    timeseries reduction is enabled.
    :param co2_price: Price of CO2 per kilogram in euros for calculating CO2
        emissions-related costs. Defaults to 1.
    :type co2_price: float, optional
    :return: Pyomo ConcreteModel instance defining the energy system's structure,
        constraints, and optimization variables.
    :rtype: pyomo.environ.ConcreteModel
    """
    print('initializing model... ')
    m = ConcreteModel()
    m.cost_weights = cost_weight_factors
    m.co2_price = co2_price # € / kg CO2 eq.
    m.components = Set(
        initialize=list({
            components[comp_group][comp]
            for comp_group in components
            for comp in components[comp_group]
        }),
        doc='All components'
    )
    m.t = Set(
        initialize=cost_weight_factors.index.get_level_values(1).unique().tolist(),
        doc='all timesteps in timeframe. Time-index is given by m.t x m.season'
    )
    m.season = Set(
        initialize=cost_weight_factors.index.get_level_values(0).unique().tolist(),
        doc='all seasons. Time-index is given by m.t x m.season'
    )
    m.sinks = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Sink)}),
        doc='All sinks'
    )
    m.el_sinks = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ElSink)}),
        doc='All el sinks'
    )
    m.th_sinks = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ThSink)}),
        doc='All th sinks'
    )
    m.sources = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Source)}),
        doc='All sinks'
    )
    m.el_sources = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ElSource)}),
        doc='All el sources'
    )
    m.th_sources = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ThSource)}),
        doc='All th sources'
    )
    m.nodes = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Node)}),
        doc='All sinks'
    )
    m.el_nodes = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ElNode)}),
        doc='All el. sinks'
    )
    m.th_nodes = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ThNode)}),
        doc='All th. sinks'
    )
    m.roomheat_nodes = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, RoomheatNode)}),
        doc='All th. sinks'
    )
    m.temp_levels = Set(
        initialize=[node.temp_level for node in m.th_nodes],
        doc='All components'
    )
    m.links = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Link)}),
        doc='All links'
    )
    m.heat_exchanger = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, HeatExchanger)}),
        doc='All links'
    )
    m.pv = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, PV)}),
        doc='All PV systems distinguished by tilt and azimuth'
    )
    m.solar_thermal = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, SolarThermal)}),
        doc='All solarthermal systems distinguished by tilt and azimuth'
    )
    m.solar_technologies = Set(
        initialize=m.pv | m.solar_thermal,
        doc='All solar technologies'
    )
    m.solar_energy_potentials = Set(
        initialize=list(set([st.potential for st in m.solar_technologies])),
        doc='All solar potentials (can be used by PV or solar thermal potentials)'
    )
    m.demands = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Demand)}),
        doc='All demands (th and el)'
    )
    m.flex_el_demands = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, FlexElDemand)}),
        doc='Flexible electricity demands, e.g. e-mob with optimized charging'
    )
    m.flex_el_demand_periods = Set(
        initialize=list({period for comp in m.components if isinstance(comp, FlexElDemand)
                         for period in comp.flex_demands.keys()}),
        doc='Set of periods where flexible demands need to be satisfied'
    )
    m.roomheat_demands = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, RoomheatDemand)}),
        doc='Roomheat demands (have to be calculated based on outside temp and thermal insulation)'
    )
    m.heat_gains = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, HeatGains)}),
        doc='Heat gains (are balanced out with roomheat demand)'
    )
    m.refurbishments = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Refurbishment)}),
        doc='Refurbishment meaning Delta in U-value'
    )
    m.imports = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Import)}),
        doc='All imports (th and el)'
    )
    m.exports = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Export)}),
        doc='All imports (th and el)'
    )
    m.el_imports = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ElImport)}),
        doc='Electrical imports'
    )
    m.el_exports = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ElExport)}),
        doc='Electrical exports'
    )
    m.storages = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, StorageTechnology)}),
        doc='All storages (th and el)'
    )
    m.el_storages = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, ElStorage)}),
        doc='Electrical storages'
    )
    m.th_multitemp_storages = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, MultiTempStorage)}),
        doc='Thermal storages that can store energy on multiple temperature levels'
    )
    m.heat_pumps = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, Heatpump)}),
        doc='All heatpumps'
    )
    m.dehs = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, DirectElHeater)}),
        doc='All direct electric heaters'
    )
    m.boilers = Set(
        initialize=list({comp for comp in m.components if isinstance(comp, CombustionHeater)}),
        doc='All boilers / combustion heaters'
    )

    m.inv_costs = Var(
        m.components,
        within=Reals,
        doc='Investment costs')
    m.fix_costs = Var(
        m.components,
        within=Reals,
        doc='Fixed costs')
    m.var_costs = Var(
        m.sources,
        within=Reals,
        doc='Variable costs')
    m.revenues = Var(
        m.exports,
        within=Reals,
        doc='revenues')
    m.is_built = Var(
        m.components,
        within=Binary,
        doc='fix_cost_inv are only applied if is_built=1'
    )
    m.co2eq_balance = Var(
        m.components,
        within=Reals,
        doc='Emissions in CO2 eq. [kg]'
    )
    m.co2_costs = Var(
        m.components,
        within=Reals,
        doc='Costs for CO2 eq. [€]'
    )

    m.p_t_drain = Var(
        m.sinks, m.t, m.season,
        within=NonNegativeReals,
        doc="Power of component drawn from node into sink at timestep s,t"
    )
    m.p_t_feed = Var(
        m.sources, m.t, m.season,
        within=NonNegativeReals,
        doc="Power of component delivered to node from source at timestep s,t"
    )
    m.p_el_t_drain = Var(
        m.el_sinks, m.t, m.season,
        within=NonNegativeReals,
        doc="Power of component drawn from node into sink at timestep s,t"
    )
    m.p_el_t_drain_temp_level = Var(
        m.heat_pumps | m.dehs, m.temp_levels, m.t, m.season,
        within=NonNegativeReals,
        doc="Electrical power of HP or DEH to provide heat at temp_level temp at timestep s,t"
    )
    m.p_el_t_feed = Var(
        m.el_sources, m.t, m.season,
        within=NonNegativeReals,
        doc="Power of component delivered to node from source at timestep s,t"
    )
    m.p_th_t_drain = Var(
        m.th_sinks, m.temp_levels, m.t, m.season,
        within=NonNegativeReals,
        doc="Power of component drawn from node into sink at timestep s,t"
    )
    m.p_th_t_feed = Var(
        m.th_sources, m.temp_levels, m.t, m.season,
        within=NonNegativeReals,
        doc="Power of component delivered to node from source at timestep s,t"
    )
    m.p_inst = Var(
        m.components,
        within=NonNegativeReals,
        doc="installed power of components"
    )
    m.p_peak_hltf = Var(
        m.el_imports,
        within=NonNegativeReals,
        doc="peak power during high load time frames for ElImports"
    )
    m.c_inst = Var(
        m.storages,
        within=NonNegativeReals,
        doc="installed capacity of storages"
    )
    m.c_inst_l = Var(
        m.th_multitemp_storages,
        within=NonNegativeReals,
        doc="installed capacity of multitemp-storages in l"
    )
    m.c_t = Var(
        m.storages, m.t, m.season,
        within=NonNegativeReals,
        doc="capacity of storages at timestep s,t"
    )
    m.c_th_t = Var(
        m.th_multitemp_storages, m.temp_levels, m.t, m.season,
        within=NonNegativeReals,
        doc="capacity of thermal multitemp storages at temp_level temp at timestep s,t"
    )

    # objective
    m.objective_function = Objective(
        expr=objective_function,
        sense=minimize,
        doc='minimize costs')

    # constraints
    m.el_node_balance = Constraint(
        m.el_nodes, m.t, m.season,
        rule=el_node_balance_rule,
        doc="energy balance at every node"
    )
    m.th_node_balance = Constraint(
        m.th_nodes, m.t, m.season,
        rule=th_node_balance_rule,
        doc="energy balance at every node"
    )


    m.inv_costs_calc = Constraint(
        m.components,
        rule=inv_costs_rule,
        doc='costs'
    )
    m.fix_costs_calc = Constraint(
        m.components,
        rule=fix_costs_rule,
        doc='costs'
    )
    m.var_costs_calc = Constraint(
        m.sources,
        rule=var_costs_rule,
        doc='costs'
    )
    m.revenues_calc = Constraint(
        m.exports,
        rule=revenues_rule,
        doc='revenues'
    )
    m.co2eq_calc = Constraint(
        m.components,
        rule=co2eq_rule,
        doc='calculate total co2eq emissions for each component'
    )
    m.co2_costs_calc = Constraint(
        m.components,
        rule=co2_costs_rule,
        doc='calculate costs for co2eq emissions for each component'
    )

    m.p_inst_min_bound = Constraint(
        m.components,
        rule=p_inst_min_rule,
        doc='lower bound for installed power'
    )
    m.p_inst_max_bound = Constraint(
        m.components,
        rule=p_inst_max_rule,
        doc='upper bound for installed power'
    )
    m.must_be_built = Constraint(
        m.components,
        rule=is_built_rule,
        doc='set components that must be installed'
    )
    m.p_t_feed_max_bound = Constraint(
        m.sources, m.t, m.season,
        rule=p_t_feed_max_rule,
        doc='upper bound for power at every timestep for sources'
    )
    m.p_t_feed_min_bound = Constraint(
        m.sources, m.t, m.season,
        rule=p_t_feed_min_rule,
        doc='lower bound for power at every timestep for sources'
    )
    m.p_t_drain_max_bound = Constraint(
        m.sinks, m.t, m.season,
        rule=p_t_drain_max_rule,
        doc='upper bound for power at every timestep for sinks'
    )
    m.p_t_drain_min_bound = Constraint(
        m.sinks, m.t, m.season,
        rule=p_t_drain_min_rule,
        doc='lower bound for power at every timestep for sinks'
    )
    m.t_full_min_bound = Constraint(
        m.sources|m.sinks,
        rule=t_full_min_rule,
        doc='set min bound for full load hours'
    )
    m.t_full_max_bound = Constraint(
        m.sources|m.sinks,
        rule=t_full_max_rule,
        doc='set max bound for full load hours'
    )
    # PV
    m.pv_p_el_t_feed_calc = Constraint(
        m.pv, m.t, m.season,
        rule=pv_gen_rule,
        doc='generation series in relation to p_inst'
    )
    # Solar thermal
    m.st_p_th_t_feed_calc = Constraint(
        m.solar_thermal, m.temp_levels, m.t, m.season,
        rule=st_gen_rule,
        doc='generation series in relation to p_inst for each temperature level'
    )
    m.st_p_t_feed_calc = Constraint(
        m.solar_thermal, m.t, m.season,
        rule=tot_st_gen_calc,
        doc='total p_t_feed calculation'
    )
    m.st_p_t_feed_constraint = Constraint(
        m.solar_thermal, m.t, m.season,
        rule=tot_st_gen_rule,
        doc='constrain total p_t_feed to maximum ST generation'
    )
    m.pv_st_area_constraint = Constraint(
        m.solar_energy_potentials,
        rule=solarpotential_area_rule,
        doc='area of SolarPotential is limited, PV and ST have to share'
    )
    # demand
    m.p_t_demand_drain_calc = Constraint(
        m.demands ^ m.roomheat_demands, m.t, m.season,
        rule=p_t_demand_rule,
        doc='set equal to demands'
    )
    m.p_t_flex_demand_drain_calc = Constraint(
        m.flex_el_demands, m.flex_el_demand_periods,
        rule=w_flex_demand_rule,
        doc='sum constraint, s.t. demand is satisfied within period'
    )

    m.p_t_flex_demand_outside_period_calc = Constraint(
        m.flex_el_demands, m.t, m.season,
        rule=w_flex_demand_outside_period_rule,
        doc='outside of flex period, demand is 0'
    )
    m.p_t_roomheat_drain_calc = Constraint(
        m.roomheat_demands, m.t, m.season,
        rule=p_t_roomheat_demand_rule,
        doc='calculate room heat demand'
    )

    m.p_t_roomheat_gains_calc = Constraint(
        m.heat_gains, m.t, m.season,
        rule=p_t_heat_gains_rule,
        doc='calculate heat gains'
    )

    # thermal and el sinks separately
    m.p_th_t_drain_calc = Constraint(
        m.th_sinks, m.t, m.season,
        rule=p_th_t_drain_rule,
        doc='set P th t in'
    )
    m.p_th_t_feed_calc = Constraint(
        m.th_sources, m.t, m.season,
        rule=p_th_t_feed_rule,
        doc='set P th t out'
    )
    m.p_el_t_drain_calc = Constraint(
        m.el_sinks ^ m.heat_pumps, m.t, m.season,
        rule=p_el_t_drain_rule,
        doc='set P el t in'
    )
    m.p_el_t_feed_calc = Constraint(
        m.el_sources, m.t, m.season,
        rule=p_el_t_feed_rule,
        doc='set P el t out'
    )
    # nodes
    m.p_inst_min_heating = Constraint(
        m.nodes,
        rule=p_inst_min_node_rule,
        doc='E.g. for heating systems there is a mininum total capacity needed (Heizlast). Without storages'
    )
    # links
    m.trans_p_t_feed_calc = Constraint(
        m.links, m.t, m.season,
        rule=trans_rule,
        doc='set transmission rule for links'
    )
    m.trans_p_th_t_feed_calc = Constraint(
        m.heat_exchanger, m.temp_levels, m.t, m.season,
        rule=trans_th_rule,
        doc='set transmission rule for links'
    )
    # storages
    m.c_inst_max_bound = Constraint(
        m.storages,
        rule=c_inst_max_rule,
        doc='upper bound for installed capacity'
    )
    m.c_inst_min_bound = Constraint(
        m.storages,
        rule=c_inst_min_rule,
        doc='lower bound for installed capacity'
    )
    m.c_t_calc = Constraint(
        m.storages ^ m.th_multitemp_storages, m.t, m.season,
        rule=c_t_rule,
        doc='calculation rule for storage capacity at timestep s,t'
    )
    m.c_t_max_bound = Constraint(
        m.storages, m.t, m.season,
        rule=c_t_max_rule,
        doc='storage capacity at timestep s,t has to be lower than installed cap'
    )
    m.c_t_min_bound = Constraint(
        m.storages, m.t, m.season,
        rule=c_t_min_rule,
        doc='storage capacity at timestep s,t has to be higher than c_min'
    )
    m.charge_bound = Constraint(
        m.storages, m.t, m.season,
        rule=max_charge_rule,
        doc='charge must be smaller than c_rate_max'
    )
    m.discharge_bound = Constraint(
        m.storages, m.t, m.season,
        rule=max_discharge_rule,
        doc='discharge must be smaller than c_rate_max'
    )
    # El. storages
    m.max_cycle_bound = Constraint(
        m.el_storages,
        rule=max_cycle_rule,
        doc='maximum number of battery cycles until EOL is reached'
    )
    m.max_endofchargecurrent = Constraint(
        m.el_storages, m.t, m.season,
        rule=max_endofchargecurrent_rule,
        doc='constraint for maximum charging current when constant voltage charging is reached'
    )
    m.max_cap_w_degradation_bound = Constraint(
        m.el_storages, m.t, m.season,
        rule=max_cap_w_degradation_rule,
        doc='calculate average capacity over lifetime'
    )
    # Multi-temp storages
    m.c_th_t_temp_calc = Constraint(
        m.th_multitemp_storages, m.temp_levels, m.t, m.season,
        rule=c_th_t_temp_rule,
        doc='calculate c_th_t for each temperature level'
    )
    m.c_th_t_tot_calc = Constraint(
        m.th_multitemp_storages, m.t, m.season,
        rule=c_th_t_tot_rule,
        doc='calculate c_t from c_th_t'
    )
    m.c_th_t_in_l = Constraint(
        m.th_multitemp_storages, m.t, m.season,
        rule=c_th_t_in_l_rule,
        doc='storage capacity in l at timestep s,t has to be lower than installed cap in l'
    )
    m.c_th_t_max_bound = Constraint(
        m.th_multitemp_storages,
        rule=c_th_t_max_rule,
        doc='storage capacity in l has to be smaller than sto.c_max_l'
    )
    m.c_th_t_min_bound = Constraint(
        m.th_multitemp_storages,
        rule=c_th_t_min_rule,
        doc='storage capacity in l has to be larger than sto.c_min_l'
    )

    # heatpumps
    m.hp_p_th_t_feed_calc = Constraint(
        m.heat_pumps, m.temp_levels, m.t, m.season,
        rule=hp_p_th_t_feed_rule,
        doc='calculate P th t feed (heat delivered from heatpumps to drain node)'
    )
    m.hp_p_th_t_drain_calc = Constraint(
        m.heat_pumps, m.t, m.season,
        rule=hp_p_th_t_drain_rule,
        doc='calculate P th t drain (heat drawn from feed node from heatpumps)'
    )
    m.p_el_t_drain_calc_temp_levels = Constraint(
        m.heat_pumps | m.dehs, m.t, m.season,
        rule=p_el_t_drain_temp_levels_rule,
        doc='calculate total P el t drain (electricity drawn from node)'
    )
    m.hp_p_th_t_feed_max = Constraint(
        m.heat_pumps, m.t, m.season,
        rule=hp_p_th_feed_max_rule,
        doc='calculate max. P th t depending on the source temperature'
    )
    # deh
    m.deh_p_th_t_feed_calc = Constraint(
        m.dehs, m.temp_levels, m.t, m.season,
        rule=deh_p_th_t_feed_rule,
        doc='calculate P th t (heat delivered from direct electric heaters)'
    )
    # refurbishment
    m.set_refurb_cap_value = Constraint(
        m.refurbishments,
        rule=set_refurb_cap_rule,
        doc='set capacity values for refurbishments, so CO2-emissions will be considered'
    )

    # Helper constraints

    # # rules for atypical power consumption
    # m.atypical_consumption = Constraint(
    #     m.el_imports, m.t, m.season,
    #     rule=peak_load_rule,
    #     doc='determine peak load during high load time frames for which grid fees are charged'
    # )
    # m.hltf_p_max = Constraint(
    #     m.el_imports,
    #     rule=hltf_pmax_rule,
    #     doc='limit peak load during high load time frames'
    # )


    return m




