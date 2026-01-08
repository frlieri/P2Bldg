import pytest
import pandas as pd
from pyomo.environ import ConcreteModel
from src.model.pyomo_model import init_pyomo_model, check_components
from src.model.components import (
    ElNode, ElDemand, ElImport, ElExport,
    ElectricLine
)
from src.const import Season
from src.helper import create_ts_multiindex

class TestPyomoModel:

    @pytest.fixture
    def basic_components(self):
        # Create a minimal set of components for optimization
        # 1. Demand (Sink)
        demand_ts = pd.Series(10.0, index=create_ts_multiindex())
        demand = ElDemand(name="load", p_t=demand_ts)
        
        # 2. Import (Source)
        imp = ElImport(name="grid_imp", var_cost_t=0.30, p_max=100.0)
        
        # 3. Export (Sink)
        exp = ElExport(name="grid_exp", var_cost_t=-0.08)
        
        # 4. Node to connect them
        node = ElNode(name="main_bus", sources=[imp], sinks=[demand, exp])
        
        return {
            'nodes': {'main_bus': node},
            'demand': {'load': demand},
            'import': {'grid_imp': imp},
            'export': {'grid_exp': exp},
            'pv': {},
            'heatpump': {},
            'batteries': {},
            'buffer_storages': {},
            'dhw_storages': {},
            'th_inertia': {},
            'solar_thermal': {},
            'deh': {},
            'boiler': {},
            'refurbishments': {},
            'links': {}
        }

    def test_check_components(self, basic_components):
        """Verifies that the component validator runs without error on valid inputs."""
        try:
            check_components(basic_components)
        except Exception as e:
            pytest.fail(f"check_components raised {e} on valid input")

    def test_init_pyomo_model_structure(self, basic_components):
        """Tests if the model initializes with correct sets and variables."""
        cost_weights = pd.Series(1.0, index=create_ts_multiindex())
        co2_price = 0.0
        
        m = init_pyomo_model(basic_components, cost_weights, co2_price)
        
        assert isinstance(m, ConcreteModel)
        # Check sets
        assert hasattr(m, 't')
        assert hasattr(m, 'season')
        assert Season.Winter in m.season
        
        # Check variables
        assert hasattr(m, 'p_el_t_feed')
        assert hasattr(m, 'p_el_t_drain')
        
        # Check if objective is created
        assert hasattr(m, 'objective_function')


    def test_link_handling(self, basic_components):
        """Tests if links (ElectricLines) are correctly added to the model variables."""
        line = ElectricLine("Link1", p_max=50.0, eff=0.99)
        basic_components['links']['Link1'] = line

        basic_components['nodes']['2nd_bus'] = ElNode(name="main_bus", sources=[basic_components['import']['grid_imp']], sinks=[line])
        basic_components['nodes']['main_bus'].sources.append(line)
        
        cost_weights = pd.Series(1.0, index=create_ts_multiindex())
        m = init_pyomo_model(basic_components, cost_weights, co2_price=0.0)
        
        assert line in m.links
