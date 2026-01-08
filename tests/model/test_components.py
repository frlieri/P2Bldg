import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from src.model.components import (
    PV, SolarPotential, Heatpump, ElStorage, ElDemand,
    ElNode, TempLevel, FlowTempLevel, ElectricLine
)
from src.helper import create_ts_multiindex

class TestComponents:

    @pytest.fixture
    def mock_solar_potential(self):
        potential = MagicMock(spec=SolarPotential)
        potential.name = "MockRoof"
        # 8760 hours of normalized generation
        potential.gen_ts_normalized = pd.Series(np.random.uniform(0, 1, 8760))
        return potential

    def test_pv_initialization(self, mock_solar_potential):
        pv = PV(name="PV1", potential=mock_solar_potential, p_inst_max=10.0, cap_cost_p=1200)
        assert pv.name == "PV1"
        assert pv.potential.name == "MockRoof"
        assert pv.p_inst_max == 10.0

    def test_el_storage_initialization(self):
        batt = ElStorage(name="Battery1", c_inst_max=10.0, cap_cost_c=500, eff_charge=0.95)
        assert batt.name == "Battery1"
        assert batt.eff_charge == 0.95
        assert batt.c_inst_max == 10.0

    def test_heatpump_cop_calculation(self):
        # Mock HP specs table
        # Columns (temperatures)
        columns = [-20, -15, -10, -7, 2, 7, 10, 20, 30, 35]
        # Data
        data = [
            [2.3, 2.5, 2.8, 3.0, 4.0, 5.2, 5.2, 8.2, 8.7, 8.7],  # COP 35
            [2.0, 2.2, 2.4, 2.6, 3.3, 4.1, 4.8, 5.4, 8.5, 8.5],  # Qmax 45
        ]
        # MultiIndex for rows
        index = pd.MultiIndex.from_tuples(
            [('COP', 35), ('Qmax', 35)],
            names=['Metric', 'Level']
        )
        # Create DataFrame
        specs = pd.DataFrame(data, index=index, columns=columns)
        
        sink_temp = TempLevel("sink", 35)
        source_temp = TempLevel("source", pd.Series(7, index=create_ts_multiindex()))
        
        hp = Heatpump(
            name="HP1", 
            temp_levels_feed=[sink_temp], 
            temp_levels_drain=[source_temp],
            specs_table=specs,
            cap_cost_p=1000
        )
        
        # Check if specs were processed
        assert hp.name == "HP1"
        assert ('COP', 35) in hp.specs_table.index

    def test_nodes_and_links(self):
        # Create an electric node with a demand and a source
        demand_ts = pd.Series([1.0] * 8760)
        el_demand = ElDemand("HouseLoad", demand_ts)
        
        pv_source = MagicMock()
        pv_source.name = "PV_Comp"
        
        node = ElNode(name="ElMain", sources=[pv_source], sinks=[el_demand])
        
        assert "PV_Comp" in [src.name for src in node.sources]
        assert "HouseLoad" in [sink.name for sink in node.sinks]

    def test_temp_levels(self):
        # Test basic temp level
        ground = TempLevel("ground", 10.0)
        assert ground.temp_t == 10.0
        
        # Test flow temp level with heating curve
        # Normally takes air temp timeseries
        air_ts = pd.Series([-10]*24 + [0]*24 + [10]*24)
        flow = FlowTempLevel(
            "HeatingFlow", 
            flow_temp=45,
            return_temp=28, 
            apply_heating_curve=True,
            temp_air_t=air_ts,
            outtemp_min_ref=-14
        )
        
        assert hasattr(flow, 'temp_t')
        # At very low air temp, flow temp should be higher/at ref
        assert flow.temp_t.iloc[0] > flow.temp_t.iloc[2*24]

    def test_electric_line(self):
        line = ElectricLine("GridToBattery", p_max=5.0, eff=0.98)
        assert line.eff == 0.98
        assert line.p_max == 5.0
