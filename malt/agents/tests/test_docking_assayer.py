import pytest
import sys

try:
    import vina
except ImportError:
    pass


@pytest.mark.skipif('vina' not in sys.modules, reason="requires vina library")
def test_vina_docking_assayer():
    import malt
    from malt import Point, Dataset
    assayer = malt.agents.assayer.DockingAssayer("1X8Y")
    point = Point("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    dataset = Dataset([point])
    assayer.assay(dataset)
