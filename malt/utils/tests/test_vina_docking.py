import pytest
import sys

try:
    import vina
except ImportError:
    pass

def test_import():
    import malt
    from malt.utils import docking

@pytest.mark.skipif('vina' not in sys.modules, reason="requires vina library")
def test_caffeine_into_1x8y():
    import malt
    from malt.utils import docking
    score = docking.vina_docking(
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "1X8Y",
    )
    assert score > 0
