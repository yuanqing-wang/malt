import pytest

def test_import():
    import malt
    from malt.utils import docking

def test_caffeine_into_1x8y():
    import malt
    from malt.utils import docking
    score = docking.vina_docking(
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "1X8Y",
    )
