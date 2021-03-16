# =============================================================================
# IMPORTS
# =============================================================================
from ..point import Point
from ..data.dataset import Dataset
from .fake_agents import FakeAssayer, FakeMerchant


def count_carbons():
    dataset = Dataset(
        [Point("C" * idx) for idx in range(10)]
    )

    import copy
    annotated_dataset = copy.deepcopy(dataset)
    for point in annotated_dataset:
        point.y = len(point.smiles)
    fake_assayer = FakeAssayer(dataset=annotated_dataset)
    fake_merchant = FakeMerchant(dataset=dataset)
    return fake_assayer, fake_merchant
