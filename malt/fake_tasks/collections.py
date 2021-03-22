# =============================================================================
# IMPORTS
# =============================================================================
from ..point import Point
from ..data.dataset import Dataset
from .fake_agents import FakeAssayer, FakeVendor


def count_carbons():
    dataset = Dataset(
        [Point("C" * idx) for idx in range(1, 100)]
    )

    import copy
    annotated_dataset = copy.deepcopy(dataset)
    for point in annotated_dataset:
        point.y = len(point.smiles)
    fake_assayer = FakeAssayer(dataset=annotated_dataset)
    fake_vendor = FakeVendor(dataset=dataset)
    return fake_vendor, fake_assayer
