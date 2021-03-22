import pytest


def test_import():
    from malt import fake_tasks


def test_construct():
    import malt

    assayer, vendor = malt.fake_tasks.collections.count_carbons()
