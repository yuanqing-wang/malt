import pytest

def test_import():
    from malt import trainer

def test_get_trainer():
    import malt
    from typing import Callable
    trainer = malt.trainer.get_default_trainer()
    assert isinstance(trainer, Callable)
