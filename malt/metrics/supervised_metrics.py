import torch
import abc
import malt

class SupervisedMetrics(abc.ABC):
    base_metric = None
    def __init__(self):
        super().__init__()

    def __call__(self, model, ds_te):
        ds_te_loader = ds_te.view(batch_size=len(ds_te))
        g, y = next(iter(ds_te_loader))
        y_hat = model(g).mean
        return getattr(malt.metrics.base_metrics, self.base_metric)(y_hat, y)

class MSE(SupervisedMetrics):
    base_metric = "mse"

class MAPE(SupervisedMetrics):
    base_metric = "mape"

class RMSE(SupervisedMetrics):
    base_metric = "rmse"

class R2(SupervisedMetrics):
    base_metric = "r2"
