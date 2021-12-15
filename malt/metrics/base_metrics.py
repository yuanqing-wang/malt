import torch

def mse(input, target):
    return torch.nn.functional.mse_loss(target, input)

def mape(input, target):
    return ((input - target).abs() / target.abs()).mean()

def rmse(input, target):
    return torch.sqrt(torch.nn.functional.mse_loss(target, input))

def r2(input, target):
    target = target.flatten()
    input = input.flatten()
    ss_tot = (target - target.mean()).pow(2).sum()
    ss_res = (input - target).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)
