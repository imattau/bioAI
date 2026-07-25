def test_imports():
    import torch
    import numpy
    import scipy
    import sklearn
    import pandas
    import networkx
    import torchhd
    import snntorch
    import gymnasium
    import stable_baselines3


def test_cuda():
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
