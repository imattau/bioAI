import torch
import snntorch as snn
from snntorch import spikegen


class TestSNN:
    def test_lif_neuron_basic(self):
        lif = snn.Leaky(beta=0.9)
        x = torch.randn(10, 1)
        spk, mem = lif(x)
        assert spk.shape == (10, 1)
        assert mem.shape == (10, 1)

    def test_rate_coding(self):
        data = torch.rand(10, 1) * 5
        spikes = spikegen.rate(data, num_steps=20)
        assert spikes.shape == (20, 10, 1)
