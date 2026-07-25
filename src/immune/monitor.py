import torch


class SelfMonitor:
    def __init__(self, hopfield_net, energy_threshold: float = 3.0):
        self.hopfield = hopfield_net
        self.energy_threshold = energy_threshold
        self.calibrated = False
        self.energy_mean: float = 0.0
        self.energy_std: float = 1.0

    def calibrate(self, activations: list[torch.Tensor]):
        energies = [self.hopfield.energy(a.flatten()).item() for a in activations]
        self.energy_mean = sum(energies) / len(energies)
        self.energy_std = (sum((e - self.energy_mean) ** 2 for e in energies) / len(energies)) ** 0.5
        self.energy_std = max(self.energy_std, 1e-8)
        self.calibrated = True

    def score(self, activation: torch.Tensor) -> dict:
        energy = self.hopfield.energy(activation.flatten()).item()
        if not self.calibrated:
            return {"energy": energy, "energy_z": 0.0, "is_anomaly": False}
        energy_z = (energy - self.energy_mean) / self.energy_std
        return {"energy": energy, "energy_z": energy_z,
                "is_anomaly": energy_z > self.energy_threshold}
