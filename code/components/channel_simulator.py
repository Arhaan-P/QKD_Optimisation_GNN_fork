import numpy as np
from scipy.constants import h, c

def h2(x):
    return -x * np.log2(x) - (1-x) * np.log2(1-x) if 0 < x < 1 else 0

class AdvancedQuantumChannelSimulator:
    def __init__(self, distance, wavelength=1550e-9, fiber_loss=0.2,
                 detector_efficiency=0.1, dark_count_rate=1e-6,
                 atmospheric_visibility=None, mean_photon_number = 0.1,
                 num_pulses = 10000):
        self.distance = distance
        self.wavelength = wavelength
        self.fiber_loss = fiber_loss
        self.detector_efficiency = detector_efficiency
        self.dark_count_rate = dark_count_rate
        self.atmospheric_visibility = atmospheric_visibility
        self.photon_energy = h * c / wavelength
        self.mean_photon_number = mean_photon_number
        self.num_pulses = num_pulses

    def calculate_channel_loss(self):
        fiber_loss_db = self.fiber_loss * self.distance
        fiber_transmission = 10 ** (-fiber_loss_db/10)

        if self.atmospheric_visibility:
            beam_divergence = 1.22 * self.wavelength / 0.1
            geometric_loss = (0.1 / (beam_divergence * self.distance)) ** 2
            atmospheric_loss = np.exp(-3.91 * self.distance / self.atmospheric_visibility)
            total_transmission = fiber_transmission * geometric_loss * atmospheric_loss
        else:
            total_transmission = fiber_transmission

        return total_transmission

    def simulate_bb84_protocol(self):
        channel_transmission = self.calculate_channel_loss()
        received_photons = np.random.poisson(
            self.mean_photon_number * channel_transmission * self.detector_efficiency,
            self.num_pulses
        )
        dark_counts = np.random.poisson(self.dark_count_rate, self.num_pulses)
        total_counts = received_photons + dark_counts
        basis_matches = np.random.choice([0, 1], self.num_pulses, p=[0.5, 0.5])
        qber = 0.5 * (1 - np.exp(-2 * self.distance / 100))
        errors = np.random.choice([0, 1], self.num_pulses, p=[1-qber, qber])
        matched_pulses = total_counts * basis_matches
        raw_key_rate = np.sum(matched_pulses) / self.num_pulses
        final_key_rate = raw_key_rate * (1 - 2 * h2(qber))

        return {
            'qber': qber,
            'raw_key_rate': raw_key_rate,
            'final_key_rate': final_key_rate,
            'channel_loss_db': -10 * np.log10(channel_transmission),
            'dark_count_probability': np.mean(dark_counts > 0)
        }