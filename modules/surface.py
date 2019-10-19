import numpy as np


me = 5.97e24    # mass earth, in kg
me = 1e10    # mass earth, in kg
re = 6.371e6    # radius earth, in m
c_heat = 1  # specific heat capacity, in J/(kg·K)
solar_constant = 1367   # W / m^2
photon_multiplier = 1e17    # batches of photons emitted at the same time
photon_constant = 1.072e15 / photon_multiplier


class Surface:

    def __init__(self, t=0, albedo=0.306, re=6371):
        self.t = t
        self.delta_t = None
        self.albedo = albedo
        self.re = re
        self.cross_area = np.pi * self.re**2
        self.surface_area = 4 * np.pi * self.re**2

    def _external_heat(self):
        self.delta_t = (1 - self.albedo) * solar_constant * self.cross_area / c_heat / me

    def _radiate(self, rad):
        n_emit = np.random.poisson(photon_constant * self.t**3)
        energy_emission = photon_multiplier * rad.radiate(self.t, n_emit)
        self.delta_t -= energy_emission * self.surface_area / c_heat / me

    def _change_temperature(self):
        if self.delta_t is None:
            return True
        self.t += self.delta_t
        return np.abs(self.delta_t) > 0.001
