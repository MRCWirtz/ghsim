import numpy as np


me = 5.97e24    # mass earth, in kg
re = 6371    # radius earth, in km
c_heat = 1  # specific heat capacity, in J/(kg·K)
solar_constant = 1367   # W / m^2
photon_multiplier = 1e40    # batches of photons emitted at the same time
photon_constant = 1.072e15 / photon_multiplier

time_acceleration = 10. * (365 * 24 * 3600)    # 1 corresponds to a second


class Surface:

    def __init__(self, t=0, albedo=0.306):
        self.t = t
        self.t_hist = []
        self.delta_t = None
        self.albedo = albedo
        self.re = re
        self.cross_area = np.pi * (1e3 * self.re)**2
        self.surface_area = 4 * np.pi * (1e3 * self.re)**2

    def _external_heat(self):
        self.delta_t = (1 - self.albedo) * solar_constant * time_acceleration * self.cross_area / c_heat / me

    def _radiate(self, rad):
        n_emit_mean = time_acceleration * self.surface_area * photon_constant * self.t**3
        n_emit = np.random.poisson(n_emit_mean)
        energy_emission = photon_multiplier * rad.radiate(self.t, n_emit)   # total energy loss in J
        self.delta_t -= energy_emission / c_heat / me

    def _change_temperature(self):
        if self.delta_t is None:
            return True
        self.t += self.delta_t
        self.t_hist.append(self.t)
        return np.abs(self.delta_t) > 0.01
