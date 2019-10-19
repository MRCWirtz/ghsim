import unittest

import numpy as np
from modules import physics


class TestEnergy(unittest.TestCase):

    def test_01_planck_boltzmann(self):
        for T in [10, 50, 100, 200, 500, 5000]:
            nu = np.linspace(*physics.wien_interval(T), 1000)
            dnu = nu[1] - nu[0]
            planck = physics.planck(nu, T, norm=False)
            # factor of pi by integration over lamberts distributed vectors (dI ~ cos(theta))
            planck = np.pi * dnu * np.sum(planck)
            stefan_b = physics.stefan_boltzmann(1, T)
            rel_diff = (planck - stefan_b) / stefan_b
            self.assertTrue(np.abs(rel_diff) < 1e-3)

    def test_02_wien_law(self):
        # in the limit for high temperatures the maximum occurence of wavelength
        # should be given by wien's law
        for T in [10, 50, 100, 200, 500, 5000]:
            nu_low, nu_high = physics.wien_interval(T)
            nus = np.linspace(nu_low, nu_high, 10000)
            planck = physics.planck(nus, T)
            rel_diff = (physics.wien_displace(T) - nus[np.argmax(planck)]) / physics.wien_displace(T)
            self.assertTrue(np.abs(rel_diff) < 1e-2)

    def test_03_number_photons(self):
        # number of emitted photons increase with T**3
        all_T = np.linspace(1, 500, 100)
        n_photons = np.zeros(all_T.size)
        for i, T in enumerate(all_T):
            nus = physics.rand_planck_frequency(T, n=10000)
            E_per_photon = np.mean(physics.h * nus)
            E_sb = physics.stefan_boltzmann(1, T)
            n_photons[i] = E_sb / E_per_photon
        n_t3 = n_photons / all_T**3
        print(np.mean(n_t3))
        # import matplotlib.pyplot as plt
        # plt.plot(all_T, n_photons / 1e15)
        # plt.xscale('log')
        # plt.yscale('log')
        # plt.show()
        rel_diff = np.abs(n_t3 - n_t3[0]) / n_t3[0]
        self.assertTrue((rel_diff < 5e-2).all())


if __name__ == '__main__':
    unittest.main()
