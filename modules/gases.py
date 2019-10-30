import numpy as np
import os
from modules.physics import c


def nu2wn(nu):
    return 0.01 * nu / c


class Gas:

    def __init__(self, element):
        path = os.path.dirname(os.path.realpath(__file__))
        path += '/data/%s_cross_section.npz' % element
        self.data = np.load(path, allow_pickle=True)
        self.interp = self.data['interp_indices'][()]
        self.wn_left, self.wn_right = self.data['wave_numbers_left'], self.data['wave_numbers_right']
        self.cross_section = self.data['cross_sections']  # in cm^2

    def get_attenuation(self, nu):
        wn = nu2wn(nu)
        cross_section = np.zeros(wn.shape)
        if self.fraction > 0:
            indices = np.rint(self.interp(wn)).astype(int)
            indices[indices < 0] = 0
            mask = (wn >= self.wn_left[indices]) & (wn < self.wn_right[indices])
            cross_section[mask] = self.cross_section[indices[mask]]
        return cross_section * self.fraction    # in cm^2


class CarbonDioxide(Gas):

    def __init__(self, fraction=0.0004):
        super(CarbonDioxide, self).__init__('co2')
        self.fraction = fraction


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from modules.physics import rand_planck_frequency

    wn_all = np.linspace(1, 2500, 1000000)
    nu_all = c * 1e2 * wn_all
    gas = CarbonDioxide(fraction=1)
    cross_sections = gas.get_attenuation(nu_all)
    nus = rand_planck_frequency(255, 1000000)
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('wave number [1 / cm]', fontsize=16)
    ax1.set_ylabel('cross section [cm^2]', fontsize=16, color='tab:red')
    ax1.plot(wn_all, cross_sections, color='red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('frequency', color='k')
    ax2.hist(nu2wn(nus), 100, histtype='step', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_yscale('log')
    plt.show()

    # show as a function of the cross section which fraction of upgoing
    # rays are blocked in the atsmosphere
    cross_sections = np.logspace(-30, -18, 100000)  # in cm^2
    number_density = 2.504e25    # molecules per m^3
    gas = CarbonDioxide(fraction=0.0004)
    attenuation = cross_sections * 1e-4 * gas.fraction * number_density
    block = 1 - np.exp(-attenuation * 1e4)
    plt.plot(cross_sections, block, color='k')
    for frac in [0.1, 0.01, 0.001]:
        plt.axhline(frac, color='red', linestyle='dotted')
        idx = np.argmin(np.abs(block - frac))
        plt.axvline(cross_sections[idx], color='red', linestyle='dotted')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('cross section [cm^2]', fontsize=14)
    plt.ylabel('fraction absorped', fontsize=14)
    plt.show()

    attenuation = number_density * (1e-4 * gas.get_attenuation(nus)) * 1e3
    block = 1 - np.exp(-attenuation * 1e4)
    plt.hist(block, bins=100, histtype='step', color='k')
    plt.yscale('log')
    plt.xlabel('blocked', fontsize=14)
    plt.ylabel('entries', fontsize=14)
    plt.show()
