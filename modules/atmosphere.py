import numpy as np

h0 = 1e4    # scale height for atmosphere, in m
mass_density = 1.2041  # kg/m^3
number_density = 2.504e25    # molecules per m^3


def get_propagation_distance(attenuation, theta, hs):
    # mask for rays that are propagated
    mask = (attenuation > 0)
    alpha = attenuation * h0 / np.cos(theta)
    u = np.random.random(theta.size)
    inner_log = np.zeros(theta.size)
    # check for rays that have interaction in atmosphere
    inner_log[mask] = 1 + np.log(1-u[mask]) / alpha[mask]
    # if the argument of the logarithm becomes smaller zero, the ray left the atmosphere
    mask = mask & (inner_log > 0)
    distance = -1/np.cos(theta[mask]) * (h0 * np.log(inner_log[inner_log > 0]) + hs[mask])
    return mask, distance


class Atmosphere:

    def __init__(self, gases):
        if not hasattr(gases, '__iter__'):
            gases = [gases]
        self.gases = gases

    def get_distance(self, nu, theta, hs):
        attenuation = np.zeros(nu.shape)
        for gas in self.gases:
            attenuation += number_density * gas.get_attenuation(nu)
        return get_propagation_distance(attenuation, theta, hs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # show as a function of the cross section which fraction of upgoing
    # rays are blocked in the atsmosphere
    cross_sections = np.logspace(-30, -18, 100000)  # in cm^2
    attenuation = cross_sections * 1e-4 * number_density * 0.0004
    block = 1 - np.exp(-attenuation*h0)
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
