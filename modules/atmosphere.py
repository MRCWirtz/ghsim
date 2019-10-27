import numpy as np

h0 = 10    # scale height for atmosphere, in km
mass_density = 1.2041  # kg/m^3
number_density = 2.504e25    # molecules per m^3


def get_propagation_distance(attenuation, theta, hs, random=True):
    # mask for rays that are propagated
    mask = (attenuation > 0)
    alpha = np.cos(theta[mask]) / attenuation[mask] / h0 / np.exp(-hs[mask] / h0)
    u = np.random.random(np.sum(mask)) if random else np.ones(np.sum(mask)) * 0.5
    inner_log = -np.ones(theta.size)
    # check for rays that have interaction in atmosphere
    inner_log[mask] = 1 + alpha * np.log(1-u)
    # if the argument of the logarithm becomes smaller zero, the ray left the atmosphere
    mask = mask & (inner_log > 0)
    distance = -h0/np.cos(theta[mask]) * np.log(inner_log[mask])
    return mask, distance


class Atmosphere:

    def __init__(self, gases):
        if not hasattr(gases, '__iter__'):
            gases = [gases]
        self.gases = gases

    def get_distance(self, nu, theta, hs):
        attenuation = np.zeros(nu.shape)
        for gas in self.gases:
            attenuation += number_density * (1e-4 * gas.get_attenuation(nu)) * 1e3  # in (1 / km)
        return get_propagation_distance(attenuation, theta, hs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 10000
    attenuation = np.logspace(-6, 3, n)
    for h in [0, 1, 3, 10]:
        mask, distance = get_propagation_distance(attenuation, np.zeros(n), h * np.ones(n), random=False)
        plt.plot(attenuation[mask], distance, label='hs: %i km' % h)
    plt.legend(fontsize=16)
    plt.xlabel('attenuation [1/km]', fontsize=14)
    plt.ylabel('distance [km]', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    for h in [0, 1, 3, 10]:
        mask, distance = get_propagation_distance(attenuation, 1e-5 + np.pi/2 * np.ones(n), h * np.ones(n), random=False)
        plt.plot(attenuation[mask], distance, label='hs: %i km' % h)
    plt.legend(fontsize=16)
    plt.xlabel('attenuation [1/km]', fontsize=14)
    plt.ylabel('distance [km]', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
