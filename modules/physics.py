import numpy as np


# some physics constants
h = 6.62607e-34    # planck constant, in m^2 * kg / s
c = 2.99792e+8     # speed of light, in m / s
k = 1.38065e-23    # boltzmann constant, in J / K
s = 5.67037e-8     # stefan boltzmann constant, in W / m^2 / K^4
b = 2.8214391 * k / h   # Wien's displacement constant, in Hz / K


def cel2kel(cel):
    """ Celsius to celvin converter """
    return cel + 273.15


def kel2cel(kel):
    """ Kelvin to celsius converter """
    return kel - 273.15


def lam2nu(lam):
    """ Wavelength to frequency converter """
    return c / lam


def nu2lam(nu):
    """ Frequency to wavelength converter """
    return c / nu


def wien_displace(T):
    """
    Wien's displacement law.abs

    :param T: temperature (in K)
    :return: frequency where planck distribution peaks (in Hz)
    """
    return b * T


def wien_interval(T):
    """ Returns frequency interval for integration """
    vmax = wien_displace(T)
    return (vmax / 300, vmax * 10)


def planck(nu, T, norm=False):
    """
    Planck distribution for black body radiation.

    :param nu: frequency (in Hz)
    :param T: temperature (in K)
    :return: energy density, in J / m^2
    """
    nu = np.atleast_1d(nu)
    a = 2.0 * h * nu**3 / c**2
    b = h * nu / k / T
    intensity = a / (np.exp(b) - 1.)
    if norm:
        intensity /= np.sum(intensity)
    return intensity


def rand_planck_frequency(T, n=1, sample=1000):
    """
    Samples random frequency from planck dsitribution.
    """
    nu_min, nu_max = wien_interval(T)
    nus = np.linspace(np.log10(nu_min), np.log10(nu_max), sample)
    density = planck(10**nus, T) * 10**nus
    idx = np.random.choice(np.arange(sample), n, p=density/density.sum())
    return 10**(nus[idx] + (nus[1] - nus[0]) * np.random.uniform(-0.5, 0.5, n))


def stefan_boltzmann(A, T, emissivity=1):
    """
    Stefan boltzmann law
    https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law

    :param A: area of surface, in m^2
    :param T: temperature of black body, in K
    :param emissivity: emissivity
    :return: total emitted power, in W
    """
    return emissivity * s * A * T**4


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    T = cel2kel(15)
    nu_min, nu_max = wien_interval(T)
    nu = np.linspace(nu_min, nu_max, 1000)
    lambdas = nu2lam(nu)
    plt.plot(lambdas, planck(nu, T=T) / lambdas)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$ [m]', fontsize=14)
    plt.ylabel(r'density', fontsize=14)
    plt.show()

    n = 100000
    plt.plot(nu, n * planck(nu, T=T, norm=True))
    rand_planck = rand_planck_frequency(T, n=n)
    bins = np.logspace(np.log10(0.9*np.min(rand_planck)), np.log10(1.1*np.max(rand_planck)), 100)
    plt.hist(rand_planck, bins=bins, histtype='step', color='grey')
    plt.axvline(wien_displace(T), color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\nu$ [Hz]', fontsize=14)
    plt.ylabel(r'density', fontsize=14)
    plt.show()
