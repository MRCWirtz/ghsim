import numpy as np
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def sigmoid(x, scale, steep, loc):
    """ sigmoid function """
    return scale/(1+np.exp(-steep * (x - loc)))


def sigmoid_mixture_model(scales, steeps, locs):
    """ Returns a sigmoid mixture model """
    def model(x):
        y = np.zeros(x.shape)
        for scale, steep, loc in zip(scales, steeps, locs):
            y += sigmoid(x, scale, steep, loc)
        return y
    return model


def sigmoid_mixture_model_wrap(x, *pars):
    n = int(len(pars) / 3)
    scales = [pars[i] for i in range(n)]
    steeps = [pars[n+i] for i in range(n)]
    locs = [pars[2*n+i] for i in range(n)]
    return sigmoid_mixture_model(scales, steeps, locs)(x)


def fit_sigmoid_mixture_model(x, y, n=4):
    """ Fits and returns the sigmoid mixture model """
    locs_init = np.min(x) + (np.max(x) - np.min(x)) * (np.arange(n) + 0.5) / n
    scales_init = (np.max(y) - np.min(y)) * np.ones(n) / n
    steeps_init = 10 * np.ones(n) / ((np.max(x) - np.min(x)) / n)
    pars = np.append(np.append(locs_init, scales_init), steeps_init)
    p_opt, _ = curve_fit(sigmoid_mixture_model_wrap, x, y, p0=list(pars), method='trf')
    scales = [p_opt[i] for i in range(n)]
    steeps = [p_opt[n+i] for i in range(n)]
    locs = [p_opt[2*n+i] for i in range(n)]
    return sigmoid_mixture_model(scales, steeps, locs)


# data can be found at:
# https://hitran.org/hitemp/

# cross section for wavelength between 625/cm and 750/cm
data_625_750 = np.genfromtxt('02_625-750_HITEMP2010.par', usecols=(1, 2))
wavelength = data_625_750[:, 0]    # in 1/cm
cross_section = data_625_750[:, 1]  # in cm^2

plt.plot(wavelength, cross_section, color='k')
plt.xlabel('wave number [1 / cm]', fontsize=16)
plt.ylabel('cross section [cm^2]', fontsize=16)
plt.savefig('c02_wavelength_625_750.png', bbox_inches='tight')
plt.close()

# cross section for wavelength between 2250/cm and 2500/cm
data_2250_2500 = np.genfromtxt('02_2250-2500_HITEMP2010.par', usecols=(1, 2))
wavelength = data_2250_2500[:, 0]    # in 1/cm
cross_section = data_2250_2500[:, 1]  # in cm^2

plt.plot(wavelength, cross_section, color='k')
plt.xlabel('wave number [1 / cm]', fontsize=16)
plt.ylabel('cross section [cm^2]', fontsize=16)
plt.savefig('c02_wavelength_2250_2500.png', bbox_inches='tight')
plt.close()

# plot all cross sections in the relevant regime
wavelength_all = np.array([])
cross_section_all = np.array([])
for _file in glob.glob('02_*.par'):
    data = np.genfromtxt(_file, usecols=(1, 2))
    wavelength_all = np.append(wavelength_all, data[:, 0])
    cross_section_all = np.append(cross_section_all, data[:, 1])

_sort = np.argsort(wavelength_all)

plt.plot(wavelength_all[_sort], cross_section_all[_sort], color='k')
plt.xlabel('wave number [1 / cm]', fontsize=16)
plt.ylabel('cross section [cm^2]', fontsize=16)
plt.savefig('c02_wavelength_all.pdf', bbox_inches='tight')
plt.close()

# absorption height for vertical upward going rays
h0 = 1e4
co2_fraction = 0.0004
n_density = 2.504e25
attenuation = cross_section_all * 1e-4 * co2_fraction * n_density
block = 1 - np.exp(-attenuation*h0)
cut = attenuation[_sort] >= 1 / (100 * h0)
wave_numbers = wavelength_all[_sort]
wn_mid = (wave_numbers[1:] + wave_numbers[:-1]) / 2.
wn_left = np.append(wn_mid[0]-2*(wn_mid[0]-wave_numbers[0]), wn_mid)[cut]
wn_right = np.append(wn_mid, wn_mid[-1]+2*(wave_numbers[-1]-wn_mid[-1]))[cut]
cross_sections = cross_section_all[_sort][cut]
print('total wavelength in database: ', len(wavelength_all))
print('subset with cut 1/(100*h0): ', len(wavelength_all[cut]))
plt.plot(wavelength_all[_sort], block[_sort], color='grey')
plt.plot(wave_numbers[cut], block[_sort][cut], color='red')
plt.xlabel('wave number [1 / cm]', fontsize=16)
plt.ylabel('fraction absorped', fontsize=16)
plt.savefig('c02_absorped.pdf', bbox_inches='tight')
plt.close()

wn_interpolate, indices = np.unique(wavelength_all[_sort], return_index=True)
interp_indices = interp1d(wn_interpolate, np.cumsum(cut)[indices]-1, kind='nearest',
                          assume_sorted=True, bounds_error=False)
plt.plot(wavelength_all[_sort], np.cumsum(cut)-1, color='k')
_wn = np.linspace(np.min(wavelength_all), np.max(wavelength_all), 100000)
plt.plot(_wn, interp_indices(_wn), color='red', alpha=0.3)
plt.xlabel('wave number [1 / cm]', fontsize=16)
plt.ylabel('indice to take', fontsize=16)
plt.savefig('c02_wave_number_indice.pdf', bbox_inches='tight')
plt.close()
np.savez('co2_cross_section.npz', wave_numbers_left=wn_left, wave_numbers_right=wn_right,
         cross_sections=cross_sections, interp_indices=interp_indices)


# _model = fit_sigmoid_mixture_model(wn_interpolate, np.cumsum(cut)[indices]-1, n=3)
# _wn = np.linspace(np.min(wavelength_all), np.max(wavelength_all), 100000)
# print(_model)
# print(_model(600 * np.ones(1)))
# print(_model(_wn))
#
# plt.plot(wavelength_all[_sort], np.cumsum(cut)-1, color='k')
# plt.plot(_wn, _model(_wn), color='red', alpha=0.5)
# plt.xlabel('wave number [1 / cm]', fontsize=16)
# plt.ylabel('indice to take', fontsize=16)
# plt.savefig('c02_wave_number_indice_sigmoid_model.pdf', bbox_inches='tight')
# plt.close()


# relevance at 270 K temperature
c = 3e8


def lam2nu(lam):
    """ Wavelength to frequency converter """
    return c / lam


def planck(nu, T, norm=False):
    """
    Planck distribution for black body radiation.

    :param nu: frequency (in Hz)
    :param T: temperature (in K)
    :return: energy density, in J / m^2
    """
    nu = np.atleast_1d(nu)
    a = 2.0 * 6.62607e-34 * nu**3 / (3e8)**2
    b = 6.62607e-34 * nu / 1.38065e-23 / T
    intensity = a / (np.exp(b) - 1.)
    if norm:
        intensity /= np.sum(intensity)
    return intensity


nu = lam2nu(1 / wavelength_all[_sort] * 1e-2)  # frequency in Hz
plt.plot(wavelength_all[_sort], cross_section_all[_sort] * planck(nu, 270) * nu)
plt.xlabel('wave number [1 / cm]', fontsize=16)
plt.ylabel('relevance', fontsize=16)
plt.savefig('c02_relevance_all.pdf', bbox_inches='tight')
plt.close()
