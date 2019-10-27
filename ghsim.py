from modules import radiation


class Simulation:

    def __init__(self):
        self.surf = None
        self.rad = None

    def set_surface(self, surf):
        self.surf = surf

    def set_radiation(self, atm, nbatch=100000, hmax=500):
        """
        :param atm: Atmosphere object
        :param n: number of vectorized calculations in one batch
        :param hmax: Maximum height over sea level, in km
        """
        self.atm = atm
        self.rad = radiation.Rays(atm, self.surf.re, n=nbatch, hmax=hmax)

    def run(self):
        while (self.surf._change_temperature()):
            self.surf._external_heat()
            self.surf._radiate(self.rad)
            print('Temperature: %.2f K' % self.surf.t)
