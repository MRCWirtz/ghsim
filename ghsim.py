from modules import radiation


class Simulation:

    def __init__(self):
        self.surf = None
        self.rad = None

    def set_surface(self, surf):
        self.surf = surf

    def set_radiation(self, atm, n=1000000, rmax=6500):
        self.atm = atm
        self.rad = radiation.Rays(atm, self.surf.re, n=n, rmax=rmax)

    def run(self):
        while (self.surf._change_temperature()):
            self.surf._external_heat()
            self.surf._radiate(self.rad)
            print('Temperature: %.1f K' % self.surf.t)
