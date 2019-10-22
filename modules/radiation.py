import numpy as np
from modules import coord, physics


class Rays:

    def __init__(self, atm, re, n, rmax):

        self.nmax = n
        self.n_active = 0
        self.rmax = rmax
        self.re = re
        self.mfp = atm.mfp
        self.x = re * coord.rand_vec(n)     # position vector on Earth surface
        self.p, self.theta = coord.rand_vec_on_surface(self.x)     # rotate to respective position on sphere
        self.nu = np.zeros(n)
        self.active = np.zeros(n).astype(bool)

    def radiate(self, T, ns):
        e_loss = 0
        while ns > 0:
            batch = min(self.nmax - self.n_active, ns)
            avail = ~self.active
            indices = np.random.choice(self.nmax, size=batch, replace=False, p=avail/float(avail.sum()))
            self.x[:, indices] = self.re * coord.rand_vec(batch)
            self.p[:, indices], self.theta[indices] = coord.rand_vec_on_surface(self.x[:, indices], batch)
            self.nu[indices] = physics.rand_planck_frequency(T, batch)
            self.active[indices] = True
            self.n_active += batch
            ns -= batch

            # do update with active rays
            assert self.n_active == np.sum(self.active), "Active mask and numbers differ!"
            while self.n_active > 0:
                # propagation step
                self._propagate()
                # check new state of rays
                radius = np.sqrt(np.sum(self.x**2, axis=0))
                self._interact_atmosphere(radius)
                e_loss += self._leave_atmosphere(radius)
                self._absorb_surface(radius)

        return e_loss

    def _propagate(self):
        distance = np.random.exponential(self.mfp, size=self.n_active)
        self.x[:, self.active] += distance * self.p[:, self.active]

    def _leave_atmosphere(self, radius):
        # check for leaving rays
        mask_leave = (radius >= self.rmax) & self.active
        self.n_active -= np.sum(mask_leave & self.active)
        self.active[mask_leave] = False
        return physics.h * np.sum(self.nu[mask_leave])

    def _absorb_surface(self, radius):
        # check for absorbed rays
        mask_absorb = (radius < self.re) & self.active
        self.n_active -= np.sum(mask_absorb)
        self.active[mask_absorb] = False

    def _interact_atmosphere(self, radius):
        # check for interaction in atmosphere
        mask_interaction = (radius >= self.re) & (radius < self.rmax) & self.active
        n_interact = np.sum(mask_interaction)
        if n_interact > 0:
            self.p[:, mask_interaction] = coord.rand_vec(n_interact)
