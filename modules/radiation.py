import numpy as np
from modules import atmosphere, coord, physics


class Rays:

    def __init__(self, atm, re, n, hmax):

        self.nmax = n
        self.n_active = 0
        self.rmax = re + hmax
        self.re = re
        self.atm = atm
        self.x = re * coord.rand_vec(n)     # position vector on Earth surface
        self.radius = np.sqrt(np.sum(self.x**2, axis=0))
        self.p, self.theta = coord.rand_vec_on_surface(self.x)     # rotate to respective position on sphere
        self.nu = np.zeros(n)
        self.active = np.zeros(n).astype(bool)
        self.propagate = np.zeros(n).astype(bool)
        self.collect_leave = np.zeros(n).astype(bool)

    def radiate(self, T, ns):
        e_loss = 0
        ntot = ns
        self.absorbed = 0.
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
                self.radius = np.sqrt(np.sum(self.x**2, axis=0))
                e_loss += self._leave_atmosphere()
                self._absorb_surface()
                self._interact_atmosphere()
        print('Batches absorbed: %i (fraction of emitted: %.5f)' % (self.absorbed, self.absorbed / max(1, ntot)))
        return e_loss

    def _propagate(self):
        hs = self.radius[self.active] - self.re
        mask_prop, distance = self.atm.get_distance(self.nu[self.active], np.pi/2-self.theta[self.active], hs)
        self.propagate[self.active] = mask_prop
        self.x[:, self.propagate] += distance * self.p[:, self.propagate]
        assert self.collect_leave.sum() == 0, "collect_leave mask is not clean"
        self.collect_leave[self.active] = ~mask_prop

    def _leave_atmosphere(self):
        # check for leaving rays
        self.n_active -= self.collect_leave.sum()
        energy_leave = physics.h * np.sum(self.nu[self.collect_leave])  # in Joule
        self.active[self.collect_leave] = False
        self.collect_leave[self.collect_leave] = False
        self.propagate[self.collect_leave] = False
        return energy_leave

    def _absorb_surface(self):
        # check for absorbed rays
        mask_absorb = (self.radius < self.re) & self.active
        self.absorbed += mask_absorb.sum()
        self.n_active -= mask_absorb.sum()
        self.active[mask_absorb] = False
        self.propagate[mask_absorb] = False

    def _interact_atmosphere(self):
        # check for interaction in atmosphere
        if self.n_active > 0:
            phi, theta = coord.rand_phi(self.n_active), coord.rand_theta(self.n_active)
            self.theta[self.active] = theta
            self.p[:, self.active] = coord.ang2vec(phi, theta)
