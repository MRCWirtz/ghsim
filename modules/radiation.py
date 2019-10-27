import numpy as np
from modules import coord, physics


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
                # print('n_active: ', self.n_active)
                # propagation step
                self._propagate()
                # check new state of rays
                self.radius = np.sqrt(np.sum(self.x**2, axis=0))
                self._interact_atmosphere()
                e_loss += self._leave_atmosphere()
                self._absorb_surface()
        print('Batches absorbed: %i (fraction of emitted: %.5f)' % (self.absorbed, self.absorbed / max(1, ntot)))
        return e_loss

    def _propagate(self):
        hs = self.radius[self.active] - self.re
        mask_prop, distance = self.atm.get_distance(self.nu[self.active], np.pi/2-self.theta[self.active], hs)
        change = np.copy(self.active)
        change[change] = mask_prop
        self.x[:, change] += distance * self.p[:, change]
        collect = np.copy(self.active)
        collect[collect] = ~mask_prop
        self.collect_leave[collect] = True

    def _leave_atmosphere(self):
        # check for leaving rays
        mask_leave = ((self.radius >= self.rmax) | self.collect_leave) & self.active
        self.n_active -= np.sum(mask_leave)
        self.active[mask_leave] = False
        self.collect_leave[mask_leave] = False
        return physics.h * np.sum(self.nu[mask_leave])  # in Joule

    def _absorb_surface(self):
        # check for absorbed rays
        mask_absorb = (self.radius < self.re) & self.active
        self.absorbed += np.sum(mask_absorb)
        self.n_active -= np.sum(mask_absorb)
        self.active[mask_absorb] = False

    def _interact_atmosphere(self):
        # check for interaction in atmosphere
        mask_interaction = (self.radius >= self.re) & (self.radius < self.rmax) & self.active
        n_interact = np.sum(mask_interaction)
        if n_interact > 0:
            phi, theta = coord.rand_phi(n_interact), coord.rand_theta(n_interact)
            self.theta[mask_interaction] = theta
            self.p[:, mask_interaction] = coord.ang2vec(phi, theta)
