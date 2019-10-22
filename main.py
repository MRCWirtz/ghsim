import numpy as np
import matplotlib.pyplot as plt
import ghsim
from modules import atmosphere, gases, surface

gas = gases.CarbonDioxide(fraction=0)

# simulation setup
sim = ghsim.Simulation()
sim.set_surface(surface.Surface())
sim.set_radiation(atmosphere.Atmosphere(gas))
sim.run()

t_hist = sim.surf.t_hist
plt.plot(np.arange(len(t_hist)), t_hist, color='red')
plt.xlabel('simulation step', fontsize=16)
plt.ylabel('temperature [K]', fontsize=16)
plt.savefig('img/t_hist.png', bbox_inches='tight')
plt.close()
