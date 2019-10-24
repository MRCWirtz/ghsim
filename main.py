import numpy as np
import os
import matplotlib.pyplot as plt
import ghsim
from modules import atmosphere, gases, surface

carbon = gases.CarbonDioxide(fraction=0.0004)

# simulation setup
sim = ghsim.Simulation()
sim.set_surface(surface.Surface())
sim.set_radiation(atmosphere.Atmosphere(carbon))
sim.run()

t_hist = sim.surf.t_hist
plt.plot(np.arange(len(t_hist)), t_hist, color='red')
plt.xlabel('simulation step', fontsize=16)
plt.ylabel('temperature [K]', fontsize=16)
os.makedirs('img', exist_ok=True)
plt.savefig('img/t_hist_co2.png', bbox_inches='tight')
plt.close()
