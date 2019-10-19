import ghsim
from modules import atmosphere, gases, surface

gas = gases.CarbonDioxide(fraction=0)

# simulation setup
sim = ghsim.Simulation()
sim.set_surface(surface.Surface())
sim.set_radiation(atmosphere.Atmosphere(gas))
sim.run()
