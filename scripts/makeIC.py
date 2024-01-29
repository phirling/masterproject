import argparse
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pNbody import profiles
from pNbody import ic
from swiftsimio import Writer
import unyt
import h5py

#################################################
# Generate a NFW halo in hydrostatic equilibrium
#
# Star particles are placed at r200 isotropically
#################################################

# Unit System used to generate the ICs (not necessarily code units!)
UnitLength_in_cm         = 3.085678e21    # kpc
UnitMass_in_g            = 1.989e43       # 10^10 Solar Mass
UnitVelocity_in_cm_per_s = 1e5            # km/sec 
UnitTime_in_s            = UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitEnergy_in_cgs        = UnitMass_in_g * UnitVelocity_in_cm_per_s**2

# Same thing using unyt (for swiftsimio)
halo_units = unyt.UnitSystem(
    "halo",
    unyt.kpc,
    unyt.unyt_quantity(1e10, units=unyt.Solar_Mass),
    unyt.unyt_quantity(1.0, units=unyt.s * unyt.kpc / unyt.km).to(unyt.Gyr),
)

# Important constants
H0 = 72.0 * unyt.km / unyt.s / unyt.Mpc
rho_c = 3*H0**2 / (8*np.pi*unyt.G)
rho_c = rho_c.to(10**10*unyt.Solar_Mass/unyt.kpc**3).value
gamma = 5.0 / 3.0
G = (unyt.G.to(unyt.kpc / (1e10*unyt.Solar_Mass) * (unyt.km/ unyt.s)**2)).value

# Compile SWIFT accordingly! for star particles to be ignored by gravity
SWIFT_NO_GRAVITY_BELOW_ID = 1000

# Parse Options
parser = argparse.ArgumentParser("IC for AGORA NFW halo")
parser.add_argument("-o",dest="output",type=str,default="nfw.hdf5",help="output file name")
parser.add_argument("--plot",action="store_true")
parser.add_argument("-logM200",type=float,default=7,help="log10 of Virial mass of halo in Msun")
parser.add_argument("-c",type=float,default=20,help="NFW Concentration of halo")
parser.add_argument("-fb",type=float,default=0.15,help="Baryonic fraction of halo (gas fraction)")
parser.add_argument("-rmax",type=float,default=None,help="Source distance radius in units of r200 (default: 1)")
parser.add_argument("-Ngas",type=int,default=1000,help="Number of gas particles to sample")
parser.add_argument("-Nstars",type=int,default=1,help="Number of star particles (sources) to sample")
args = parser.parse_args()

# Parameters of the halo
fb = args.fb                       # Baryonic fraction
M200 = 10.0**(float(args.logM200)) / 1e10
c = args.c
write_mass_fractions = True
outputfilename = str(args.output)

'''
Recall:
Virial radius r200 == radius s.t. mean dens inside r200 is 200*rho_c
Virial mass M200 == total mass inside r200 == 4/3πr200^3 * 200 * rho_c
'''

# Derived NFW parameters
delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
rho_0 = delta_c * rho_c
r_s = 1/c * (M200/(4/3.*np.pi*rho_c*200))**(1/3.)
r200 = c*r_s

# =======================
# NFW physical quantities
# =======================
# Gas Gas density
def Density(r):
    return fb*rho_0/((r/r_s)*(1+r/r_s)**2)

# Total Gas mass inside the radius r
def Mr(r):
    return 4*np.pi*rho_0*r_s**3 * ( np.log(1+r/r_s) - (r/r_s)/(1+r/r_s) )

# Circular velocity
def Vcirc(r):
    return  np.sqrt(G*Mr(r)/r)

# Integrand for the pressure
def integrand(r):
    return Density(r) * G * Mr(r) /  r**2

# Pressure
def P(r,rmax):
    Pr = quad(integrand, r, rmax, args=())
    return Pr[0]

# Specific energy
def U(P,r):
    u = P/(gamma-1)/Density(r)
    return u

# The model is truncated at some radius rmax (we want to fill the cube of side 2*r200)
if args.rmax is None:
    rmax = np.sqrt(3) * r200
else:
    rmax = float(args.rmax) * np.sqrt(3) * r200

rmin = 1e-3*r_s
nr   = 1000 
rr = 10**np.linspace(np.log10(rmin),np.log10(rmax),nr)

# Box size
L = 2*rmax

# Numerical parameters
Ngas = int(args.Ngas)
Mgas = fb * Mr(rmax) # Since the model is truncated at arbitrary rmax, the total particle mass must be = M(rmax)
mgas = Mgas / Ngas

# Estimate of dynamical time in internal coordinates (= period of circular orbit at r_s)
vcirc_at_rs = Vcirc(r_s)
tdyn = (2*np.pi*r_s / vcirc_at_rs * UnitTime_in_s * unyt.s).to('Myr')

# Print info
print("=== Internal Unit System === ")
print(f"UnitLength_in_cm         = {UnitLength_in_cm        : .4e}")
print(f"UnitMass_in_g            = {UnitMass_in_g           : .4e}")
print(f"UnitVelocity_in_cm_per_s = {UnitVelocity_in_cm_per_s: .4e}") 
print(f"UnitTime_in_s            = {UnitTime_in_s           : .4e}")
print(f"UnitEnergy_in_cgs        = {UnitEnergy_in_cgs       : .4e}")
print("")
print("=== Parameters in Internal Units: (Mpc, Msun, km/s), time in Myr === ")
print(f"G                = {G           : .5e}")
print(f"Critical density = {rho_c       : .5e}")
print(f"r200             = {r200            : .5e}")
print(f"r_s              = {r_s             : .5e}")
print(f"Ngas             = {Ngas            : n}")
print(f"M200 (Tot)       = {M200            : .5e}")
print(f"M200 (Gas)       = {fb*M200         : .5e}")
print(f"M200 (DM)        = {(1-fb)*M200     : .5e}")
print(f"mgas             = {mgas            : .5e}")
print(f"Estimate of tdyn = {tdyn: .5e}")
print(f"rmax             = {rmax        : .5e}")
print(f"boxsize          = {L           : .5f}")
print(f"boxsize / 2*r200 = {L/(2*r200)  : .2f}")

# ================================================
# Compute physical quantities in integration range
# ================================================
Vc = Vcirc(rr)
rho = Density(rr) 
Ps = np.zeros(len(rr))
for i in range(len(rr)):
    Ps[i] = P(rr[i],10*rmax) # Integral is to infty, here, use 10rmax
us = U(Ps,rr)

# =========================================
# Create Nbody object to generate positions
# NOTE: The Nbody object isn't used to
# write the ICs !
# =========================================
addargs = (r_s,)
pr_fct = profiles.nfw_profile
mr_fct = profiles.nfw_mr

Neps_des = 10.  # number of des. points in eps
ng = 256  # number of division to generate the model
rc = 0.1  # default rc (if automatic rc fails) length scale of the grid
dR = 0.1

Rqs, rc, eps, Neps, g, gm = ic.ComputeGridParameters(Ngas, addargs, rmax, M200, pr_fct, mr_fct, Neps_des, rc, ng)    
nb = ic.nfw(Ngas, r_s, rmax, dR, Rqs, name=args.output, ftype='swift')
nb.verbose = True

# Set gas mass
nb.mass = mgas * np.ones(Ngas)

# Units
nb.UnitLength_in_cm         = UnitLength_in_cm        
nb.UnitMass_in_g            = UnitMass_in_g           
nb.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s
nb.Unit_time_in_cgs         = UnitLength_in_cm/UnitVelocity_in_cm_per_s

# interpolate the specific energy
u_interpolator = interp1d(rr, us, kind='linear')
nb.u_init = u_interpolator(nb.rxyz())

# interpolate the density
rho_interpolator = interp1d(rr, rho, kind='linear')
nb.rho = rho_interpolator(nb.rxyz())

# Set hydro smoothing length (HSML)
rsp_1 = nb.get_rsp_approximation()
density, hsml = nb.ComputeDensityAndHsml(Hsml=rsp_1,DesNumNgb=48)
nb.rsp_init = hsml

# Set box size & shift particles
nb.boxsize = L
nb.pos += L / 2

# ==========================================================
# Create a swiftsimio writer, sample stars and write IC file
# ==========================================================
boxsize = L * unyt.kpc * np.array([1.0,1.0,1.0])
w = Writer(unit_system=halo_units,box_size=boxsize,dimension=3)

w.gas.coordinates = nb.pos * unyt.kpc
w.gas.velocities = nb.vel * unyt.km/unyt.s
w.gas.smoothing_length = hsml * unyt.kpc
w.gas.masses = nb.mass * 1e10 * unyt.Solar_Mass
w.gas.internal_energy = nb.u_init * (unyt.km/unyt.s)**2
w.gas.particle_ids = np.arange(SWIFT_NO_GRAVITY_BELOW_ID,SWIFT_NO_GRAVITY_BELOW_ID+Ngas)

# Generate star particles placed in random directions at r200
nstar = int(args.Nstars)
star_pos = np.empty((nstar,3))
star_vel = np.zeros((nstar,3))
star_mass = 1e-6 * mgas * np.ones(nstar) # to not mess with gravity

gen = np.random.default_rng(100)
phi_rand = gen.uniform(0.0, 2 * np.pi, nstar)
theta_rand = np.arccos(gen.uniform(-1.0, 1.0, nstar))

if args.rmax is None:
    r_star = r200
else:
    r_star = float(args.rmax) * r200
star_pos[:,0] = r_star * np.sin(theta_rand) * np.cos(phi_rand)
star_pos[:,1] = r_star * np.sin(theta_rand) * np.sin(phi_rand)
star_pos[:,2] = r_star * np.cos(theta_rand)                   

star_id = np.arange(nstar) # < NO_GRAVITY

w.stars.coordinates = (star_pos + L/2.0) * unyt.kpc
w.stars.velocities = star_vel * unyt.km/unyt.s
w.stars.masses = star_mass * 1e10 * unyt.Solar_Mass
w.stars.particle_ids = star_id
w.stars.smoothing_length = w.gas.smoothing_length[:1] * np.ones(nstar)

w.write(outputfilename)

if write_mass_fractions:
    # Now open file back up again and add RT data.
    F = h5py.File(outputfilename, "r+")
    header = F["Header"]
    nparts = header.attrs["NumPart_ThisFile"][0]
    parts = F["/PartType0"]

    # Create initial ionization species mass fractions.
    X = 1.0
    HIdata = np.ones((nparts), dtype=np.float32) * 0.9998 * X
    parts.create_dataset("MassFractionHI", data=HIdata)
    HIIdata = np.ones((nparts), dtype=np.float32) * 2.0e-4 * X
    parts.create_dataset("MassFractionHII", data=HIIdata)
    HeIdata = np.ones((nparts), dtype=np.float32) * (1.0-X)
    parts.create_dataset("MassFractionHeI", data=HeIdata)
    HeIIdata = np.ones((nparts), dtype=np.float32) * 1e-20
    parts.create_dataset("MassFractionHeII", data=HeIIdata)
    HeIIIdata = np.ones((nparts), dtype=np.float32) * 1e-20
    parts.create_dataset("MassFractionHeIII", data=HeIIIdata)

    # close up, and we're done!
    F.close()
