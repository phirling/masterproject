import unyt
import numpy as np
from scipy.integrate import quad

mass_hydrogen_g = unyt.hydrogen_mass.to('g').value
"""Hydrogen Mass in g"""
G_kpc_Ms_kms = (unyt.G.to(unyt.kpc / (unyt.Solar_Mass) * (unyt.km/ unyt.s)**2)).value
"""Gravitational constant in kpc * (km/s)^2 / Ms"""
G_cgs = (unyt.G.to('cm**3/(g*s**2)')).value
"""Gravitational constant in CGS"""
k_B_cgs = (unyt.kb_cgs).value
"""Boltzmann constant in CGS"""
Msun_per_kpc3_to_cgs = (unyt.Msun.to('g') / (1*unyt.kpc).to('cm')**3).value
"""Density conversion factor Ms/kpc**3 --> g/cm**3"""

def get_NFW_props(logM200,c,h = 0.72):
    """
    Get the physical properties of a NFW halo

    Parameters
    ----------
    logM200 : float
        log10 of the virial mass of the halo in solar masses
    c : float
        Concentration of the halo
    h : float
        Dimensionless hubble parameter ("small h")
    """
    M200 = 10**logM200 * unyt.Solar_Mass
    
    H0 = 100 * h * unyt.km / unyt.s / unyt.Mpc
    rho_c = 3*H0**2 / (8*np.pi*unyt.G)
    rho_c = rho_c.to(10**10*unyt.Solar_Mass/unyt.kpc**3)
    G = (unyt.G.to(unyt.kpc / (1e10*unyt.Solar_Mass) * (unyt.km/ unyt.s)**2))

    # Derived NFW parameters
    delta_c = 200/3. * c**3/( np.log(1+c) - c/(1+c) )
    rho_0 = delta_c * rho_c
    r_s = 1/c * (M200/(4/3.*np.pi*rho_c*200))**(1/3.)
    r200 = c*r_s

    nfw = {
        'M200' : M200,
        'c' : c,
        'rho_c' : rho_c,
        'rho_0' : rho_0,
        'r_s' : r_s,
        'r_200' : r200
    }

    return nfw

# Gas Gas density
def NFW_Density(r,rho_0,r_s):
    return rho_0/((r/r_s)*(1+r/r_s)**2)

# Total Gas mass inside the radius r
def NFW_Mr(r,rho_0,r_s):
    return 4*np.pi*rho_0*r_s**3 * ( np.log(1+r/r_s) - (r/r_s)/(1+r/r_s) )

# Circular velocity
def NFW_Vcirc(r,rho_0,r_s,G):
    return  np.sqrt(G*NFW_Mr(r,rho_0,r_s)/r)

# Pressure
def NFW_Pressure(r,rmax,rho_0,r_s,G):
    integrand = lambda r : NFW_Density(r,rho_0,r_s) * G * NFW_Mr(r,rho_0,r_s) /  r**2
    Pr = quad(integrand, r, rmax)
    return Pr[0]

# Specific energy
def NFW_internal_energy(P,r,rho_0,r_s):
    gamma = 5.0 / 3.0
    u = P/(gamma-1)/NFW_Density(r,rho_0,r_s)
    return u

# fact = 1.0109 @ N=256