from swiftsimio import load as load_swift
import numpy as np
from astropy.constants import m_p, k_B, M_sun

def estimate_grav_softening(filename,npart=10):
    """
    Estimate a good gravitational softening length

    Computes the radius of a sphere (from halo center)
    which contains npart particles

    Parameters
    ----------
    filename : str
        Name of the IC file
    npart : int
        Number of particles in the sphere giving the softening
    
    Returns
    -------
    eps : float
        Softening length in kpc
    """
    data = load_swift(filename)
    L = data.metadata.boxsize[2]
    pos = data.gas.coordinates
    pos -= L/2.0
    radii = np.sqrt((pos**2).sum(axis=1))
    radii = np.sort( radii.to('kpc').value )
    return radii[npart-1]

def get_density_profile_sph(filename,XH=1.0):
    data = load_swift(filename)
    L = data.metadata.boxsize[2]
    pos = data.gas.coordinates - L/2.
    print(data.metadata.time.to('Myr'))
    radii = ( np.sqrt((pos**2).sum(axis=1)) ).to('kpc').value
    mdens_gas = data.gas.densities.to('g/cm**3').value
    ndens = (XH * mdens_gas) / m_p.cgs.value

    return radii, ndens

def get_mass_sum_sph(filename,r,species='H',XH=1.0):
    data = load_swift(filename)
    L = data.metadata.boxsize[2]
    pos = data.gas.coordinates - L/2.
    radii = ( np.sqrt((pos**2).sum(axis=1)) ).to('kpc').value
    if hasattr(data.gas,'ion_mass_fractions'):
        xfrac = data.gas.ion_mass_fractions.HII
    else:
        xfrac = data.gas.hii

    if species == 'H':
        mf = 1.0
    elif species == 'HI':
        mf = 1.0 - xfrac
    elif species == 'HII':
        mf = xfrac
    else:
        raise ValueError("Unknown species:",species)
    masses = data.gas.masses * mf
    Mr = (masses * (radii <= r)).sum()
    return Mr

def get_mean_photon_flux(filename,r):
    data = load_swift(filename) 
    L = data.metadata.boxsize[2]
    pos = data.gas.coordinates - L/2.
    radii = ( np.sqrt((pos**2).sum(axis=1)) ).to('kpc').value
    photon_fluxes = np.sqrt(data.gas.photon_fluxes.Group1X**2 +
                            data.gas.photon_fluxes.Group1Y**2 +
                            data.gas.photon_fluxes.Group1Z**2)
    masked_flux = (photon_fluxes * (radii <= r))
    meanflux = masked_flux.sum() / np.count_nonzero(masked_flux)
    return meanflux.to('erg/s/cm**2')