import numpy as np
from pNbody import mapping
from .my_format import GridSnapshot
from swiftsimio import load
from pNbody import Nbody
import unyt

def indices_particles_in_box(pos,x0,x1):
    in_x = np.logical_and(pos[:,0] >= x0, pos[:,0] <= x1)
    in_y = np.logical_and(pos[:,1] >= x0, pos[:,1] <= x1)
    in_z = np.logical_and(pos[:,2] >= x0, pos[:,2] <= x1)
    in_all = np.logical_and(np.logical_and(in_x,in_y),in_z)
    return np.where(in_all == 1)[0]


def swift_to_grid(fname,N,b,frsp,output,xHII = 2.0e-4):
    """
    Apply SPH smoothing to a set of particles to project their physical quantities on a cartesian grid

    The particle file must be formatted for SWIFT, i.e. must be readable by the swiftsimio module
    and contains quantities with physical dimensions using the unyt package.
    
    Arguments
    ---------
    fname : string
        Name of the file containing the particles
    N : int
        Size of the cartesian grid in each dimension
    b : float
        Size of the cubic subregion to grid. This will be the box size of the gridded data.
        It is given as a length in kpc from the center of the particle box.
        Example: The particles have (x,y,z) positions in the interval [0,1] Mpc.
        If we pass -b 400, the resulting grid will span the box [0.1, 0.9] Mpc.
    frsp : float
        Factor by which to multiply the smoothing lenghts of the particles. 3 is a good value
        in 3D to avoid weird artifacts
    output : string
        Name of the output file to write
    xHII : float
        Fraction of ionized gas to write to the file
    """
    grid_boxsize_kpc = 2*b*unyt.kpc #(xmax-xmin)

    data = load(fname)
    boxsize_kpc = data.metadata.boxsize[2].to(unyt.kpc).value
    # We project only the gas particles!
    mass_1e10Ms = (data.gas.masses.to(1e10 * unyt.Solar_Mass).value).astype(np.float32) # IMPORTANT: otherwise mkmap freaks out
    pos_kpc = data.gas.coordinates.to(unyt.kpc).value
    hsml_kpc = data.gas.smoothing_length.to(unyt.kpc).value
    u_kmps2 = (data.gas.internal_energy.to((unyt.km/unyt.s)**2).value).astype(np.float32)
    npart = pos_kpc.shape[0]

    dr = grid_boxsize_kpc / N
    dV = dr**3

    print("Ngrid:                   ",N)
    print("Number of particles:     ",npart)
    print("SWIFT Box size [kpc]:    ",boxsize_kpc)
    print("Grid Box size [kpc]:     ",grid_boxsize_kpc)
    print("Grid cell size [kpc]:    ",dr.value)

    # We normalize the positions to the grid box size. Any particle outside
    # the box size will be excluded from the gaussian map
    xmin = boxsize_kpc/2.0 - b
    xmax = boxsize_kpc/2.0 + b
    idx = indices_particles_in_box(pos_kpc,xmin,xmax)
    total_mass_in_box_1e10Ms = mass_1e10Ms[idx].sum()

    grid_pos_norm = ((pos_kpc - xmin) / grid_boxsize_kpc).value
    h = (frsp * hsml_kpc / grid_boxsize_kpc).value

    print(f"INFO: Doing SPH smoothing with rsp factor {frsp:.3f}")
    print("Smoothing mass...")
    M_filtered = mapping.mkmap3dksph(grid_pos_norm,mass_1e10Ms,np.ones(npart,np.float32),h, (N, N, N),verbose=1)
    print("Smoothing internal energy...")
    U_filtered = mapping.mkmap3dksph(grid_pos_norm,mass_1e10Ms * u_kmps2,np.ones(npart,np.float32),h, (N, N, N),verbose=1)
    u_filtered = np.where(M_filtered > 0,U_filtered / M_filtered,0.0)

    # Set correct units and convert
    M_filtered = M_filtered * 1e10 * unyt.Solar_Mass
    u_filtered = u_filtered * (unyt.km/unyt.s)**2
    dens_cgs = (M_filtered/dV).to('g/cm**3')
    ndens_cgs = (dens_cgs/unyt.mass_hydrogen).to('1/cm**3')
    u_cgs = u_filtered.to(unyt.erg/unyt.g)
    print(ndens_cgs.max())
    mass_cons_err = ( M_filtered.value.sum()/1e10 - total_mass_in_box_1e10Ms) / total_mass_in_box_1e10Ms
    print("Mass conservation error (relative):", mass_cons_err)

    # Save result
    xfrac = float(xHII) * np.ones((N,N,N))
    gs = GridSnapshot(N=N,dens_cgs=ndens_cgs.value,u_cgs=u_cgs.value,xfrac=xfrac,boxsize=grid_boxsize_kpc/1000,time=0.0)
    gs.write(str(output))


def pnbody_to_grid(fname,N,region,frsp,output,xHII = 2.0e-4,fill=True):
    """
    Apply SPH smoothing to a set of particles to project their physical quantities on a cartesian grid

    The particle file must be formatted for SWIFT, i.e. must be readable by the swiftsimio module
    and contains quantities with physical dimensions using the unyt package.
    
    Arguments
    ---------
    fname : string
        Name of the file containing the particles
    N : int
        Size of the cartesian grid in each dimension
    region : list
        [xmin,xmax,ymin,ymax,zmin,zmax] in kpc
    frsp : float
        Factor by which to multiply the smoothing lenghts of the particles. 3 is a good value
        in 3D to avoid weird artifacts
    output : string
        Name of the output file to write
    xHII : float
        Fraction of ionized gas to write to the file
    """
    xmin = region[0]
    xmax = region[1]
    ymin = region[2]
    ymax = region[3]
    zmin = region[4]
    zmax = region[5]
    L = xmax - xmin
    if (ymax-ymin) != L or (zmax-zmin) != L:
        raise ValueError("Region must be cubic")

    nb = Nbody(fname)
    boxsize_kpc = nb.boxsize[0] * nb.atime

    # We project only the gas particles!
    id_gas = np.where(nb.tpe == 0)[0]
    pos_kpc = nb.Pos(units='kpc')[id_gas]
    mass_1e10Ms = nb.Mass(units='Msol')[id_gas] / 1e10
    hsml_kpc = (nb.rsp * nb.atime)[id_gas] #data.gas.smoothing_length.to(unyt.kpc).value
    u_kmps2 = nb.u[id_gas] #(data.gas.internal_energy.to((unyt.km/unyt.s)**2).value).astype(np.float32)
    npart = len(mass_1e10Ms)

    # if fill:
    #     nfill = int((npart**(1./3))/10)
    #     rfill = np.linspace(xmin,xmax,nfill)
    #     XX,YY,ZZ = np.meshgrid(rfill,rfill,rfill)
    #     pos_fill = np.empty((nfill**3,3))
    #     pos_fill[:,0] = XX.flatten()
    #     pos_fill[:,1] = YY.flatten()
    #     pos_fill[:,2] = ZZ.flatten()
    #     mass_fill = mass_1e10Ms[0] / 100 * np.ones(nfill)
    #     print(pos_fill)
    #     nb2 = Nbody(status='new',pos=pos_fill,mass=mass_fill)
    #     hsml_fill = nb2.get_rsp_approximation()
    #     print(hsml_fill)
    #     #hsml_fill = nb2.rsp
    
    dr = L / N * unyt.kpc
    dV = dr**3

    print("Ngrid:                   ",N)
    print("Number of particles:     ",npart)
    print("SWIFT Box size [kpc]:    ",boxsize_kpc)
    print("Grid Box size [kpc]:     ",L)
    print("Grid cell size [kpc]:    ",dr.value)

    # We normalize the positions to the grid box size. Any particle outside
    # the box size will be excluded from the gaussian map
    print(pos_kpc)
    grid_pos_norm = np.copy(pos_kpc)
    grid_pos_norm[:,0] = ((pos_kpc[:,0]-xmin) / L)
    grid_pos_norm[:,1] = ((pos_kpc[:,1]-ymin) / L)
    grid_pos_norm[:,2] = ((pos_kpc[:,2]-zmin) / L)
    h = (frsp * hsml_kpc / L)

    print(f"INFO: Doing SPH smoothing with rsp factor {frsp:.3f}")
    print(grid_pos_norm)
    print(mass_1e10Ms)
    print(h)
    print("Smoothing mass...")
    M_filtered = mapping.mkmap3dksph(grid_pos_norm,mass_1e10Ms,np.ones(npart,np.float32),h, (N, N, N),verbose=1)
    print("Smoothing internal energy...")
    U_filtered = mapping.mkmap3dksph(grid_pos_norm,mass_1e10Ms * u_kmps2,np.ones(npart,np.float32),h, (N, N, N),verbose=1)
    u_filtered = np.where(M_filtered > 0,U_filtered / M_filtered,0.0)

    # Set correct units and convert
    NDENS_MIN = 1e-10 * 1.0 / unyt.cm**3
    U_MIN = 1e8 * (unyt.erg / unyt.g)
    M_filtered = M_filtered * 1e10 * unyt.Solar_Mass
    u_filtered = u_filtered * (unyt.km/unyt.s)**2
    dens_cgs = (M_filtered/dV).to('g/cm**3')
    ndens_cgs = np.maximum((dens_cgs/unyt.mass_hydrogen).to('1/cm**3') , NDENS_MIN)
    u_cgs = np.maximum(u_filtered.to(unyt.erg/unyt.g) , U_MIN)
    print(ndens_cgs.max())
    #print(np.nanmin(ndens_cgs[ndens_cgs.nonzero()]))

    # Save result
    xfrac = float(xHII) * np.ones((N,N,N))
    gs = GridSnapshot(N=N,dens_cgs=ndens_cgs.value,u_cgs=u_cgs.value,xfrac=xfrac,boxsize=L/1000,time=0.0)
    gs.write(str(output))