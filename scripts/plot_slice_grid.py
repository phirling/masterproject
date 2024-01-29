import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpph import GridSnapshot
from matplotlib.colors import Normalize
import cmasher as cmr
import unyt
import matplotlib as mpl
import os
gamma = 5.0 / 3.0 # Adiabatic index

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="Nbody file")
parser.add_argument("-massfile",type=str,default=None,help="File that contains a mass density to plot")
parser.add_argument("--show",action='store_true',help="Use LaTeX")
parser.add_argument("-b",type=float,default=None,help="Box around center")
parser.add_argument("-XH",type=float,default=1.0,help="Hydrogen mass fraction")
parser.add_argument("-x", type=int,default=None)
parser.add_argument("-y", type=int,default=None)
parser.add_argument("-z", type=int,default=None)
parser.add_argument("-interp", type=str,default=None,help="Imshow interpolation to use")
parser.add_argument("--ionized",action='store_true',help="Show ionized H fraction rather than neutral")
parser.add_argument("--tex",action='store_true',help="Use LaTeX")
args = parser.parse_args()

if args.tex: mpl.rcParams["text.usetex"] = True

# Set color normalization for temperature and HI/HII fraction
norm_T = Normalize(vmin=3.2,vmax=5)
norm_x = Normalize(vmin=-5,vmax=0)
cmap_ndens = 'viridis'
cmap_T = 'magma'
cmap_x = 'Spectral_r'

# Hydrogen mass fraction
XH = args.XH

def slice_grid(fname):
    # Load data
    gs = GridSnapshot(fname)
    u_cgs = gs.u_cgs * unyt.erg / unyt.g
    boxsize_Mpc = gs.boxsize
    time_Myr = gs.time
    N = u_cgs.shape[0]

    # Print info
    print(f"t       = {time_Myr:.2f} Myr")
    print(f"boxsize = {boxsize_Mpc*1000:.2f} kpc")
    
    # Decide whether to plot ionized or neutral fraction
    if args.ionized:
        xfrac = gs.xfrac
    else:
        xfrac = 1.0 - gs.xfrac
        
    # Optionally plot the Hydrogen number density as first panel (if provided)
    plot_dens = False
    nquant = 2
    if args.massfile is not None:
        file_mass = str(args.massfile)
        gs_mass = GridSnapshot(file_mass)
        dens_cgs = XH * gs_mass.dens_cgs
        nquant = 3
        plot_dens = True

    # Compute temperature
    mu_grid = 1.0 / (XH * (1+gs.xfrac+0.25*(1.0/XH-1.0)))
    temp_cgs = ((gamma-1) * mu_grid * unyt.mass_hydrogen/unyt.kb * u_cgs).to('K').value

    # Create figure and set info
    fig, ax = plt.subplots(1,nquant,figsize=(nquant*5,4.0),tight_layout=True,squeeze=False)
    for k in range(nquant):
        ax[0,k].set_xlabel("$x$ [kpc]")
        ax[0,k].set_ylabel("$y$ [kpc]")
        ax[0,k].text(0.02, 0.98, f"$t={time_Myr:.2f}$ Myr",
                verticalalignment='top', horizontalalignment='left',
                transform=ax[0,k].transAxes,
                color='white', fontsize=9)
    if plot_dens: ax[0,0].set_title(r"$\log n_\mathrm{H}$ [cm$^{-3}$]")
    ax[0,nquant-2].set_title(r"$\log T$ [K]")
    if args.ionized: ax[0,nquant-1].set_title(r"$\log x_\mathrm{HI}$")
    else: ax[0,nquant-1].set_title(r"$\log x_\mathrm{HII}$")

    # Set region and image extent
    dr_kpc = 1000 * boxsize_Mpc / N
    boxsize_kpc = 1000*boxsize_Mpc
    if args.b is None:
        il = 0
        ir = N
    else:
        b = args.b
        il = int(N//2 - b/dr_kpc)
        ir = int(N//2 + b/dr_kpc)
    extent_kpc = [il*dr_kpc - boxsize_kpc/2,
                ir*dr_kpc - boxsize_kpc/2,
                il*dr_kpc - boxsize_kpc/2,
                ir*dr_kpc - boxsize_kpc/2]

    if args.x is not None:
        if plot_dens: dens_slice = dens_cgs[int(args.x),:,:]
        temp_slice = temp_cgs[int(args.x),:,:]
        x_HI_slice = xfrac[int(args.x),:,:]
    elif args.y is not None:
        if plot_dens: dens_slice = dens_cgs[:,int(args.y),:]
        temp_slice = temp_cgs[:,int(args.y),:]
        x_HI_slice = xfrac[:,int(args.y),:]
    elif args.z is not None:
        if plot_dens: dens_slice = dens_cgs[:,:,int(args.z)]
        temp_slice = temp_cgs[:,:,int(args.z)]
        x_HI_slice = xfrac[:,:,int(args.z)]
    else:
        ctr = N//2-1
        if plot_dens: dens_slice = dens_cgs[il:ir,il:ir,ctr]
        temp_slice = temp_cgs[il:ir,il:ir,ctr]
        x_HI_slice = xfrac[il:ir,il:ir,ctr]

    iitp = str(args.interp)
    if plot_dens: im0 = ax[0,0].imshow(np.log10(dens_slice.T),extent=extent_kpc,origin='lower',cmap=cmap_ndens,interpolation=iitp)
    im1 = ax[0,nquant-2].imshow(np.log10(temp_slice.T),extent=extent_kpc,origin='lower',cmap=cmap_T,norm=norm_T,interpolation=iitp)
    im2 = ax[0,nquant-1].imshow(np.log10(x_HI_slice.T),extent=extent_kpc,origin='lower',cmap=cmap_x,norm=norm_x,interpolation=iitp)

    if plot_dens: plt.colorbar(im0)
    plt.colorbar(im1)
    plt.colorbar(im2)

    return fig, ax

# ==================================
# Loop through files and make images
# ==================================
for k,fname in enumerate(args.files):
    print("Working on",fname)
    fig, ax = slice_grid(fname)

    if args.show:
        plt.show()
    else:
        outfn = os.path.splitext(fname)[0] + "_image.png"
        print(outfn)
        #outfn = fname.split(".")[-2] + "_image.png"
        fig.savefig(outfn,dpi=200)

    plt.close(fig)
