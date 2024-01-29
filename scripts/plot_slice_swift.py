import numpy as np
import matplotlib.pyplot as plt
import argparse
import cmasher as cmr
import swiftsimio
import unyt
from swiftsimio.visualisation.slice import slice_gas
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Constants
gamma = 5.0 / 3.0 # Adiabatic index

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="Snapshots to process")
parser.add_argument("--mass",action="store_true",help="Plot mass density")
parser.add_argument("--temp",action="store_true",help="Plot temperature")
parser.add_argument("--xHII",action="store_true",help="Plot ionized fraction")
parser.add_argument("--xHI",action="store_true",help="Plot neutral fraction")
parser.add_argument("--HII",action="store_true",help="Plot HII mass density")
parser.add_argument("--HI",action="store_true",help="Plot HI mass density")
parser.add_argument("--pressure",action="store_true",help="Plot pressure")
parser.add_argument("--show",action='store_true',help="Show plot rather than save it")
parser.add_argument("--tex",action='store_true',help="Use LaTeX")
parser.add_argument("--nolim",action='store_true',help="No color limits on temperature and xfrac")
parser.add_argument("-N",type=int,default=512)
parser.add_argument("-b",type=float,default=None,help="Box around center")
parser.add_argument("-z",type=float,default=None,help="Z-coordinate of slice in internal units (default: middle)")
args = parser.parse_args()

# Read in desired quantities to plot and set figure properties:
# H density, temperature, ionized fraction, neutral fraction, HII density, HI density, pressure
quantities = []
titles = []
cmaps = []
norms = []
if args.mass:
    quantities.append("Hmass")
    titles.append(r"$\log n_\mathrm{H}$ [cm$^{-3}$]")
    cmaps.append("viridis")
    #norms.append(Normalize(-1,4.5))
    norms.append(Normalize())
if args.temp:
    quantities.append("temp")
    titles.append(r"$\log T$ [K]")
    cmaps.append("afmhot")
    if args.nolim: vmax = None
    else: vmax = 4.8
    norms.append(Normalize(vmin=None,vmax=vmax))
if args.xHII:
    quantities.append("xHII")
    titles.append(r"$\log x_\mathrm{HII}$")
    cmaps.append("Spectral_r")
    if args.nolim:
        vmin = -4
        vmax = 0
    else:
        vmin = None
        vmax = None
    norms.append(Normalize(vmin=vmin,vmax=vmax))
if args.xHI:
    quantities.append("xHI")
    titles.append(r"$\log x_\mathrm{HI}$")
    cmaps.append("Spectral_r")
    if args.nolim:
        vmin = None
        vmax = 0
    else:
        vmin = -5
        vmax = 0
    norms.append(Normalize(vmin=vmin,vmax=vmax))
if args.HII:
    quantities.append("HIImass")
    titles.append(r"$\log n_\mathrm{HII}$ [cm$^{-3}$]")
    cmaps.append("cmr.dusk")
    norms.append(Normalize())
if args.HI:
    quantities.append("HImass")
    titles.append(r"$\log n_\mathrm{HI}$ [cm$^{-3}$]")
    cmaps.append("cmr.dusk")
    norms.append(Normalize())
if args.pressure:
    pass # TODO
    #quantities.append("pressure")

nquant = len(quantities)
if  nquant == 0:
    raise ValueError("Please specify a quantity to slice")
elif nquant > 4:
    raise ValueError("Can plot at most 4 quantities")

# Miscellaneous properties
Npx = int(args.N)
if args.tex: mpl.rcParams["text.usetex"] = True

# ======================
# Main plotting function
# ======================
def slice_snapshot(name):
    # Load data and set slice
    data = swiftsimio.load(name)
    L = data.metadata.boxsize[2]
    if hasattr(data.metadata,"time"):
        t = data.metadata.time
    else: t = 0.0 * unyt.Myr
    print(f"t       = {t.to('Myr'):.2f}")
    print(f"boxsize = {L.to('kpc'):.2f}")
    if args.z is None: z = L/2.0
    else: z = float(args.z) * L.units

    # Set region and image extent
    if args.b is None:
        xl = 0.0 * L.units
        xr = L
        yl = 0.0 * L.units
        yr = L
    else:
        b = args.b * L.units
        xl = L/2 - b
        xr = L/2 + b
        yl = L/2 - b
        yr = L/2 + b
    region = unyt.unyt_array([xl,xr,yl,yr])
    extent = (region - L/2.).value

    # Slicing function
    def myslice(quant):
        return slice_gas(
        data, z_slice=z,resolution=Npx,
        project=quant,parallel=True,
        region=region,periodic=False
    )

    # Slice total mass density
    mass_density_map = myslice("masses")

    # Slice ionized/neutral hydrogen density and mass fractions
    if hasattr(data.gas,'ion_mass_fractions'):
        print("IonMassFrac")
        data.gas.HIImass = data.gas.ion_mass_fractions.HII * data.gas.masses
    elif hasattr(data.gas,'hii'):
        data.gas.HIImass = data.gas.hii * data.gas.masses
    else:
        data.gas.HIImass = 2e-4 * data.gas.masses
    # print(data.gas.ion_mass_fractions.HI.max())
    # print(data.gas.ion_mass_fractions.HII.max())
    # print(data.gas.ion_mass_fractions.HeI.max())
    # print(data.gas.ion_mass_fractions.HeII.max())
    # print(data.gas.ion_mass_fractions.HeIII.max())
    massHII_density_map = myslice("HIImass")
    massHI_density_map = mass_density_map - massHII_density_map
    xHII_map = massHII_density_map / mass_density_map
    xHI_map = 1.0*xHII_map.units - xHII_map

    # Slice temperature if requested
    if "temp" in quantities:
        energy_density_map = myslice("total_energies")
        internal_energy_map = energy_density_map / mass_density_map
        mu_map = 1.0 / (1.0 + xHII_map.value)
        temperature_map = (gamma - 1.0) * mu_map * unyt.mh / unyt.kb * internal_energy_map
        temperature_map.convert_to_cgs()

    # Set units
    mass_density_map.convert_to_cgs()
    massHI_density_map.convert_to_cgs()
    massHII_density_map.convert_to_cgs()

    # Append requested images
    # Note mass densities are shown in units of the hydrogen mass
    images = []
    log_mh_g = np.log10(unyt.hydrogen_mass.to('g'))
    if "Hmass" in quantities:
        images.append(np.log10(mass_density_map.value) - log_mh_g)
    if "temp" in quantities:
        images.append(np.log10(temperature_map.value))
    if "xHII" in quantities:
        images.append(np.log10(xHII_map.value))
    if "xHI" in quantities:
        images.append(np.log10(xHI_map.value))
    if "HIImass" in quantities:
        images.append(np.log10(massHII_density_map.value) - log_mh_g)
    if "HImass" in quantities:
        images.append(np.log10(massHI_density_map.value) - log_mh_g)

    # Make figure
    fig, ax = plt.subplots(1,nquant,figsize=(nquant*5,4.0),tight_layout=True,squeeze=False)
    for i in range(nquant):
        ax[0,i].set_xlabel("x [kpc]")
        ax[0,i].set_ylabel("y [kpc]")
        ax[0,i].set_title(titles[i])
        im = ax[0,i].imshow(images[i].T,origin='lower',
                            extent=extent,cmap=cmaps[i],
                            norm=norms[i])
        plt.colorbar(im)
        ax[0,i].text(0.02, 0.98, f"$t={t.to('Myr').value:.2f}$ Myr",
            verticalalignment='top', horizontalalignment='left',
            transform=ax[0,i].transAxes,
            color='white', fontsize=9)

    return fig, ax


# ==================================
# Loop through files and make images
# ==================================
for k,fname in enumerate(args.files):
    print("Working on",fname)
    fig, ax = slice_snapshot(fname)

    if args.show:
        plt.show()
    else:
        outfn = fname.split(".")[0] + "_image.png"
        fig.savefig(outfn,dpi=200)
    plt.close(fig)
