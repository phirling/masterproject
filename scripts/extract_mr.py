import numpy as np
import argparse
import mpph
import h5py
import os

"""
To extract M(r):
1. Explicitely give a radius r in kpc
2. If none is given, defaults to the largest radius of the grid
"""
# Parse user input
parser = argparse.ArgumentParser(description="Extract M(r) from snapshots")
parser.add_argument("files",nargs='+',help="Snapshot files")
parser.add_argument("-massfile",type=str,default=None,help="File to extract density")
parser.add_argument("-r", type=float, default=None, help="Radius for M(r)")
parser.add_argument("-o", type=str, default="profiles.hdf5", help="Output file")
parser.add_argument("--verbose",action='store_true')
args = parser.parse_args()

outfn = str(args.o)
fn_mass = str(args.massfile)

# Create new output HDF5 file if it doesn't exist
if args.verbose: print("Saving profiles to '" + outfn + "' ...")

# Load the file containing the density grid
#gs_mass = mpph.GridSnapshot(args.massfile)
#dens_cgs = gs_mass.dens_cgs

# Now loop through snapshots and extract the profiles
for k,fn in enumerate(args.files):
    gs = mpph.GridSnapshot(fn)
    if args.r is None:
        r = gs.boxsize * 1000 / 2.0 # gs.boxsize is in Mpc
    else:
        r = float(args.r)
    
    # Extract masses
    M_H = mpph.get_mass_sum_grid(fn,fn_mass,r,'H')
    M_HI = mpph.get_mass_sum_grid(fn,fn_mass,r,'HI')
    M_HII = mpph.get_mass_sum_grid(fn,fn_mass,r,'HII')
    gas_Mass = np.array([M_H,M_HI,M_HII])

    # Append result to HDF5 file (will create file otherwise)
    with h5py.File(outfn,"a") as f:
        hname = os.path.splitext(os.path.basename(fn))[0]
        if hname in f:
            if args.verbose: print("Group '" + hname + "' exists, write into")
        else:
            if args.verbose: print("Create group '" + hname + "'")
            f.create_group(hname)
        
        #if "gas_Masses" in f[hname]:
        #    f[hname+"/gas_Masses"][:] = gas_Mass
        #else:
        #    f[hname].create_dataset('gas_Masses',data=gas_Mass,dtype=np.float64)
        f[hname].create_dataset('gas_Masses',data=gas_Mass,dtype=np.float64)
        f[hname+"/gas_Masses"].attrs['r'] = r