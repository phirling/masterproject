import numpy as np
import argparse
import mpph
import h5py
import os
from tqdm import tqdm

# Parse user input
parser = argparse.ArgumentParser(description="Extract M(r) from snapshots")
parser.add_argument("files",nargs='+',help="Snapshot files")
parser.add_argument("-r", type=float, default=None, help="Radius for M(r)")
parser.add_argument("-o", type=str, default="HMass.txt", help="Output file")
args = parser.parse_args()

#fn = args.files[0]
nfiles = len(args.files)
res = np.empty((nfiles,4))
for i,fn in enumerate(tqdm(args.files)):
    mH = mpph.get_mass_sum_sph(fn,args.r,species='H')
    mHI = mpph.get_mass_sum_sph(fn,args.r,species='HI')
    mHII = mH - mHI
    t_myr = mpph.snapshot_time_myr(fn)
    res[i,:] = [t_myr,mH,mHI,mHII]

np.savetxt(str(args.o),res,header="t [Myr], MH [Msun], MHI [Msun], MHII [Msun]")
#print(f"{args.species} mass: {mpph.get_mass_sum_sph(fn,args.r,species=str(args.species)):.3e}")

#print(mpph.get_mean_photon_flux(fn,args.r))
