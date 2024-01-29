import argparse
from mpph import estimate_grav_softening
import unyt

parser = argparse.ArgumentParser()
parser.add_argument("files", nargs="+", help="Snapshots to process")
parser.add_argument("-npart", type=int,default=10, help="Number of particles in sphere to estimate eps")
args = parser.parse_args()

npart = int(args.npart)
print(f"Estimating softening as radius of sphere containing {npart} particles...")
for k,fn in enumerate(args.files):
    eps = estimate_grav_softening(fn,npart)
    print(f"-> For IC file '{fn}', eps = {eps:.5f} kpc")