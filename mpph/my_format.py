import numpy as np
import h5py

class GridSnapshot:
    def __init__(self,file=None,N=None,dens_cgs = None, u_cgs = None, xfrac = None,
                 boxsize = None,time = None):
        """Snapshot data structure
        
        Attributes
        ----------
        N : int
            Mesh side length
        dens_cgs : 3d array of float64
            Hydrogen mass density in g/cm3
        u_cgs : 3d array of float64
            Internal specific energy in erg/g
        xfrac : 3d array of float64
            Ionized fraction of hydrogen
        boxsize : float
            Box size in Mpc
        time : float
            Snapshot time in Myr
        """
        if file is None:
            if N is None:
                raise ValueError("When no input file is given, need to provide at least a grid size N")
            else:
                self.N = N

                self.dens_cgs = dens_cgs

                if u_cgs is None:       self.u_cgs = np.zeros((N,N,N))
                else:                   self.u_cgs = u_cgs

                if xfrac is None:       self.xfrac = np.zeros((N,N,N))
                else:                   self.xfrac = xfrac

                if boxsize is None:     self.boxsize = 0.0
                else:                   self.boxsize = boxsize

                if time is None:        self.time = 0.0
                else:                   self.time = time
        else:
            with h5py.File(file,"r") as f:
                if 'dens_cgs' in f:
                    self.dens_cgs = np.array(f['dens_cgs'],dtype=np.float64)
                else:
                    self.dens_cgs = None
                self.u_cgs = np.array(f['u_cgs'],dtype=np.float64)
                self.xfrac = np.array(f['xfrac'],dtype=np.float64)
                self.boxsize = float(f.attrs['boxsize'])
                if 'time' in f.attrs: self.time = f.attrs['time']
                else: self.time = 0.0
                self.N = self.xfrac.shape[0]
            pass

    def write(self,fname):
        print("Writing to",fname," ... ")
        with h5py.File(fname,"w") as f:
            f.attrs['boxsize'] = self.boxsize
            f.attrs['time'] = self.time
            if self.dens_cgs is not None:
                f.create_dataset('dens_cgs',data=self.dens_cgs,dtype=np.float64,compression="gzip")
            f.create_dataset('u_cgs',data=self.u_cgs,dtype=np.float64,compression="gzip")
            f.create_dataset('xfrac',data=self.xfrac,dtype=np.float64,compression="gzip")