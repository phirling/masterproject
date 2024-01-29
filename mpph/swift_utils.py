import unyt
from swiftsimio import load

def snapshot_time_myr(fname):
    data = load(fname)
    if hasattr(data.metadata,"time"):
        t = data.metadata.time
    else: t = 0.0 * unyt.Myr
    return float(t.to('Myr').value)