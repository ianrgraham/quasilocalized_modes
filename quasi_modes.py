"""
Script to grab a hoomd file and the local mode data for it
"""

import argparse
import numpy as np
#import pandas as pd
import sys
import time
import os
import gsd.hoomd
import gsd.fl
import hessian_calc as hc

parser = argparse.ArgumentParser(description="Produce quasilocalized mode data from hoomd trajectories.")
parser.add_argument('-d','--dir',help="Directory of the trajectory.", type=float, default=1.1)
cmdargs = parser.parse_args()

root_dir = cmdargs.dir

pjoin = os.path.join

traj = pjoin(root_dir,"traj.gsd")

# get traj info
f = gsd.fl.open(traj, 'rb')
nframes = f.nframes
f.close()

f = gsd.hoomd.open(traj, 'rb')

quasi = pjoin(root_dir, "quasi")

os.makedirs(quasi)

# probably should save data to disk in batches so not to end with

for i in np.arange(0, nframes, 10):
    outfile = pjoin(quasi, i)
    if i%50 == 0:
        print(outfile)
    mc = hc.mode_calculator_gsd(f, i)
    filt_vecs = np.array(mc.filter_modes())
    evecs = mc.evecs.T
    evals = mc.evals
    np.savez(outfile, filt_vecs=filt_vecs, evecs=evecs, evals=evals)

    

