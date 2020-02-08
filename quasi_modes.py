"""
Script to grab a hoomd file and the local mode data for it
"""

import argparse
import numpy as np
import pandas as pd
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

# probably should save data to disk in batches so not to end with

for i in np.arange(0, nframes, 10):
    mc = hc.mode_calculator_gsd(f, i)
    filt_vecs = mc.filter_modes()
    evecs = mc.evecs
    evals = mc.evals

