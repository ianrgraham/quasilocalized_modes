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
parser.add_argument('-d','--dir',help="Directory of the trajectory.", type=str, default="")
parser.add_argument('-r','--removeRattlers',help="Remove rattlers from Hessian calculation.", action='store_true')
cmdargs = parser.parse_args()

root_dir = cmdargs.dir

remove_ratt = cmdargs.removeRattlers

SSP = int(np.round(4*float(root_dir.split('/')[-1].split("_")[-2].replace("maxStrain",""))/0.0001))

print(SSP)

overwrite = True

assert(root_dir != "")

pjoin = os.path.join

traj = pjoin(root_dir,"traj.gsd")

# get traj info
f = gsd.fl.open(name=traj, mode='rb')
nframes = f.nframes
f.close()

print(nframes)

with gsd.hoomd.open(name=traj, mode='rb') as t:

    if remove_ratt:
        quasi = pjoin(root_dir, "quasi_noratt")
    else:
        quasi = pjoin(root_dir, "quasi")

    os.makedirs(quasi, exist_ok=True)

    # probably should save data to disk in batches so not to end with

    for i in range(0, 2*SSP+1, 10):
        outfile = pjoin(quasi, str(i))
        if os.path.exists(outfile+".npz") and not overwrite:
            print("Output already exists. We won't waste time reproducing it")
            continue
        if i%50 == 0:
            print("On step", outfile)
            time1 = time.time()
        try:
            mc = hc.mode_calculator_gsd(t, i, remove_rattlers=remove_ratt)
            filt_vecs = np.array(mc.filter_modes())
            evecs = mc.evecs.T
            evals = mc.evals
            if i%SSP == 0:
                np.savez_compressed(outfile, filt_vecs=filt_vecs, evecs=evecs, evals=evals)
            else:
                np.savez_compressed(outfile, filt_vecs=filt_vecs, evals=evals)
            if i%50 == 0:
                print(f"Step took {time.time()- time1} seconds to complete")
        except:
            print("Something went wrong! This really shouldn't happen")
            continue
    j = int(nframes//SSP)
    for i in range((j-2)*SSP, nframes, 10):
        outfile = pjoin(quasi, str(i))
        if os.path.exists(outfile+".npz") and not overwrite:
            print("Output already exists. We won't waste time reproducing it")
            continue
        if i%50 == 0:
            print("On step", outfile)
            time1 = time.time()
        try:
            mc = hc.mode_calculator_gsd(t, i, remove_rattlers=remove_ratt)
            filt_vecs = np.array(mc.filter_modes())
            evecs = mc.evecs.T
            evals = mc.evals
            if i%SSP == 0:
                np.savez_compressed(outfile, filt_vecs=filt_vecs, evecs=evecs, evals=evals)
            else:
                np.savez_compressed(outfile, filt_vecs=filt_vecs, evals=evals)
            if i%50 == 0:
                print(f"Step took {time.time()- time1} seconds to complete")
        except:
            print("Something went wrong! This really shouldn't happen")
            continue
