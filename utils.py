import numpy as np
import math
import os
import h5py

def parse_corr_file(fileBase, i, j, NT):
    values=[]

    corrFile = os.path.join(fileBase,"corr_op.{}_op.{}.dat".format(i,j))
    conjugateCorr = False
    if not os.path.exists(corrFile):
        corrFile = os.path.join(fileBase,"corr_op.{}_op.{}.dat".format(j,i))
        if not os.path.exists(corrFile):
            raise RuntimeError("Couldn't find any file to fill data for corr i={},j={}".format(i,j))
        conjugateCorr=True    

    with open(corrFile, 'r') as data:
        
        for line in data:
            cols=line.split(' ')
            values.append(float(cols[0])+float(cols[1])*1j)
    
    NC=len(values)//NT 
    if len(values) % NT != 0:
        raise RuntimeError("Number of values not divisible by NT, NT is wrong!")

    values = np.reshape(values, (NC,NT))
    if conjugateCorr:
        np.conjugate(values)

    return values


def log_effective_mass(corr):
    avg=np.mean(corr, axis=0)
    res=[]

    for t in range(len(avg)-1):
        try:
            res.append(math.log(avg[t]/avg[t+1]))
        except ValueError:
            res.append(-1j*math.pi + math.log(-avg[t]/avg[t+1]))
        except: 
            raise
    
    return res


def foldCorr(c,NT):
    res=[c[0]]
    for t in range(1,int(NT/2)-1):
        res.append((c[t]+c[NT-t])/2.)
    res.append(c[int(NT/2)])
    
    return res

def mesonDatasetToNumpy(dataset):
    data = dataset['corr']
    arr = []
    for elem in data:
        arr.append(complex(elem[0],elem[1]))
    return np.array(arr)


def get_nc_start_fin(dir):
    files=[f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir,f))]
    files=[f for f in files if f.split('.')[1]=='t0']
    ncStart=int(files[0].split('.')[2])
    ncFinish=ncStart
    for f in files:
        cfg = int(f.split('.')[2])
        if cfg < ncStart:
            ncStart=int(cfg)
        if cfg > ncFinish:
            ncFinish=int(cfg)

    return ncStart, ncFinish



def get_all_pion_source_corrs(pionDir, sourceTimes, ncstart, ncfinish, step):
    pionCorrs={}
    for tsource in sourceTimes:

        corrs=[]
        for cfg in range(ncstart, ncfinish, step):
            fileName=os.path.join(pionDir,'wall_ll.t{}.{}.h5'.format(tsource,cfg))
            file = h5py.File(fileName,'r')
            meson0=file['meson']['meson_0']
            corr = mesonDatasetToNumpy(meson0)
            corr = np.roll(corr, -tsource)
            corrs.append(corr)

        pionCorrs[tsource]=np.asarray(corrs)
    return pionCorrs