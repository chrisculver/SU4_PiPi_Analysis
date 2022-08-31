import numpy as np
import math
import os

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
