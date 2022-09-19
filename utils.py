import numpy as np
import math
import os
import h5py
import lsqfit
from model_avg_paper.stats import model_avg
from model_avg_paper import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gvar 
from jk import *

def multi_exp_model(t, p, Nexc=2):
    ans = 0.0
    for i in range(0, Nexc):
        ans += p["A{}".format(i)] * np.exp(-p["E{}".format(i)] * t)
    return ans

fitModelSingleExp = lambda t,p : multi_exp_model(t,p,1)
paramInitSingleExp = {
    "A0": 1, "E0": 0.5,
    #"A1": 0.1, "E1": 1.5,
    #"A2": 0.0001, "E2": 0.1
}

fitModelTripleExp = lambda t,p : multi_exp_model(t,p,3)
paramInitTripleExp = {
    "A0": 1, "E0": 0.5,
    "A1": 0.1, "E1": 1.5,
    "A2": 0.0001, "E2": 0.1
}


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

def get_pion_dir(baseDir,beta,ml,mh):
    mlDir=os.path.join(baseDir,'b{}_ml{}_mh{}'.format(beta,ml,mh))
    pionDir=os.path.join(mlDir,'mesons')
    return pionDir

def get_pipi_dir(baseDir,beta,ml,mh):
    mlDir=os.path.join(baseDir,'b{}_ml{}_mh{}'.format(beta,ml,mh))
    pionDir=os.path.join(mlDir,'pipi')
    return pionDir

def get_pion_from_file(file):
    return file['meson']['meson_0']

def get_pipi_from_file(file):
    return file['pipi']


def get_all_source_corrs(dir, baseName, sourceTimes, ncstart, ncfinish, step, h5function):
    pionCorrs={}

    for tsource in sourceTimes:

        corrs=[]
        for cfg in range(ncstart, ncfinish, step):
            fileName=os.path.join(dir,'{}.t{}.{}.h5'.format(baseName,tsource,cfg))
            file = h5py.File(fileName,'r')
            meson0=h5function(file)
            corr = mesonDatasetToNumpy(meson0)
            corr = np.roll(corr, -tsource)
            corrs.append(corr)

        pionCorrs[tsource]=np.asarray(corrs)
    return pionCorrs

def closest_E(params, ref):
    ens=[val.mean for name,val in params.items() if name[0]=='E']
    ensShift=[np.abs(val.mean-ref) for name,val in params.items() if name[0]=='E']
    ers=[val.sdev for name,val in params.items() if name[0]=='E']
    
    idx=np.argmin(ensShift)
    return gvar.gvar(ens[idx],ers[idx])

fitTypes = {'single': {'func': fitModelSingleExp, 'params': paramInitSingleExp},
            #'triple': {'func': fitModelTripleExp, 'params': paramInitTripleExp}
}

def perform_many_fits(mean,cov, tiStart, tiEnd, tfStart, tfEnd, NT):
    ts = np.array([t for t in range(len(mean))])
    obsVStmin=[]
    probVStmin=[]
    for name,fitType in fitTypes.items():
        for ti in range(7,19):
            for tf in range(20,25):
                if tf-ti > len(fitType['params']):
#                    try: 
                    fit=lsqfit.nonlinear_fit(
                            data=(ts[ti:tf+1], mean[ti:tf+1], cov[ti:tf+1,ti:tf+1]),
                            fcn = fitType['func'], 
                            p0=fitType['params']
                        )
#                    except:
#                        continue 
                    if name=='single':
                        obsVStmin.append(fit.p["E0"])
                    else:
                        # get single exp ref energy
                        # E_2 = 
                        singleExpEn=0
                        obsVStmin.append(closest_E(fit.p,singleExpEn))

                    prob = get_raw_model_prob(fit, IC="AIC", N_cut=(NT/2-tf)+ti)
                    probVStmin.append(prob)
    
    return obsVStmin, probVStmin



def old_eff_mass_mavg_plot():
    #from jk import *
    for ml in mls:
        mlDir=os.path.join(baseDir,'b{}_ml{}_mh{}'.format(beta,ml,mh))
        pionDir=os.path.join(mlDir,'mesons')

        ncStart, ncFinish = get_nc_start_fin(pionDir)
        NCFG = (ncFinish-ncStart)/ncStep

        sourceTimes=[i for i in range(0,64,8)]
        avgPionCorr=get_avg_pion_corr_folded(sourceTimes)

        effMass=jackKnife(log_effective_mass,avgPionCorr)

        plt.errorbar([t for t in range(len(effMass[0]))], effMass[0].real, yerr=effMass[1].real,
            linestyle="None", marker=".", lw=1, color="black")

        plt.plot([t for t in range(len(effMass[0]))],[e0(ml) for t in range(len(effMass[0]))],linestyle=(0,(5,10)),color="gray")
        plt.plot([t for t in range(len(effMass[0]))],[e0(ml) for t in range(len(effMass[0]))], color="red")
        #plt.fill_between(m+del,m-del)
        plt.xlim(0,32)
        plt.ylim(e0(ml)-10*e0err(ml),e0(ml)+10*e0err(ml))
        plt.title("m_l={}".format(ml))
        plt.show()


def model_average_summary_plot(avgPionCorr, energies, model_probs, mAvg):
    effMass=jackKnife(log_effective_mass,avgPionCorr)

    fig = plt.figure(figsize=(14,6))

    gs = gridspec.GridSpec(2,2, width_ratios=[2,1])

    axTopRight = fig.add_subplot(gs[0,1])
    axBotRight = fig.add_subplot(gs[1,1])
    axLeft = fig.add_subplot(gs[:,0])


    ts=[t for t in range(len(effMass[0]))]
    axLeft.fill_between(ts, [(mAvg.mean-mAvg.sdev) for t in ts],[(mAvg.mean+mAvg.sdev) for t in ts],
                    color='lightgray', alpha=0.5)

    axLeft.errorbar([t for t in range(len(effMass[0]))], effMass[0].real, yerr=effMass[1].real,
            linestyle="None", marker=".", lw=1, color="black")

    axLeft.plot(ts, [mAvg.mean for t in ts], color='gray')
    axLeft.set_ylabel('$aE~\\left[m_{\\pi}\\right]$')
    axLeft.set_xlabel('t')


    axTopRight.errorbar([i for i in range(len(energies))], [e.mean for e in energies], yerr=[e.sdev for e in energies],
        linestyle="None", marker=".", lw=1, color='black')
    axTopRight.errorbar([len(energies)+1], [mAvg.mean], yerr=[mAvg.sdev],
        linestyle="None", marker=".", lw=1, color='red')



    yScale=2.5

    axTopRight.set_ylim((mAvg.mean-yScale*mAvg.sdev), (mAvg.mean+yScale*mAvg.sdev))

    axTopRight.tick_params(
        axis='x',
        labelbottom=False,
    )
    axTopRight.set_ylabel('$aE~\\left[m_{\\pi}\\right]$')

    axBotRight.scatter([i for i in range(len(energies))], [p for p in model_probs/np.sum(model_probs)])
    axBotRight.tick_params(
        axis='x',
        labelbottom=False,
    )
    axBotRight.set_xlabel('fit')
    axBotRight.set_ylabel('$p$')