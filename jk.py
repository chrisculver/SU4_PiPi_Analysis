import numpy as np
import math 

# the first index of the data array should run over the N samples

def jackKnifeSamples(data, b=1):
  N=len(data)
  if not (N/b).is_integer():
    print('Binning data error, len(data)={} and bin size ={}'.format(N,b))
    raise ValueError

  indices=np.arange(N)
  bins=np.split(indices, N/b)
#  print(bins) 
  jackKnifeSamples=[]
  for b in bins:
    jackKnifeSamples.append([e for i,e in enumerate(data) if i not in b])

  return np.array(jackKnifeSamples)


def jackKnife(func, data, b=1):
  samples=jackKnifeSamples(data,b)
  vals=[func(s) for s in samples]
  N=len(vals)
  var=np.var(vals, axis=0)

  return (np.mean(vals, axis=0), np.sqrt(var*(N-1.)))


def jackKnifeCov(func, data, b=1):
  samples=jackKnifeSamples(data,b)
  vals=np.array([func(s) for s in samples])

  nm0 = lambda data : np.mean(data, axis=0)

  cov = np.array([[nm0(vals[:,i]*vals[:,j])-nm0(vals[:,i])*nm0(vals[:,j]) 
                    for i in range(len(vals[0]))] 
                    for j in range(len(vals[0]))])*(len(vals)-1)

  return [np.mean(vals,axis=0), cov]



#data1=np.array([1.1,1.0,1.2,1.5,1.2,1.3])
#data2=np.array([2.4,2.3,2.6,2.9,2.4,2.4])


#print(jackKnife(np.mean, data1, 1))
#print(jackKnife(np.mean, data1, 2))
#print(jackKnife(np.mean, data1, 3))
#print()
#nmean = lambda data : np.mean(data,axis=0)
#cov1=jackKnifeCov(nmean, np.array([data1, data2]).transpose(), 1)
#cov2=jackKnifeCov(nmean, np.array([data1, data2]).transpose(), 2)
#cov3=jackKnifeCov(nmean, np.array([data1, data2]).transpose(), 3)

#print(cov1)
#print(math.sqrt(cov1[1][0,0]),"  ", math.sqrt(cov1[1][1,1]))
#print()
#print(cov2)
#print(math.sqrt(cov2[1][0,0]),"  ", math.sqrt(cov2[1][1,1]))
#print()
#print(cov3)
#print(math.sqrt(cov3[1][0,0]),"  ", math.sqrt(cov3[1][1,1]))


  
  
