
#
# Template for simulation to be run on each process 
#
from scipy.integrate import odeint
import numpy as np 

params={'kon': 1,
        'koff': 1,
        'bMax': 5,
        'scale':1}       

def dydt(ys,ts,params):
  kon = params['kon']
  koff = params['koff']
  bMax = params['bMax']
  scale = params['scale']

  ca,cab,na = ys
  dca_dt = +koff*cab - kon*ca*(bMax-cab)
  dcab_dt = -dca_dt
  dna_dt  = -scale*na

  return dca_dt,dcab_dt,dna_dt
  
def simulation(
  varDict=params,      # dictionary of parameters to be used for simulation
  returnDict=dict()    # dictionary output to be returned by simulation
  ):
  varDict = params; print("OVERWRITING") 

  ts = np.linspace(0,10,1000)
  y0s = [0.3,0,1]
  ys = odeint(dydt,y0s,ts,args=(varDict,))
  #ys = np.zeros([5,5]) 

  data = dict() # This is a nuisance, but keeping backward compatible w Gotran stuff 
  data['t'] = ts          
  data['Cai'] = ys[:,0]
  data['CaB'] = ys[:,1]
  data['Nai'] = ys[:,2]

  returnDict['data'] = data       
  return returnDict
  

