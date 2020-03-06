
#
# Template for simulation to be run on each process 
#
from scipy.integrate import odeint
import numpy as np 

class Runner():
  def __init__(self):
    self.params={
        'kon': 0.1,
        'koff': 0.1,
        'bMax': 5,
        'scale':1}       

  def dydt(self,ys,ts,params):
    kon = params['kon']
    #print(kon)
    koff = params['koff']
    bMax = params['bMax']
    scale = params['scale']
  
    ca,cab,na = ys
    dca_dt = +koff*cab - kon*ca*(bMax-cab)
    dcab_dt = -dca_dt
    dna_dt  = -scale*na
  
    return dca_dt,dcab_dt,dna_dt
    
  def simulate(
    self, 
    varDict=None,        # dictionary of parameters to be used for simulation
    returnDict=dict(),    # dictionary output to be returned by simulation
    jobDuration = 25e3   # [ms]
    ):
    if varDict is None:
      varDict = self.params; 
    #print("OVERWRITING") 
    #varDict['bMax']=5.
    #varDict['scale']=1.
    
    #print("sdf",varDict['kon'])
   
  
    # range is in [s]
    ms_to_s = 1e-3
    ts = np.linspace(0,jobDuration*1e-3,1000)
    print(ts[-1])
    y0s = [0.3,0,1]
    ys = odeint(self.dydt,y0s,ts,args=(varDict,))
    #ys = np.zeros([5,5]) 
  
    data = dict() # This is a nuisance, but keeping backward compatible w Gotran stuff 
    data['t'] = ts          
    data['Cai'] = ys[:,0]
    #print(ys[:,0])
    data['CaB'] = ys[:,1]
    data['Nai'] = ys[:,2]
  
    returnDict['data'] = data       
    return returnDict
    
  
