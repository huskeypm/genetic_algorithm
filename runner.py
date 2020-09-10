"""Template for simulation to be run on each process"""
from scipy.integrate import odeint
import numpy as np 

class Runner():
  """Runner object for interfacing with genetic algorithm"""
  def __init__(self):
    """Initialize a Runner object"""
    self.params={
        "kon": 0.1,
        "koff": 0.1,
        "bMax": 5,
        "scale": 1
    }       

  def dydt(self, ys, ts, params):
    """Return the change in the variables measured in `ys` over time.
    
    This is the function that is passed to ODEINT.

    Parameters
    ----------
    ys : 1D np.ndarray
        The values of the parameters in the current state.
    ts : int or float
        The time point with which to solve this equation.
    params : dict
        Parameter dictionary that has the parameter names as keys and the parameter value as the 
        dictionary's values.

    Returns
    -------
    tuple
        Tuple of (dca_dt, dcab_dt, dna_dt). Generally, returns change in value `i` over time for 
        each `i` in given `ys`.
    """
    kon = params['kon']
    koff = params['koff']
    bMax = params['bMax']
    scale = params['scale']
  
    ca, cab, na = ys
    dca_dt = +(koff * cab) - (kon * ca * (bMax-cab))
    dcab_dt = -dca_dt
    dna_dt  = -(scale * na)
  
    return dca_dt, dcab_dt, dna_dt
    
  def simulate(self, varDict=None, returnDict=dict(), jobDuration=25e3):
    """Run the simulation with given parameters over duration specified in `jobDuration`

    This is the function that is called within the genetic algorithm.

    Parameters
    ----------
    varDict : dict, optional
        Dictionary of parameters to be used for simulation, by default None and uses `self.params`.
    returnDict : dict, optional
        Dictionary output to be returned by simulation, by default empty dictionary.
    jobDuration : float, optional
        Duration of the simulation in [ms], by default 25e3.

    Returns
    -------
    dict
        returnDict populated with the data in `returnDict["data"]`.
    """
    if varDict is None:
      varDict = self.params

    # range is in [s]
    ms_to_s = 1e-3
    ts = np.linspace(0,jobDuration*1e-3,1000)
    y0s = [0.3,0,1]
    ys = odeint(self.dydt,y0s,ts,args=(varDict,))
  
    # Form the `data` dictionary for packaging up all of the simulation information.
    # NOTE: This is a nuisance, but keeping backward compatible with Gotran stuff.
    data = dict()
    data['t'] = ts          
    data['Cai'] = ys[:,0]
    data['CaB'] = ys[:,1]
    data['Nai'] = ys[:,2]
  
    returnDict['data'] = data       
    return returnDict
    
  
