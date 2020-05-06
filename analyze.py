"""
For processing worker bee outputs
Note: the microglia repo has different analyses in analyzeGotran
      that might be useful, like measuring slope, etc. Don't 
      reinvent that wheel!
"""

import math
#import cPickle as pickle
#import matplotlib.pylab as plt
import numpy as np

class empty:pass


import re

# loads data stored in ode object
# returns a single array with quantity of interest (valsIdx) 
# which is the time series of the idxName  
# and the time (.t)
def GetData(data,idxName):
    #print "getting data" 
    datac = empty()
    datac.valsIdx = data[idxName]
    datac.t = data['t']           

    return datac

### taken from fitter.py/analyzeODE.py, used to process data made to put into panda format at the end.
# Most of the original implementation has been scrapped
from scipy import interpolate
def ProcessDataArray(
      dataSub,
      mode,
      timeRange=[0,1e3],         # [ms]
      timeInterpolations=None,   # [ms] array, will interpolate the values of valueTimeSeries at the provided times
      key=None):

      
      # Time is listed in seconds [ms] 
      # in this case, the t's are in the same units as tsteps 
    #   print('timeRange', timeRange)
      timeSeries = dataSub.t
    #   print('dataSub timeSeries',timeSeries)
      idxMin = (np.abs(timeSeries-timeRange[0])).argmin()  # looks for entry closest to timeRange[i]
    #   print('idxMin', idxMin)
      idxMax = (np.abs(timeSeries-timeRange[1])).argmin()
    #   print('idxMax', idxMax)
      timeSeries = dataSub.t[idxMin:idxMax]
      valueTimeSeries = dataSub.valsIdx[idxMin:idxMax]
      #print "obj.timeRange[0]: ", timeRange[0]
      #print "valueTimeSeries: ", valueTimeSeries
   
      #print "dataSub.valsIdx: ", dataSub.valsIdx 
      if mode == "max":
          result = np.max(valueTimeSeries)
      elif mode == "min":
          result = np.min(valueTimeSeries)
      elif mode == "mean":
          result = np.mean(valueTimeSeries)
      elif mode == "val_vs_time":
          #print('timeInterpolations',timeInterpolations)
          #print('timeSeries',timeSeries)
          #print('valueTimeSeries',valueTimeSeries)
          # interps predicted time values onto time points in expt (truth) data
          #result = np.interp(timeInterpolations,timeSeries,valueTimeSeries,
          #        fill_value='extrapolate') # can be dangerous....
          f = interpolate.interp1d(timeSeries,valueTimeSeries,
                  fill_value='extrapolate') # can be dangerous....
          result = f(timeInterpolations)                                     
          #print("interp", result )   
          #quit()

      else:
          raise RuntimeError("%s is not yet implemented"%output.mode)

      return result
