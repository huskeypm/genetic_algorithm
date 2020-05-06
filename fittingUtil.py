"""
Functions for working with data 
"""

# creates interpolator for data 
from scipy.interpolate import interp1d
def InterpolateData(time, data):
  f = interp1d(time,data,fill_value='extrapolate')
  # to use: xnew, f(xnew)
  
  return f

  


