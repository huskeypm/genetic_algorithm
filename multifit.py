"""
Prototype for automated fitting 
"""
import numpy as np
import matplotlib.pylab as plt

import fittingAlgorithm as fA


# ### Generte input data
# This is where measured data from microscope will go. For now I am rolling the dice with system parameters and generating several data sets for the algorithm to train on 

# In[17]:


import runner
from fittingAlgorithm import OutputObj

simulation = runner.Runner()
varDictDflt = simulation.params.copy()
testState = "Cai"         
timeDur = 1e3
numCells = 2

# generates data for multiple runs of simulator 
def GenerateData(
        simulation = None,
        yamlFile = None
        ): 

  if simulation is None:
      import runner 
      simulation = runner.Runner()

  recordedData = []
  
  for i in range(numCells):
    # default parameters for model 
    varDicti = varDictDflt.copy()
    varDicti["kon"]+= 0.2*np.random.randn(1)
    varDicti["koff"]+= 0.2*np.random.randn(1)
    
    # place holder/perform simulation 
    returnDict = dict() # return results vector
    print("WARNING: should also pass in yamlfile here too") 
    simulation.simulate(varDicti,returnDict,jobDuration = timeDur) 
  
    ## do output processing
    data = returnDict['data']
    tRef = data['t']*1e3 # turn [s] back to [ms]
    #print(timeDur)
    caiRef = data[testState] + 0.005*np.random.randn( np.shape(tRef)[0] )
  
    plt.plot(tRef,caiRef)
    recordedDatai = {'t':tRef, 'Cai':caiRef}  
    recordedData.append( recordedDatai )

  return recordedData
  
  
# ### Now we fit each transient separately 

# In[21]:


def FitData(
        recordedData,
        simulation=None,      
        yamlFile = None
        ): 

  if simulation is None:
      import runner 
      simulation = runner.Runner()
  import analyze
  
  #cellNum = 0
  #recordedDatai = recordedData[ cellNum ]
  numRandomDraws = 10
  numIters = 5
  numRandomDraws = 2
  numIters = 2
  
  for cellNum in range(numCells):
      recordedDatai = recordedData[ cellNum ]
      
      # parameters to vary/initial guesses
      variedParamDict = {
          "kon":  [0.5,0.2],         
          "koff":  [1.5,0.2],         
      }
      timeRange = [0,9] # in [s]
      
      # what we are fitting to 
      outputList= { 
      #"Cai":OutputObj("Cai","val_vs_time",[  0, 2],
      #[1,0.5,0.15],timeInterpolations=[  0,1,2]) # check that interpolated values at 0, 100, 200 are 1, 0.5 ...
          "Cai":OutputObj("Cai","val_vs_time",timeRange,
                          recordedDatai['Cai'],
                          timeInterpolations=recordedDatai['t']) # check that interpolated values at 0, 100, 200 are 1, 0.5 ...
      }        
  
      # actual fitting process 
      #simulation = runner.Runner()
      results = fA.run(
          simulation,
          yamlVarFile = yamlFile,                 
          variedParamDict = variedParamDict,
          jobDuration = timeDur, # [ms]
          numRandomDraws = numRandomDraws,  
          numIters = numIters,    
          #sigmaScaleRate = 0.45,
          outputList = outputList,
          debug = False,
          verboseLevel=1
      )
      
      # store results from fitting 
      recordedDatai['bestFitDict']  = results['bestFitDict']
      fits = results['bestFitDict']
      
      #print(fits)  
      plt.figure()
      testState="Cai"
      data = analyze.GetData(results['data'],testState)
      #ts = analyze.GetData(results['data'],'t')
      plt.plot(data.t,data.valsIdx,label="fit"+str(fits))#str(fits))
      ts = recordedDatai['t']*1e-3
      cai = recordedDatai['Cai']
      plt.plot(ts,cai,label="input")
      title=testState+"%d"%cellNum
      plt.title(title)
      plt.legend()
      plt.gcf().savefig(title+".png")

def doit():      
  import runner 
  simulation = runner.Runner()
  yamlFile = "inputParams.yaml"

  data = GenerateData(simulation,yamlFile)
  FitData(data,simulation,yamlFile) 

#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################

def validation():
    raise RuntimeError("Need to add") 


#
# Message printed when program run without arguments 
#
def helpmsg():
  scriptName= sys.argv[0]
  msg="""
Purpose: 
 
Usage:
"""
  msg+="  %s -validation" % (scriptName)
  msg+="""
  
 
Notes:

"""
  return msg

#
# MAIN routine executed when launching this script from command line 
#
if __name__ == "__main__":
  import sys
  msg = helpmsg()
  remap = "none"

  if len(sys.argv) < 2:
      raise RuntimeError(msg)

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1
  #  #print "arg"

  # Loops over each argument in the command line 
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-test"):
      #arg1=sys.argv[i+1] 
      doit()#(arg1)
      quit()
  





  raise RuntimeError("Arguments not understood")




