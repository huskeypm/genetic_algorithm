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
yamlVarFile = "inputParams.yaml"
testState = "Cai"         
timeDur = 1e3


recordedData = []

numCells = 2
for i in range(numCells):
  # default parameters for model 
  varDicti = varDictDflt.copy()
  varDicti["kon"]+= 0.2*np.random.randn(1)
  varDicti["koff"]+= 0.2*np.random.randn(1)
  
  # place holder/perform simulation 
  returnDict = dict() # return results vector
  simulation.simulate(varDicti,returnDict,jobDuration = timeDur) 

  ## do output processing
  data = returnDict['data']
  tRef = data['t']*1e3 # turn [s] back to [ms]
  #print(timeDur)
  caiRef = data[testState] + 0.005*np.random.randn( np.shape(tRef)[0] )

  plt.plot(tRef,caiRef)
  recordedDatai = {'t':tRef, 'Cai':caiRef}  
  recordedData.append( recordedDatai )


# ### Now we fit each transient separately 

# In[21]:


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
    simulation = runner.Runner()
    results = fA.run(
        simulation,
        yamlVarFile = "inputParams.yaml",
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

