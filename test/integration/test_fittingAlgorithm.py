import os
import sys

import numpy as np

ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.join(ROOT, "..", "..")
sys.path.append(PROJECT_ROOT)

import fittingAlgorithm as fa

def test_smoke_1():
  """A simple smoke test to see if anything is on fire.

  If this fails we know at least SOMETHING is wrong.
  """
  # parameters to vary 
  stddev = 0.2
  variedParamDict = {
    "kon":  [0.5, stddev],
    "koff":  [5.0, stddev]
  }

  # list of observables to be scored by GA 
  outputList = { 
      "Nai":fa.OutputObj(
          "Nai", "val_vs_time", [0, 2],
          [1, 0.5, 0.15],
          timeInterpolations=[0, 1, 2]
      ) # [ms] check that interpolated values at 0, 100, 200 are 1, 0.5 ... 
  }
         
  simulation = fa.runner.Runner()
  results = fa.run(
      simulation,
      yamlVarFile = "inputParams.yaml",
      variedParamDict = variedParamDict,
      jobDuration = 30e3, # [ms]
      numRandomDraws = 8,  
      numIters = 5,    
      outputList = outputList,
      debug = True
  )

  # Ensure that the output is sufficiently accurate.
  accuracy_threshold = 0.4 # empirically derived value
  assert (results["bestFitness"] < accuracy_threshold), "Best fitness is too inaccurate!"

  # Ensure that the parameters have converged to the correct value.
  bestVarDict_truth = {
    "kon": 0.6395284713285316,
    "koff": 6.559989301751445
  }
  bestVarDict_test = results["bestFitDict"]
  for key in bestVarDict_truth.keys():
    assert(np.isclose(bestVarDict_test[key], bestVarDict_truth[key])), (
      "\"{key}\" in bestVarDict did not validate!".format(key=key))
  
def test_smoke_2():
  """Here we try to optimize the sodium buffer to get the correct free Na concentration"""
  testState = "Cai"
  simulation = fa.runner.Runner()
  results = fa.run(
    simulation,
    yamlVarFile = "inputParams.yaml",
    variedParamDict = fa.variedParamListDefault,
    jobDuration = 25e3, # [ms] 
    numRandomDraws = 3,
    numIters = 10,
    sigmaScaleRate = 0.45,
    outputParamName = "Container",
    outputParamSearcher = testState,
    outputParamMethod = "min",
    outputParamTruthVal=0.1,
    debug = True
  )

  refKon = 0.6012
  bestKon = results['bestFitDict']
  bestKon = bestKon['kon']
  assert(np.abs(refKon - bestKon) < 1e-3), "FAIL!"
  print("PASS!!") 