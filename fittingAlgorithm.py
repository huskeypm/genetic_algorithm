#!/usr/bin/env python3
import sys
import os
#sys.path.append("./fitting_sensitivity/")

import multiprocessing
import random
from os import getpid
import copy

import yaml
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import imp
from fittingUtil import InterpolateData


# USER EDITS THESE
import runner # simulation engine                 
import analyze # analyses specific to runner 
ms_to_s = 1e-3


## Default structure for indicating parameters that need to be randomized
stddev = 2.0
variedParamListDefault= {
  # paramDict[myVariedParam] = [variedParamTruthVal, 0.2] # for log normal
  "kon":  [5.0,stddev],
}

class OutputObj:
    """Default structure for observables to be 'scored' by genetic algorithm"""
    def __init__(self, odeKeyName, mode, timeRange, truthValue, timeInterpolations=None):
      """Initialize an instance of `OutputObj`

      Parameters
      ----------
      name : str
          Name for the measurable (state name as written in the ode file) 
      mode : str
          The type of comparison to be made to the 'truth' data in `truthValue`.
      timeRange : list or 1D np.ndarray
          The time interval during which to assess the measurable [ms].
      truthValue : int/float or 1D np.ndarray
          The truth value(s) for which the measurable will be assessed against.
      timeInterpolations : int/float or 1D np.ndarray, optional
          Where truth value occurs [ms]. If no value is given, assess the measurable at the points
          in the given `timeRange`. This is used if interpolation is necessary. By default None and
          no interpolation is done.
      """
      self.odeKeyName = odeKeyName
      self.mode = mode
      self.timeRange = np.array(timeRange) # [ms], NEED TO ADD
      self.timeInterpolations= np.copy(timeInterpolations)# if ndarray, will interpolate the values of valueTimeSeries at the provided times
      if isinstance(timeInterpolations,np.ndarray):
        1 
      self.truthValue = np.array(truthValue,dtype=np.float)
      self.result = None

## Format:
# Key: OuputObj(state name, metric of comparison, time range over which to compute metric, truth value)
# Note that multiple OutputObjs can be considered (like channels) 
outputListDefault = {
  "Cai": OutputObj(
    "Cai",
    "mean",
    [8,10], # in [s]
    0.1 # value you want 
  ),
  "Nai": OutputObj(
    "Nai",
    "val_vs_time",
    [0, 2],
    [1, 0.5, 0.15],
    timeInterpolations=[0, 1, 2] # check that interpolated values at 0, 100, 200 are 1, 0.5 ... 
  ) 
}

class empty:pass

class Results:
  """Holds result information from job to be compared with experimental data"""
  def __init__(self, outputResults, jobDict, jobNum):
    """Initialize instance of `Results`
    
    Parameters
    ----------
    outputResults : dict
        The output data from the simulation. This can either be processed or not. Of the structure:
            outputResults = {
              "t": <array of time points>
              <for each measurable>
              "<measurable name>": <measurable values at time points>
            }
    jobDict : dict
        The dictionary describing this job.
    jobNum : int
        The number of the job that produced these results.
    """
    self.outputResults = outputResults
    self.pid = getpid()
    self.jobDict = jobDict
    self.jobNum = jobNum

def workerParams(jobDict, skipProcess=False, verbose=False):
    """Executes an input .ode file on each worker process

    Parameters
    ----------
    jobDict : dict
        Dictionary giving the parameters for the job we're running.
    skipProcess : bool, optional
        , by default False.
    verbose : bool, optional
        , by default False.

    Returns
    -------
    tuple
      Tuple of results with (jobNum, results) used for comparison against experiment where:
        + jobNum : int 
          + indicating the number of the job.
        + results : Results
          + The results of the job.
    """
    simulation = jobDict['simulation']  # simulation object
    odeName = jobDict['odeModel']
    jobNum = jobDict['jobNum']
    dtn = jobDict['jobDuration'] # [ms]
    variedParamDict = jobDict['varDict']
    fixedParamDict =jobDict['fixedParamDict']
    if 'tsteps' in jobDict:
      tsteps = jobDict['tsteps']
    else:
      tsteps = None

    print("Worker bee %d, Job %d "%(getpid(),jobNum))

    outputList = jobDict['outputList']

    if outputList == None:
        outputList = outputListDefault
        print("No outputList given, using outputListDefault.") 

    # prune keys from 'fixedParamList' if in variedParamDict already
    if fixedParamDict is not None:
      for key in variedParamDict:
        fixedParamDict.pop(key, None)

    # create new var Dict with all parameters
    varDict = dict()
    for key,val in variedParamDict.items() :
        varDict[key]=val
    if isinstance(fixedParamDict, dict):
      for key,val in fixedParamDict.items() :
        varDict[key]=val
    if verbose:
      print("Running with these varied parameters:")
      for key,val in variedParamDict.items() :
        print("  ",key,val)
        1

    ## create varDict for runParams
    ## Launch job with parameter set
    returnDict = dict() # return results vector
    simulation.simulate(varDict,returnDict,jobDuration = dtn) 

    ## do output processing
    data = returnDict['data']
    if skipProcess:
      outputResults = data
    else:
      outputResults = ProcessWorkerOutputs(data,outputList,tag=jobNum)

    ## package additional useful information
    results = Results(outputResults, jobDict, jobNum)

    return jobNum, results

def ProcessWorkerOutputs(data, outputList, tag=99):
  """Given data dictionary, pulls out subsection of data

  Data subset is evaluated based on 'obj.mode', which defines the type of analysis done.
  See OutputObj class definition and ProcessDataArray function

  Parameters
  ----------
  data : dict
      The output data from the simulation. Of the structure:
          data = {
              "t": <array of time points>
              <for each measurable>
              "<measurable name>": <measurable values at time points>
          }
  outputList : dict
      Dictionary holding the name(s) of the measurable(s) as keys and `OutputObj`(s) as the values.
  tag : int, optional
      The job number for this job, by default 99. NOT CURRENTLY USED.

  Returns
  -------
  dict
      The new, processed, outputResults, where everything except the result is copied from given 
      `outputList`. `outputList[key].result` is the newly processed results.
  """
  outputResults = {}
  for key,obj in outputList.items():
    dataSub = analyze.GetData(data, obj.odeKeyName)

    result = analyze.ProcessDataArray(dataSub,obj.mode,obj.timeRange,obj.timeInterpolations,key=key)

    resultObj = copy.copy(obj)
    resultObj.result = result
    outputResults[key]=resultObj

  return outputResults

def YamlToParamDict(yamlVarFile):
  """Reads YAML file and puts into parameter dictionary

  Parameters
  ----------
  yamlVarFile : str
      File name/path to the YAML file.

  Returns
  -------
  dict
      The parameter dictionary read in, if given a filepath in `yamlVarFile`.
  """
  fixedParamDict = None
  if yamlVarFile is not None:
    with open(yamlVarFile, 'r') as fp:
      fixedParamDict = yaml.load(fp)

    # converting to float since yamml doesnt know science notation
    for key, val in fixedParamDict.items():
      fixedParamDict[key] = np.float(val)
  return fixedParamDict


def PandaData(jobOutputs, csvFile="example.csv"):
  """Stores all data into a pandas object, which simplifies analyses later

  Parameters
  ----------
  jobOutputs : dict
      A dictionary with worker numbers as keys and job objects as values.
  csvFile : str, optional
      [description], by default "example.csv"

  Returns
  -------
  pandas.DataFrame
      The dataframe holding our job outputs.
  """
  masterDict = dict()

  # get dictionary for each job and append it to a 'master' dictionary
  for workerNum, jobObj in jobOutputs.items():
    jobDict = StoreJob(job1= jobObj)
    jobID = jobDict['jobID']
    masterDict[jobID]=jobDict

  # store data in pandas dataframe
  df = pd.DataFrame(masterDict)
  df = df.T
  if csvFile != None:
    df.to_csv(csvFile)
  return df    

# Stores job information into a dict that can be used with pandas
def StoreJob(job1):
    """Stores job information into a dict that can be used with pandas

    Parameters
    ----------
    job1 : dict
        A dictionary holding job information.

    Returns
    -------
    dict
        A dictionary that works well when converting to a pandas.DataFrame object.
    """
    pandasDict = dict()
    tag = "%d_%d"%(job1.jobNum,job1.pid)
    pandasDict['jobID'] = tag
    pandasDict['jobNum'] = job1.jobNum

    # pull out inputs
    varDict = job1.jobDict['varDict']
    for param,value in varDict.items():
        pandasDict[param] = value

    # pull out its results vector
    outputResults = job1.outputResults
    for output,result in outputResults.items():
        pandasDict[output] = result.result

    return pandasDict
# creates cross-overmutations by copying over randommized parameter from another randomly 
# chosen child. Checks to make sure parameters are different  
#
# select 3 at random (assuming 50% of a parameters variations will improve, 50% worsen, 
# Pimprov = Pv1prov*Pv2improv = 0.25
# Pworsen = 1-Pimprove
# Pworsen_Nruns = (1-Pimprove)**N ~ 0.4 for three runs 
def Crossovers(jobList,numCrossOvers=3
  ):

  # randomize indices
  numChildren = len(jobList) 
  idxs = np.arange( numChildren )   
  np.random.shuffle(idxs)
  #print(idxs)

  # apply crossovers 
  numCrossOvers = np.int(np.min([numCrossOvers, numChildren/2]))
  children=[]
  for i in range(numCrossOvers):
    # select paired alleles (idx1 will 'receive' idx2 info) 
    idx1, idx2 = idxs[2*i], idxs[2*i+1]

    child1 = jobList[idx1]
    child2 = jobList[idx2]

    vD1 = child1['varDict']
    vD2 = child2['varDict']
    # overwrites child2-varied param in child1 with child2's value 
    if child1['variedParm'] is not child2['variedParm']:
      children.append(idx1)
      val2 = vD2[ child2['variedParm'] ]
      vD1[ child2['variedParm'] ]  = val2
    else:  # conflict, skip
      1

  # final job List 
  print("%d children crossed over "%len(children), children)

#
#
# Pulled out parameter randomization 
#
def randParams(simulation, jobList, defaultVarDict, fixedParamDict, parmDict, tsteps, 
  numRandomDraws, randomDrawAllIters, iters, sigmaScaleRate, distro, outputList, jobDuration,
  odeModel=None):
  """Pulled out parameter randomization that stores formed `jobDict`s to be run in `jobList`

  Parameters
  ----------
  simulation : [type]
      [description]
  jobList : [type]
      [description]
  defaultVarDict : [type]
      [description]
  fixedParamDict : [type]
      [description]
  parmDict : [type]
      [description]
  tsteps : [type]
      [description]
  numRandomDraws : [type]
      [description]
  randomDrawAllIters : [type]
      [description]
  iters : [type]
      [description]
  sigmaScaleRate : [type]
      [description]
  distro : [type]
      [description]
  outputList : [type]
      [description]
  jobDuration : [type]
      [description]
  odeModel : [type], optional
      [description], by default None
  """
  ctr=0
  ctr = len(jobList)  # should start from last jobList
  for parameter,values in parmDict.items():
  
      ## generate random pertubrations
      # draw from normal distribution
      print("Sigm needs to be specific to each var") 
      mu,sigma = values
      #rescaledSigma = sigma/(sigmaScaleRate * iters)
      rescaledSigma = sigma*np.exp(-sigmaScaleRate * (iters-1))
      #rescaledSigma = sigma
      # distro = "lognormal"
      if distro=="normal":
        randomDraws = np.random.normal(mu,rescaledSigma,numRandomDraws)
      if distro=="lognormal":
        unif = np.random.normal(0,rescaledSigma,numRandomDraws)
        randomDraws = np.exp(unif) * mu
  
      randomDraws = np.sort(randomDraws)
  
      # create a list of jobs
      randomDrawAllIters.append(randomDraws)
      #listN = [{parameter:val,'jobNum':i} for i,val in enumerate(randomDraws)]
      #jobList+=listN
      for val in randomDraws:
          varDict = copy.copy(defaultVarDict)
          varDict[parameter] = val
  
          jobDict =  {
                      'simulation':simulation,
                      'odeModel':odeModel,
                       'varDict':varDict,'fixedParamDict':fixedParamDict,
                      'jobNum':ctr,'jobDuration':jobDuration, 'tsteps':tsteps,
                      'outputList':outputList,
                      'variedParm':parameter}
          jobList.append( jobDict )
          ctr+=1
  
  # now selecting subset via reservoire sampling 
  N = numRandomDraws              

  # something stupid is happing with muiltiple parents/param
  #sample = [];
  #for i,line in enumerate(jobList): 
  #  if i < N:
  #    sample.append(line)
  #  elif i >= N and random.random() < N/float(i+1):
  #    replace = random.randint(0,len(sample)-1)
  #    sample[replace] = line
  #jobList = sample 


  # renumber job num for indexing later 
  for i,line in enumerate(jobList): 
     # old print line['jobNum'] 
     line['jobNum'] = i

  # apply cross overs 
  crossOvers = True
  numCrossOvers=3
  if crossOvers:
    Crossovers(jobList,numCrossOvers)

def fittingAlgorithm(simulation, odeModel, myVariedParamKeys, variedParamDict=None, 
  fixedParamDict=None, numCores=5, numRandomDraws=3, jobDuration=2000, tsteps=None,
  outputList=None, truthValues=None, sigmaScaleRate=1., maxRejectionsAllowed=3, numIters=10,
  distro='lognormal', verbose=2):
  """Genetic algorithm that randomizes params, selects best solution, repeats for given iterations
  
  Genetic algorithm that randomizes the provided parameters (1 for now), selects the solution that 
  minimizes the error, and repeats this process for a given number of iterations.

  Parameters
  ----------
  simulation : [type]
      [description]
  odeModel : [type]
      [description]
  myVariedParamKeys : [type]
      Supports multiple params, hopefully.
  variedParamDict : dict
      The varied parameter dictionary, by default None.
  fixedParamDict : dict, optional
      Input set of fixed parameters/values, by default None.
  numCores : int, optional
      Number of cores over which jobs are run, by default 5.
  numRandomDraws : int, optional
      Number of random draws for each parameter, by default 3.
  jobDuration : int, optional
      Job run time, [ms], by default 2000.
  tsteps : np.linspace, optional
      `np.linspace` of time steps [ms], by default None.
  outputList : [type], optional
      [description], by default None.
  truthValues : [type], optional
      [description],, by default None.
  sigmaScaleRate : float, optional
      Rate at which sigma is reduced by iteration (larger values, faster decay), by default 1.0.
  maxRejectionsAllowed : int, optional
      Number of rejected steps in a row before exiting alg, by default 3.
  numIters : int, optional
      [description], by default 10.
  distro : str, optional
      Distribution with which we select new parameters, by default "lognormal".
  verbose : int, optional
      The verbosity option, by default 2.

  Returns
  -------
  tuple
      Tuple of the following: randomDrawAllIters, bestDrawAllIters, previousFitness

  Raises
  ------
  RuntimeError
      Raises RuntimeError if `numCores` > 1. This is a debugging feature.
  """
  #PKH adding pseudocode for multiple parents
  nParents = 2

  ## Initialize trial param list 
  trialParamVarDict = copy.copy( variedParamDict ) 
  trialParamVarDicts=[]
  for i in range(nParents):
    trialParamVarDicts.append( copy.deepcopy( variedParamDict ) ) 

  iters = 0
  allErrors = []
  errorsGood_array = []
  flag = True
  randomDrawAllIters = []
  bestDrawAllIters = []
  rejection = 0
  previousFitness = 1e9
  while iters < numIters and rejection<maxRejectionsAllowed:  
      ## Create 'master' varDict list
      iters += 1

      numVariedParams = 0

      print("iter", iters, " out of", numIters)

      #PKH need to allocate one per parent 
      #defaultVarDict = dict()
      #parmDict = trialParamVarDict 
      defaultVarDicts = []
      parmDicts = []
      for i in range(nParents):
        defaultVarDicts.append(dict())
        parmDicts.append(trialParamVarDicts[i]) 
        print("p%d"%i, trialParamVarDicts[i])
        if iters < 2:
          break 
      
      #PKH populate one per parent 
      #for parameter,values in parmDict.items():
      #    defaultVarDict[parameter] = values[0]  # default value
      #    print("Inputs: ", parameter, values[0])
      #    numVariedParams+=1
      
      for i in range(len(defaultVarDicts)):
        defaultVarDicti = defaultVarDicts[i]
        parmDicti= parmDicts[i]
        numVariedParams=0
        for parameter,values in parmDicti.items():
          defaultVarDicti[parameter] = values[0]  # default value
          print("Inputs: ", parameter, values[0])
          numVariedParams+=1


      ## determine core count
      totNumRandomDraws = numRandomDraws*numVariedParams
      #numJobs = np.int(numRandomDraws/nParents) # *numParams
      numCores = np.min( [numCores, totNumRandomDraws] ) 
      print("Using %d cores for %d jobs"%(numCores,totNumRandomDraws))

      ## Create a list of jobs with randomized parameters
      # Here we create a much larger job list than we can actually use, so that we can randomly select a subset of which
      # This is mostly important for the multi-variable cases
      jobList = []

      ## randomizes multiple parametrs 
      #PKH do once per parent 
      #randParams(
      #  simulation,
      #  jobList,defaultVarDict,fixedParamDict,parmDict,tsteps,numRandomDraws,randomDrawAllIters,iters,sigmaScaleRate,distro,outputList)
      #print(len(jobList))
      for i in range(len(defaultVarDicts)):
        numRandomDrawsi = np.int(numRandomDraws/nParents)
    
        defaultVarDicti = defaultVarDicts[i]
        parmDicti= parmDicts[i]
        randParams(simulation, jobList, defaultVarDicti, fixedParamDict, parmDicti, tsteps, 
          numRandomDrawsi, randomDrawAllIters, iters, sigmaScaleRate, distro, outputList, 
          jobDuration, odeModel)

      # this value should be numRandomDraws*numParents 
      #print(jobList)
        
    
      ## Run jobs
      if numCores > 1:
          print("Multi-threading")
          with multiprocessing.Pool(processes=numCores) as pool:
              jobOutputs = dict( pool.map(workerParams, jobList) )
      else:
          print("Restricting to one job only/assuming results are all that's needed") 
          jobNum, results = workerParams(jobList[0])
          raise RuntimeError("PKH Needs to fig - give dataframe save" )

      # Shouldn't have to write csv for these
      myDataFrame = PandaData(jobOutputs, csvFile=None)

      jobFitnesses = np.ones( len(myDataFrame.index) ) * -1
      jobNums      = np.ones( len(myDataFrame.index), dtype=int ) * -1
      for i in range(len(myDataFrame.index)):
          # score 'fitnesss' based on the squared error wrt each output parameter
          fitness = 0.0
          for key, obj in outputList.items():
              odeKey = obj.odeKeyName  
              result = myDataFrame.loc[myDataFrame.index[i],odeKey]

              # Decide on scalar vs vector comparisons
              if not isinstance(result, np.ndarray):
                result = np.array( result )

              # sum over squares
              error = np.sum((result - obj.truthValue) ** 2)
              normFactor = np.sum(obj.truthValue ** 2)
              normError = np.sqrt(error/normFactor) 
              if verbose >= 2:
                print("result: ", result, "truthValue: ", obj.truthValue)

              fitness += normError

          # compute sqrt
          jobFitnesses[i] =  np.sqrt(fitness)

          # These lines are intended to correct for a discrepancy between the pandas numbering and the job list
          # It works, but its confusing
          jobNums[i] = myDataFrame.loc[myDataFrame.index[i],'jobNum']
          myDataFrame.loc[myDataFrame.index[i],'fitness'] = jobFitnesses[i]

      #
      # Summarize results
      #
      print("myDataFrame:\n", myDataFrame)
      print("jobFitnesses: ", jobFitnesses)
      # find best job
      #PKH need to sort and collect nParents best 
      pandasSortedIndices = np.argsort( jobFitnesses ) 
      #nprint(pandasSortedIndices[0])
      pandasIndex = np.argmin( jobFitnesses )
      #nprint(pandasSortedIndices[0],pandasIndex)
      pandasIndex = pandasSortedIndices[0]
      jobIndex = jobNums[ pandasIndex ]
      jobIndices =  jobNums[ pandasSortedIndices[0:nParents] ] 
      print("Best jobIndices: ", jobIndices)

      # grab the job 'object' corresponding to that index
      #bestJob = jobList[ jobIndex ]
      currentFitness = jobFitnesses[pandasIndex]
      #print(jobIndices, type(jobList))
      bestJobs = []
      for i,jobIdx in enumerate(jobIndices):
        #print("Dbl fit",i, jobFitnesses[ pandasSortedIndices[i] ] ) 
        bestJobs.append( jobList[ jobIdx ] )    

      if iters == 1:
        previousFitness = currentFitness

      if currentFitness <= previousFitness:
        # get its input params/values
        bestJobi = bestJobs[0]
        bestVarDict = bestJobi[ 'varDict' ]
        print("bestVarDict: " , bestVarDict)
        print("currentFitness", currentFitness)
        previousFitness = currentFitness
        rejection = 0

        #variedParamVal = bestVarDict[ myVariedParamKey ]
        #bestDrawAllIters.append(variedParamVal)

        # update 'trialParamDict' with new values, [0] represents mean value of paramater
        #PKH do for each parent 
        #for myVariedParamKey, variedParamVal in bestVarDict.items():
        #  trialParamVarDict[ myVariedParamKey ][0]  = variedParamVal
          # [1] to represent updating stdDev value
          # trialParamVarDict[ myVariedParam ][1]  = variedStdDevVal

        #for i in range(nParents):
        for i in range(len(defaultVarDicts)):
          #print("iter") 
          #print(trialParamVarDicts)
          trialParamVarDicti = trialParamVarDicts[i] 
          bestJobi = bestJobs[i] 
          bestVarDicti= bestJobi[ 'varDict' ]
          #defaultVarDicti = defaultVarDicts[i]
          #parmDicti= parmDicts[i]
          for myVariedParamKey, variedParamVal in bestVarDicti.items():
            trialParamVarDicti[ myVariedParamKey ][0]  = variedParamVal

          print("Parent rank %d"%i,trialParamVarDicti)          

      else:
        print("Old draw is better starting point, not overwriting starting point") 
        rejection+=1
        print("Rejected %d in a row (of %d) "%(rejection,maxRejectionsAllowed) )

      bestJobi = bestJobs[0]
      bestDrawAllIters.append( bestJobi[ 'varDict' ] ) 

      print("iter", iters, " out of", numIters)
      print("")
      print("######")
      print("")

  return randomDrawAllIters, bestDrawAllIters, previousFitness

def run(simulation, odeModel=None, myVariedParam=None, variedParamTruthVal=5.0, 
  variedParamDict=None, timeStart=0, jobDuration=30e3, tsteps=None, fileName=None,
  numRandomDraws=5, numIters=3, sigmaScaleRate=0.15, outputList=None, outputParamName="Nai",
  outputParamSearcher="Nai", outputParamMethod="mean", outputParamTruthTimes=None, 
  outputParamTruthVal=12.0e-3, maxCores=30, yamlVarFile=None, outputYamlFile=None, 
  debug=False, fixedParamDict=None, verboseLevel=2, distro='lognormal', output_dir="."):
  """Run the genetic algorithm

  This is the one you should mostly interface with.
    (see test/integration/test_fittingAlgorithm.validation()) 

  Parameters
  ----------
  simulation : [type]
      [description]
  odeModel : [type], optional
      "shannon_2004_rat.ode", NOTE NEEDS TO BE ANTIQUATED, by default None.
  myVariedParam: [type], optional
      [description], by default None.
  variedParamTruthVal : int, optional
      [description], by default 5.0.
  variedParamDict : dict, optional
      [description], by default None.
  timeStart : int, optional
      [ms] discard data before this time point, by default 0.
  jobDuration : float, optional
      [ms] simulation length, by default 30e3.
  tsteps : [type], optional
      Can input nonuniform times (non uniform linspace), by default None.
  fileName : str, optional
      [description], by default None.
  numRandomDraws : int, optional
      [description], by default 5.
  numIters : int, optional
      Number of iterations the genetic algorithm will perform, by default 3.
  sigmaScaleRate : float, optional
      The rate at which sigma is scaled every iteration, the larger the more it is scale.
      By default 0.15.
  outputList : [type]
      Instead of manually passing in output param, comparison method etc, define list here (see 
      default above), by default None.
  outputParamName : str, optional
      General name for objective object, by default "Nai". If `outputList` is not specified, this is
      the name of the single-variable output formed.
  outputParamSearcher : str, optional
      Name of index in return array, by default "Nai". Ensure that this is the same as 
      `outputParamName`.
  outputParamMethod : str, optional
      [description], by default "mean".
  outputParamTruthTimes : [type], optional
      Time points ([ms]) at which to interpolate predicted values. Used where TruthVal is an 
      array, by default None.
  outputParamTruthVal : float or 1D np.ndarray, optional
      [description], by default 12.0e-3.
  maxCores : int, optional
      The maximum number of cores to use. The algorithm will not attempt to use more cores than
      your workstation has, but specifying this option can free up other cores of your workstation
      to handle tasks outside of the genetic algorithm. By default 30.
  yamlVarFile : str, optional
      [description], by default None.
  outputYamlFile : str, optional
      The name of the YAML file to store results, by default None.
  debug : bool, optional
      Whether or not to start in debug mode, by default False.
  fixedParamDict : [type]
      In case fixedParamDict s.b. passed in, by default None.
  verboseLevel : int, optional
      The verbosity level of the output. 2 to show everything, 1 to show a bit, by default 2.
  distro : str, optional
      Distribution with which we select new parameters, by default 'lognormal'.
  output_dir : str
      The path to the directory where figures/output will be saved.

  Returns
  -------
  dict:
      The results dictionary.
  """
  # Check inputs  
  if myVariedParam is None and variedParamDict is None:
    raise RuntimeError("Must define either myVariedParam or variedParamDict") 
  elif myVariedParam is not None and variedParamDict is not None:
    raise RuntimeError("Cannot define BOTH myVariedParam and variedParamDict") 
  ## Define parameter, its mean starting value and the starting std dev
  elif myVariedParam is not None:
    variedParamDict = {myVariedParam:[variedParamTruthVal, 0.2]} # for log normal

  # open yaml file with variables needed for sim
  if fixedParamDict is None:
    fixedParamDict = YamlToParamDict(yamlVarFile)

  # debug mode
  if debug:
    print(
"""
WARNING: In debug mode.
Fixing random seed
"""
    )
    np.random.seed(10)
    random.seed(10)

  # Data analyzed over this range
  if tsteps is None: 
    timeRange = [timeStart * ms_to_s, jobDuration * ms_to_s] # [s] range for data (It's because of the way GetData rescales the time series)
  else: 
    timeRange =[timeStart, tsteps[-1]]


  ## Define the observables and the truth value
  if outputList is None: 
    print("Generating single-variable output list") 
    outputList = {
      outputParamName:OutputObj(
        outputParamSearcher,
        outputParamMethod,
        timeRange,
        outputParamTruthVal,
       timeInterpolations= outputParamTruthTimes)
    }


  # Run
  numJobs_tot = numRandomDraws * len(variedParamDict.keys())
  numCores = np.min([numJobs_tot, maxCores])
  results = trial(simulation, odeModel=odeModel, variedParamDict=variedParamDict,
                  outputList=outputList, fixedParamDict=fixedParamDict,
                  numCores=numCores, numRandomDraws=numRandomDraws,
                  jobDuration=jobDuration, tsteps=tsteps, numIters=numIters,
                  sigmaScaleRate=sigmaScaleRate, fileName=fileName,distro=distro,
                  verbose=verboseLevel, output_dir=output_dir)

  if outputYamlFile is not None:
    OutputOptimizedParams(results['bestFitDict'],originalYamlFile=yamlVarFile,outputYamlFile=outputYamlFile)

  return results

"""
The genetic algorithm
"""
def trial(simulation, odeModel, variedParamDict, outputList, fixedParamDict=None, numCores=2, 
  numRandomDraws=2, jobDuration=4e3, tsteps=None, numIters=2, sigmaScaleRate=1.0, fileName=None,
  distro='lognormal', verbose=2, output_dir="."):
  """The genetic algorithm

  Parameters
  ----------
  simulation : [type]
      [description]
  odeModel : [type]
      [description]
  variedParamDict : [type]
      [description]
  outputList : [type]
      [description]
  fixedParamDict : dict, optional
      Dictionary of ode file parameters/values, which are not randomized, by default None.
  numCores : int, optional
      Maximum number of processors used at a time, by default 2.
  numRandomDraws : int, optional
      Number of random draws for parameters list in `paramDict`. NOTE: `paramDict` should probably 
      be passed in if using this. By default 2.
  jobDuration : float, optional
      Simulation length [ms], by default 4e3.
  tsteps : [type], optional
      Optional time steps [ms], by default None.
  numIters : int, optional
      Number of iterations for the genetic algorithm to perform, by default 2.
  sigmaScaleRate : float
      The rate at which sigma is scaled every iteration, the larger the more it is scale. By default
      1.0
  fileName : str
      [description]
  distro : str
      Distribution with which we select new parameters, by default "lognormal".
  verbose : int
      The verbosity option, by default 2.
  output_dir : str
      The path to the directory where figures/output will be saved.

  Returns
  -------
  dict
      The results dictionary.
  """
  odeModel = None 

  print("WHY is this wrapper needed") 
  # get varied parameter (should only be one for now)
  keys = [key for key in variedParamDict.keys()]

  ## do fitting and get back debugging details
  allDraws,bestDraws,fitness = fittingAlgorithm(
    simulation,
    odeModel,keys, variedParamDict=variedParamDict,fixedParamDict=fixedParamDict,
      numCores=numCores, numRandomDraws=numRandomDraws, 
      jobDuration=jobDuration, tsteps=tsteps,
      outputList=outputList,numIters=numIters, sigmaScaleRate=sigmaScaleRate,
      distro=distro,
      verbose=verbose)
  bestFitDict =  bestDraws[-1]
  print("Best fit parameters", bestFitDict) 

  ## plot performance
  if fileName is not None:
    print("WARNING: skipping Plotting of debuggin data, since current plotter broken. need to adapt to plotting dictionary values instead of a single value, since varying multiple params" )
    1
    #PlotDebuggingData(allDraws,bestDraws,numIters,numRandomDraws,title="Varied param %s"%variedParamKey,fileName=fileName)
  else:
    print("Leaving!!")
    1

  results ={
    'outputList': outputList,
    'allDraws': allDraws,
    'bestDraws': bestDraws,
    'bestFitDict': bestFitDict,  
    'bestFitness': fitness
  }

  ## do a demorun with single worker to demonstrate new fit
  results['fixedParamDict'] = fixedParamDict
  results['data']  = DisplayFit(
    simulation,
    odeModel, jobDuration=jobDuration,tsteps=tsteps,fixedParamDict=fixedParamDict,results=results,
    output_dir=output_dir)

  return results

def DisplayFit(simulation,
         odeModel=None, 
         jobDuration=30e3,tsteps=None,
         fixedParamDict=None,
         results=None,
         output_dir="."):           
  print("Running demo with new parameters for comparison against truth" )

  # run job with best parameters
  outputList = results['outputList']
  varDict = results['bestFitDict'] # {variedParamKey: results['bestFitParam']}
  jobDict =  {
               'simulation':simulation,
               'odeModel':odeModel,'varDict':varDict,'fixedParamDict':fixedParamDict,'jobNum':0,
              'jobDuration':jobDuration, 'tsteps':tsteps,
               'outputList':results['outputList']}
  dummy, workerResults = workerParams(jobDict,skipProcess=True, verbose=True)

  # cludgy way of plotting result
  for key in results['outputList'].keys():   
    1
  #key = outputList.keys()[0]
  obj= outputList[key]
  testStateName = obj.odeKeyName
  data = workerResults.outputResults
  dataSub = analyze.GetData(data,testStateName)

  plt.figure()
  ts = dataSub.t
  plt.plot(ts,dataSub.valsIdx,label="pred")
  #print("SDF",obj.truthValue)
  #print(obj.timeInterpolations) 
  #print( isinstance( None, np.ndarray ) ) # obj.timeInterpolations,np.ndarray))
  #if isinstance( obj.timeInterpolations,np.ndarray):
  #print(np.size(obj.timeInterpolations))

  if np.size(obj.timeInterpolations) > 1: 
    plt.scatter(obj.timeInterpolations,obj.truthValue,label="truth")
  else:
    plt.plot([np.min(ts),np.max(ts)],[obj.truthValue,obj.truthValue],'r--',label="truth")

  plt.title(testStateName)
  plt.legend(loc=0)
  file_path = os.path.join(output_dir, testStateName + ".png")
  plt.gcf().savefig(file_path)


  return data

def PlotDebuggingData(allDraws,bestDraws,numIters,numRandomDraws,title=None,fileName=None):
  # put into array form
  allDraws = np.asarray(allDraws)
  bestDraws = np.asarray(bestDraws)

  # create a matrix of random draws versus iteration
  vals= np.ndarray.flatten(allDraws)
  iters = np.repeat([np.arange(numIters)],numRandomDraws)
  scatteredData= np.asarray(zip(iters,vals))

  veryBest = bestDraws[-1]
  norm_by = 1/veryBest

  
  plt.scatter(scatteredData[:,0], norm_by*scatteredData[:,1],label="draws")
  plt.plot(np.arange(numIters), norm_by*bestDraws, label="best")
  plt.legend()
  if title!= None:
    plt.title(title)

  plt.xlabel("number of iterations")
  plt.xlim([0,numIters])
  plt.ylim([0,np.max(norm_by*scatteredData[:,1])])

  plt.ylabel("value (relative to best %e)"%veryBest)

  if fileName == None:
    plt.gcf().savefig("mytest.png")
  else:
    plt.gcf().savefig(fileName)

# Stores output files into new yaml file 
def OutputOptimizedParams(
  bestFitDict,
  originalYamlFile = None,
  outputYamlFile = "newparams.yaml"
  ):

  # load old 
  paramDict = YamlToParamDict(originalYamlFile)
  #print(paramDict)
 
  # Overwrite
  for key in bestFitDict:
    paramDict[key] = bestFitDict[key]

  #print(paramDict)

  # save as yaml 
  with open(outputYamlFile, 'w') as outfile:
    yaml.dump(paramDict, outfile, default_flow_style=False)
  print("Saved new outputfile with optimized parameters")

##################################
#
# Revisions
#       10.08.10 inception
#
##################################

def helpmsg():
  """Message printed when program run without arguments"""
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
  msg = helpmsg()
  remap = "none"

  odeModel="shannon_2004_rat.ode"
  yamlVarFile = None
  myVariedParam="I_NaK_max"
  variedParamTruthVal=5.0
  jobDuration= 30e3 # [ms] simulation length
  fileName=None
  numRandomDraws=3
  numIters=3
  sigmaScaleRate=0.15
  outputParamName="Nai"
  outputParamSearcher="Nai"
  outputParamMethod="mean"
  outputParamTruthVal=12.0e-3
  timeStart = 0
  debug = False
  runGA = False

  #fileIn= sys.argv[1]
  #if(len(sys.argv)==3):
  #  1

  # Loops over each argument in the command line
  for i,arg in enumerate(sys.argv):
    #if(arg=="-odeModel"):
    #  odeModel = sys.argv[i+1]

    if(arg=="-myVariedParam"):
      myVariedParam = sys.argv[i+1]

    elif(arg=="-variedParamTruthVal"):
      variedParamTruthVal = np.float(sys.argv[i+1])

    elif(arg=="-jobDuration"):
      jobDuration = np.float(sys.argv[i+1])

    elif(arg=="-fileName"):
      fileName = sys.argv[i+1]

    elif(arg=="-numRandomDraws"):
      numRandomDraws = np.int(sys.argv[i+1])

    elif(arg=="-numIters"):
      numIters = np.int(sys.argv[i+1])

    elif(arg=="-sigmaScaleRate"):
      sigmaScaleRate = np.float(sys.argv[i+1])

    elif(arg=="-outputParamName"):
      outputParamName = sys.argv[i+1]

    elif(arg=="-outputParamSearcher"):
        outputParamSearcher = sys.argv[i+1]

    elif(arg=="-outputParamMethod"):
        outputParamMethod = sys.argv[i+1]

    elif(arg=="-outputParamTruthVal"):
        outputParamTruthVal = np.float(sys.argv[i+1])
    elif(arg=="-timeStart"):
        timeStart=np.float(sys.argv[i+1])
    elif(arg=="-fixedvars"):
      yamlVarFile = sys.argv[i+1]
    elif(arg=="-debug"):
      debug = True
    elif(arg=="-run"): 
      runGA = True
 
  if runGA: 
    simulation = runner.Runner()
    run(
      simulation,
      odeModel=odeModel,
      yamlVarFile = yamlVarFile,
      myVariedParam=myVariedParam,
      variedParamTruthVal=variedParamTruthVal,
      timeStart = timeStart,
      jobDuration=jobDuration,
      fileName=fileName,
      numRandomDraws=numRandomDraws,
      numIters=numIters,
      sigmaScaleRate=sigmaScaleRate,
      outputParamName=outputParamName,
      outputParamSearcher=outputParamSearcher,
      outputParamMethod=outputParamMethod,
      outputParamTruthVal=outputParamTruthVal,
      debug = debug )
  else:
    print("Add -run to command line") 



  #raise RuntimeError("Arguments not understood")

#### python fittingAlgorithm.py -odeModel shannon_2004_rat.ode -myVariedParam I_NaK_max -variedParamTruthVal 5.0 -jobDuration 30e3 -fileName This_Is_A_Test.png -numRandomDraws 3 -numIters 3 -sigmaScaleRate 0.15 -outputParamName Nai -outputParamSearcher Nai -outputParamMethod mean -outputParamTruthVal 12.0e-3 &
