#!/usr/bin/env python3
import sys
#sys.path.append("./fitting_sensitivity/")

import multiprocessing
import random
from os import getpid
import numpy as np
import copy
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

## Default structure for observables to be 'scored' by genetic algorithm
class OutputObj:
    #def __init__(self,name,mode):
    def __init__(self,
            name, # Name for measurable
            mode, # type of comparison [mean, etc]
            timeRange, # time interval during which to assess measurable
            truthValue, # scalar/array 'truth' values
            timeInterpolations= None ): # scalar/array where truth value occurs [?? ms ??] 

      self.name = name
      self.mode = mode
      self.timeRange = np.array(timeRange) #[5e4,10e4]  # NEED TO ADD
      self.timeInterpolations= np.copy(timeInterpolations)# if ndarray, will interpolate the values of valueTimeSeries at the provided times
      if isinstance(timeInterpolations,np.ndarray):
        self.timeInterpolations*=ms_to_s
      self.truthValue = np.array(truthValue,dtype=np.float)
      self.result = None

#outputs = ["Cai","Nai"]
#outputListDefault = { "Nai":OutputObj("Nai","mean"),
#                      "Cai":OutputObj("Cai","max")}

## Format:
# Key: state name, metric of comparison, time range over which to compute metric, truth value
outputListDefault = { "Cai":OutputObj("Cai","mean",[8,10], # in [s]
                       0.1),          # value you want 
                      "Nai":OutputObj("Nai","val_vs_time",[  0, 2],
                      [1,0.5,0.15],timeInterpolations=[  0,1,2]) # check that interpolated values at 0, 100, 200 are 1, 0.5 ... 
                    }

class empty:pass

"""
Executes an input .ode file on each worker process.
Returns result that is used for comparison against expt
"""
def workerParams(
    jobDict,
    skipProcess=False,
    verbose = False): 

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

    ###CMTprint "varDict: ", varDict

    outputList = jobDict['outputList']
    #print("outputList: ", outputList)
    ###CMTprint "outputListDefault: ", outputListDefault
    if outputList == None:
        outputList = outputListDefault
        print("No outputList given, using outputListDefault.") 

    # prune keys from 'fixedParamList' if in variedParamDict already
    if fixedParamDict is not None:
      for key in variedParamDict:
        fixedParamDict.pop(key, None)

    #print(fixedParamDict)

    # create new var Dict with all parameters
    varDict = dict()
    for key,val in variedParamDict.items() :
        varDict[key]=val
    if isinstance(fixedParamDict, dict):
      for key,val in fixedParamDict.items() :
        varDict[key]=val
    if verbose:
      #for key,val in varDict.iteritems() :
      print("Running with these varied parameters:")
      for key,val in variedParamDict.items() :
        print("  ",key,val)
        1

    #quit()

    ## create varDict for runParams
    ###CMTprint "before runParamsFast"
    ## Launch job with parameter set
    returnDict = dict() # return results vector
    simulation.simulate(varDict,returnDict,jobDuration = dtn) 

    ###CMTprint "after runParamsFast"

    ## do output processing
    data = returnDict['data']
    ###CMTprint "DATA: ", data
    if skipProcess:
      outputResults = data
    else:
      # print('data',data)
      # print('outputList',outputList)
      outputResults = ProcessWorkerOutputs(data,outputList,tag=jobNum)
    #if verbose:
    #  for key,val in outputResults.iteritems() :
    #    print "  ",key,val.result

    ## package additional useful information
    results = empty()
    results.outputResults = outputResults
    results.pid = getpid()
    results.jobDict = jobDict
    results.jobNum = jobNum


    return jobNum,results

"""
Given data dictionary, pulls out subsection of data
Data subset is evaluate based on 'obj.mode', which defines the type of analysis done.
See OutputObj class definition and ProcessDataArray function
"""
def ProcessWorkerOutputs(data,outputList, tag=99):
  outputResults = {}
  ###CMTprint "outputList: ", outputList
  for key,obj in outputList.items():
    ###CMTprint "key: ", key, "obj: ", obj
    ###CMTprint "outputList: ", outputList
    ###CMTprint "in the for loop"
    ###CMTprint "obj.timeRange: ", obj.timeRange
    dataSub = analyze.GetData(data, obj.name)
    #print('dataSub', dataSub)

    ###CMTprint "dataSub: ", dataSub
    ###CMTprint "dataSub.valsIdx: ", dataSub.valsIdx
    ###CMTprint "damax",np.max(dataSub.t)
    result = analyze.ProcessDataArray(dataSub,obj.mode,obj.timeRange,obj.timeInterpolations,key=key)

    #output.result = result
    resultObj = copy.copy(obj)
    resultObj.result = result
    #outputResults.append( resultObj )
    outputResults[key]=resultObj

  return outputResults



#
# Reads yaml file and puts into parameter dictionary 
# 
import yaml
def YamlToParamDict(yamlVarFile):
  fixedParamDict = None
  if yamlVarFile is not None:
    with open(yamlVarFile ) as fp:
      fixedParamDict = yaml.load(fp)
      #varDict[key] = np.float( val )
    # converting to float since yamml doesnt know science notation
    for key, val in fixedParamDict.items():
      fixedParamDict[key] = np.float(val)
      #print(key, type(val))
      #print(key, type(fixedParamDict[key]), fixedParamDict[key])
  return fixedParamDict


#
# stores all data into a pandas object, which simplifies analyses later
#
def PandaData(jobOutputs,csvFile="example.csv"):
  masterDict = dict()

  # get dictionary for each job and append it to a 'master' dictionary
  for workerNum, jobObj in jobOutputs.items():
    jobDict = StoreJob(job1= jobObj)
    jobID = jobDict['jobID']
    masterDict[jobID]=jobDict


  # store data in pandas dataframe
  df = pd.DataFrame(masterDict)
  df = df.T
  if csvFile!=None:
    df.to_csv(csvFile)
  return df    


#def PandaDataOLD(jobOutputs,csvFile="example.csv"):
#  raise RuntimeError("Not using")
#  masterDict = dict()
#
#  # get dictionary for each job and append it to a 'master' dictionary
#  for workerNum, jobObj in jobOutputs.iteritems():
#    jobDict = StoreJob(job1= jobObj)
#    jobID = jobDict['jobID']
#    masterDict[jobID]=jobDict
#
#
#  # store data in pandas dataframe
#  df = pd.DataFrame(masterDict)
#  df = df.T
#  df.to_csv(csvFile)
#  return df

# Stores job information into a dict that can be used with pandas
def StoreJob(job1):
    pandasDict = dict()
    tag = "%d_%d"%(job1.jobNum,job1.pid)
    pandasDict['jobID']=tag
    pandasDict['jobNum']=job1.jobNum

    # pull out inputs
    varDict = job1.jobDict['varDict']
    for param,value in varDict.items():
        ###CMTprint param, value
        pandasDict[param] = value

    # pull out its results vector
    outputResults = job1.outputResults
    for output,result in outputResults.items():
        ###CMTprint output, result.result
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
    idx1,idx2 = idxs[2*i], idxs[2*i+1]
    #print("#############") 
    #print(idx1,idx2) 

    child1 = jobList[idx1]
    child2 = jobList[idx2]
    #print(child1) 
    #print(child2) 
    #print(child1['variedParm']) 
    #print(child2['variedParm']) 
    vD1 = child1['varDict']
    vD2 = child2['varDict']
    # overwrites child2-varied param in child1 with child2's value 
    if child1['variedParm'] is not child2['variedParm']:
      children.append(idx1)
      val2 = vD2[ child2['variedParm'] ]
      #print(val2)
      vD1[ child2['variedParm'] ]  = val2
    else:  # conflict, skip
      #print("CLASH - skip") 
      1
    #print("New child",child1) 

  # final job List 
  print("%d children crossed over "%len(children), children)
  #print("Final list",jobList)

#
#
# Pulled out parameter randomization 
#
def randParams(
  simulation,
  jobList,defaultVarDict,fixedParamDict,parmDict,tsteps,numRandomDraws,randomDrawAllIters,iters,sigmaScaleRate,distro,outputList,
  jobDuration,
  odeModel=None): 
  ctr=0
  ctr = len(jobList)  # should start from last jobList
  #print("ctr",ctr) 
  for parameter,values in parmDict.items():
  
      ## generate random pertubrations
      # draw from normal distribution
      print("Sigm needs to be specific to each var") 
      mu,sigma = values
      ###CMTprint "sigma: ", sigma
      #rescaledSigma = sigma/(sigmaScaleRate * iters)
      rescaledSigma = sigma*np.exp(-sigmaScaleRate * (iters-1))
      ###CMTprint "rescaledSigma: ", rescaledSigma, " rate ", sigmaScaleRate
      #rescaledSigma = sigma
      # distro = "lognormal"
      if distro=="normal":
        randomDraws = np.random.normal(mu,rescaledSigma,numRandomDraws)
      if distro=="lognormal":
        unif = np.random.normal(0,rescaledSigma,numRandomDraws)
        randomDraws = np.exp(unif) * mu
  
      randomDraws = np.sort(randomDraws)
  
      # create a list of jobs
      ###CMTprint parameter, " random draws:"
      ###CMTprint randomDraws
      randomDrawAllIters.append(randomDraws)
      #listN = [{parameter:val,'jobNum':i} for i,val in enumerate(randomDraws)]
      #jobList+=listN
      ###CMTprint "JobList: ", jobList
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
          #print(ctr)
          ctr+=1
          ###CMTprint "JobList2: ", jobList
  
  #print(jobList)
  #for i, v in enumerate(jobList):
  #  print("x", v['jobNum'])
  # now selecting subset via reservoire sampling 
  N = numRandomDraws              

  # something stupid is happing with muiltiple parents/param
  #sample = [];
  #for i,line in enumerate(jobList): 
  #  ###CMTprint line['jobNum']
  #  if i < N:
  #    sample.append(line)
  #  elif i >= N and random.random() < N/float(i+1):
  #    replace = random.randint(0,len(sample)-1)
  #    sample[replace] = line
  #jobList = sample 

  ###CMTprint "Print new" 
  # renumber job num for indexing later 
  for i,line in enumerate(jobList): 
     # old print line['jobNum'] 
     line['jobNum'] = i



  # apply cross overs 
  crossOvers = True
  numCrossOvers=3
  if crossOvers:
    Crossovers(jobList,numCrossOvers)


#
# Genetic algorithm that randomizes the provided parameters (1 for now), selects the solution that minimizes the error, and repeats this process for a given number of iterations
#
def fittingAlgorithm(
  simulation,
  odeModel,
  myVariedParamKeys, # Supports multiple params, hopefully 
  variedParamDict = None,
  fixedParamDict=None, # optional, input set of fixed parameters/values
  numCores=5,  # number of cores over which jobs are run
  numRandomDraws=3,  # number of random draws for each parameter
  jobDuration = 2000, # job run time, [ms]
  tsteps = None, # linspace of time steps (optional) [ms]  
  outputList = None,
  truthValues = None,
  sigmaScaleRate = 1., # rate at which sigma is reduced by iteration (larger values, faster decay) 
  maxRejectionsAllowed=3,  # number of rejected steps in a row before exiting alg. 
  numIters = 10,
  distro = 'lognormal', # distribution with which we select new parameters
  verbose=2
  ):

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
      #print("Start") 
      #print(trialParamVarDicts)

      ## Create 'master' varDict list
      iters += 1

      numVariedParams = 0


      print("iter", iters, " out of", numIters)
      ###CMTprint "parmDict: " , parmDict

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

      ###CMTprint "outputList: ", outputList
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
        randParams(
          simulation,
          jobList,defaultVarDicti,fixedParamDict,parmDicti,tsteps,numRandomDrawsi,randomDrawAllIters,iters,sigmaScaleRate,distro,outputList,jobDuration,odeModel)
        #print("njobs",len(jobList))
      # this value should be numRandomDraws*numParents 
      #print(jobList)
        
    
      ## Run jobs
      if numCores > 1:
          print("Multi-threading")
          pool = multiprocessing.Pool(processes = numCores)
          # print('jobList',jobList)
          # for i,job in enumerate(jobList):
          #   print('jobList',i, job)
          jobOutputs = dict( pool.map(workerParams, jobList))#, outputList ) )
      else:
          print("Restricting to one job only/assuming results are all that's needed") 
          jobNum, results = workerParams(jobList[0])
          raise RuntimeError("PKH Needs to fig - give dataframe save" )

      # Shouldn't have to write csv for these
      #myDataFrame = fitter.PandaData(jobOutputs,csvFile=None) # "example.csv")
      myDataFrame = PandaData(jobOutputs,csvFile=None) # "example.csv")

      #allErrors.append([])
      #errorsGood_array.append([])

      jobFitnesses = np.ones( len(myDataFrame.index) )*-1
      jobNums      = np.ones( len(myDataFrame.index),dtype=int )*-1
      for i in range(len(myDataFrame.index)):
          #jobOutputs_copy = jobOutputs.copy()
          #slicedJobOutputs = jobOutputs_copy[slicer[]]
          #allErrors.append([myDataFrame.index[i]])
          #errorsGood_array.append([myDataFrame.index[i]])
          ###CMTprint myDataFrame.index[i]
          ###CMTprint myDataFrame.loc[myDataFrame.index[i],'jobNum']
          
          # score 'fitnesss' based on the squared error wrt each output parameter
          fitness = 0.0
          for key,obj in outputList.items():
              ###CMTprint "outputList: ", key
              result = myDataFrame.loc[myDataFrame.index[i],key]

              # Decide on scalar vs vector comparisons
              if not isinstance(result,np.ndarray):
              #  print "already an array"
              #else:
                result = np.array( result )

              # sum over squares
              error = np.sum((result - obj.truthValue) ** 2)
              normFactor = np.sum(obj.truthValue ** 2)
              normError = np.sqrt(error/normFactor) 
              if verbose>=2:
                print("result: ", result, "truthValue: ", obj.truthValue)
              #allErrors[iters-1].append(error)


              #if error <= (obj.truthValue * 0.001):
                  #errorsGood_array[iters-1].append(True)
              #else:
                  #errorsGood_array[iters-1].append(False)
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
      print("myDataFrame: ")
      print(myDataFrame)
      
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
      ###CMTprint "jobFitnes: " , jobFitnesses[jobIndex]

      # grab the job 'object' corresponding to that index
      #bestJob = jobList[ jobIndex ]
      currentFitness = jobFitnesses[pandasIndex]
      #print(jobIndices, type(jobList))
      bestJobs = []
      for i,jobIdx in enumerate(jobIndices):
        #print("Dbl fit",i, jobFitnesses[ pandasSortedIndices[i] ] ) 
        bestJobs.append( jobList[ jobIdx ] )    

      ###CMTprint "bestJob: ", bestJob

      ###CMTprint("CurrentFitness/Previous", currentFitness,previousFitness)
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
          ###CMTprint myVariedParamKey, variedParamVal 
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
            #print("HACK") 
            #trialParamVarDicti[ myVariedParamKey ][1]  = 0.

          print("Parent rank %d"%i,trialParamVarDicti)
          

      else:
        print("Old draw is better starting point, not overwriting starting point") 
        rejection+=1
        print("Rejected %d in a row (of %d) "%(rejection,maxRejectionsAllowed) )
        
      ###CMTprint allErrors

      #if errorsGood_array[iters-1].count(False) == 0:
          #errorsGood = True
      #else:
          #errorsGood = False
          ###CMTprint "Error is not good, need to run another iteration."
      #print("End")
      #print(trialParamVarDicts) 

      #iters += 1
      bestJobi = bestJobs[0]
      bestDrawAllIters.append( bestJobi[ 'varDict' ] ) 

      print("iter", iters, " out of", numIters)
      print("")
      print("######")
      print("")

      #if iters >= numIters: # or errorsGood:
      #    flag = False

  #return results

    #for key, results in outputs.iteritems():
    #  print key

  ## push data into a pandas object for later analysis
  #myDataFrame = PandaData(jobOutputs,csvFile="example.csv")
  
  #return myDataFrame
  return randomDrawAllIters, bestDrawAllIters,previousFitness

def test3():
  simulation = runner.Runner(),
  yamlVarFile = "inputParams.yaml",
  testState = "Cai",         
  # parameters to vary 
  variedParamDict = {
    "kon":  [0.5,0.2],         
    "koff":  [5.0,0.2],         
    },

  # get trial simulation results 
  returnData = []
  for i in range(2):
    varDict={
            'kon':0.4+i*0.1,'koff':4.4}

    returnDict=simulate(
      varDict=varDict,        # dictionary of parameters to be used for simulation
 #   returnDict=dict(),    # dictionary output to be returned by simulation
      jobDuration = 25e3   # [ms]
    )

    cai = analyze.GetData('Cai',returnDict['data'])
    returnData.append( cai )
    

  if 1:
    raise RuntimeError("DSF")
    # once above works, do
    traces = cellDetect()
    for trace in traces:
        outputObj=outPutList["Cai"]
        outputObj.vals = cai
        outputObj.ts   = ts   

    results.cellNum=1
    allResults.append(results)
    print("make plots with cell number, legend") 

  timeRange = [0, 2] # where to measure 
  vals = cai
  outputList= { 
    #"Cai":OutputObj("Cai","val_vs_time",[  0, 2],
    #[1,0.5,0.15],timeInterpolations=[  0,1,2]) # check that interpolated values at 0, 100, 200 are 1, 0.5 ...
    "Cai":OutputObj("Cai","val_vs_time",timeRange,
    vals,timeInterpolations=ts) # check that interpolated values at 0, 100, 200 are 1, 0.5 ...
    }

  results = run(
    simulation,
    yamlVarFile = yamlVarFile,            
    variedParamDict = variedParamDict,
    jobDuration = 30e3, # [ms]
    numRandomDraws = 8,  
    numIters = 5,    
    #sigmaScaleRate = 0.45,
    outputList = outputList,
    debug = True
)

def test1():
  # parameters to vary 
  stddev = 0.2
  variedParamDict = {
    # paramDict[myVariedParam] = [variedParamTruthVal, 0.2] # for log normal
    "kon":  [0.5,stddev],
    "koff":  [5.0,stddev]
  }

  # list of observables to be scored by GA 
  outputList = { 
    #"Cai":OutputObj("Cai","mean",[8,10], # in [s]
    # 0.1),          # value you want 
    "Nai":OutputObj("Nai","val_vs_time",[  0, 2],
    [1,0.5,0.15],timeInterpolations=[  0,1,2]) # check that interpolated values at 0, 100, 200 are 1, 0.5 ... 
  }

  #testState = "Cai"          
  simulation = runner.Runner()
  results = run(
    simulation,
    yamlVarFile = "inputParams.yaml",
    variedParamDict = variedParamDict,
    jobDuration = 30e3, # [ms]
    numRandomDraws = 8,  
    numIters = 5,    
    #sigmaScaleRate = 0.45,
    outputList = outputList,
    debug = True
)



  
# Here we try to optimize the sodium buffer to get the correct free Na concentration
def validation():

  testState = "Cai"
  simulation = runner.Runner()
  results = run(
    simulation,
    yamlVarFile = "inputParams.yaml",
    variedParamDict = variedParamListDefault,
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




"""
The genetic algorithm wrapper
This is the one you should mostly interface with (see validation()) 
"""
def run(
  simulation,
  odeModel=None,   # "shannon_2004_rat.ode", # NEEDS TO BE ANTIQUATED 
  myVariedParam=None,          
  variedParamTruthVal=5.0,
  variedParamDict = None,      
  timeStart= 0, # [ms] discard data before this time point
  jobDuration= 30e3, # [ms] simulation length
  tsteps = None, # can input nonuniform times (non uniform linspace) 
  fileName=None,
  numRandomDraws=5,
  numIters=3,
  sigmaScaleRate=0.15,
  outputList = None, # instead of manually passing in output param, comparison method etc, define list here (see default above) 
  outputParamName="Nai", # general name for objective object 
  outputParamSearcher="Nai", # name of index in return array 
  outputParamMethod="mean",
  outputParamTruthTimes=None, # time points ([ms]) at which to interpolate predicted values. Used where TruthVal is an array
  outputParamTruthVal=12.0e-3, # can be an array
  maxCores = 30,
  yamlVarFile = None,
  outputYamlFile = None,
  debug = False,
  fixedParamDict = None, # in case fixedParamDict s.b. passed in 
  verboseLevel = 2, # show everything [2], show a bit [1]
  distro = 'lognormal' # distribution with which we select new parameters
):

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
    print("""
WARNING: In debug mode.
Fixing random seed
""") 
    np.random.seed(10)
    random.seed(10)

  # Data analyzed over this range
  if tsteps is None: 
    timeRange = [timeStart*ms_to_s,jobDuration*ms_to_s] # [s] range for data (It's because of the way GetData rescales the time series)
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
  numCores = np.min([numRandomDraws*len(variedParamDict.keys()),maxCores])
  results = trial(simulation,odeModel=odeModel,variedParamDict=variedParamDict,
                  outputList=outputList,fixedParamDict=fixedParamDict,
                  numCores=numCores,numRandomDraws=numRandomDraws,
                  jobDuration=jobDuration,tsteps=tsteps,
                  numIters=numIters,sigmaScaleRate=sigmaScaleRate,
                  fileName=fileName,distro=distro,
                  verbose=verboseLevel)

  if outputYamlFile is not None:
    OutputOptimizedParams(results['bestFitDict'],originalYamlFile=yamlVarFile,outputYamlFile=outputYamlFile)

  return results

"""
The genetic algorithm
"""
def trial(
  simulation,
  odeModel,
  variedParamDict,
  outputList,
  fixedParamDict=None, # dictionary of ode file parameters/values (optional), which are not randomized
  numCores = 2, # maximum number of processors used at a time
  numRandomDraws = 2,# number of random draws for parameters list in 'parmDict' (parmDict should probably be passed in)
  jobDuration = 4e3, # [ms] simulation length
  tsteps= None,  # optional time steps [ms] 
  numIters=2,
  sigmaScaleRate = 1.0,
  fileName = None,
  distro = 'lognormal', # distribution with which we select new parameters
  verbose =2
  ):
  odeModel = None 

  print("WHY is this wrapper needed") 
  # get varied parameter (should only be one for now)
  keys = [key for key in variedParamDict.keys()]
  #variedParamKey = keys[0]
  #if len(keys)>1:
  #  raise RuntimeError("can only support one key for now")


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
    odeModel, jobDuration=jobDuration,tsteps=tsteps,fixedParamDict=fixedParamDict,results=results)

  return results

def DisplayFit(simulation,
         odeModel=None, 
         jobDuration=30e3,tsteps=None,
         fixedParamDict=None,
         results=None):           
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
  testStateName = obj.name
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
  plt.gcf().savefig(testStateName + ".png")


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
  ###CMTprint bestDraws
  ###CMTprint veryBest
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
  


#!/usr/bin/env python
import sys
##################################
#
# Revisions
#       10.08.10 inception
#
##################################



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

  #if len(sys.argv) < 2:
  #    raise RuntimeError(msg)

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
  #  ###CMTprint "arg"

  # Loops over each argument in the command line
  for i,arg in enumerate(sys.argv):
    # calls 'doit' with the next argument following the argument '-validation'
    if(arg=="-validation"):
      validation()
      quit()
    elif(arg=="-test1"):
      test1()
      quit()
    elif(arg=="-test2"):
      test2()
      quit()

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
