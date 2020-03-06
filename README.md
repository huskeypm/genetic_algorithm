### genetic_algorithm
Genetic algorithm for fitting ODEs

## components 
runner.py - your simulation engine
analyze.py - basic functions for analyzing outputs from runner.py dictionaries
inputParams.yaml - optional for passing in fixed parameters to runner.py

fittingAlgorithm.py - the workhorse (see 'to run' below) 


## To run
# simple case 
python3 fittingAlgorithm.py -test1

# more detailed
python3 fittingAlgorithm.py -run -myVariedParam kon -variedParamTruthVal 2.0 -fileName This_Is_A_Test.png -numRandomDraws 3 -numIters 3 -sigmaScaleRate 0.15 -outputParamName Cai -outputParamSearcher Cai -outputParamMethod mean -outputParamTruthVal 0.1 -fixedvars inputParams.yaml 

## TODO
- select top N candidates for propagation, not just one
- add in mutation pseudocode 

## Done 
- Add yaml reader

## To set up for new cases 
- Ingredients 
-- runner (runner.py) is the wrapper for your code. 
-- analysis (analysis.py) contains many analyses for comparing runner results (curves, etc) with your fitting data
-- inputParams.yaml  - default parameters for your code 
-- define outputList object for your values you want to fit to (e.g. from expt) 

- To interface the GA with your code, you need to include a function called
simulate() in your code that expects a dictionary of parameters and 
parameter values, e.g. 
  varDict['kon'] = 1.
as well as a blank dictionary called returnDict. It needs to be structured as a class, so check runner.py for an example  
- rest of instructions are in exampleFitting.ipynb....

## Bells and whistles
Cross-overs added 


