# genetic_algorithm
Genetic algorithm for fitting ODEs

##
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
