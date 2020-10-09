# Genetic Algorithm
Genetic algorithm for fitting ODEs

# Table of Contents
- [Genetic Algorithm](#genetic-algorithm)
- [Table of Contents](#table-of-contents)
  - [components](#components)
  - [To run](#to-run)
- [simple case](#simple-case)
- [more detailed](#more-detailed)
  - [TODO](#todo)
  - [Done](#done)
  - [To set up for new cases](#to-set-up-for-new-cases)
  - [Bells and whistles](#bells-and-whistles)
  - [Testing](#testing)
    - [Running Tests](#running-tests)
    - [Writing Tests](#writing-tests)

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

## Testing

It is important to validate any code before it is committed (and develop new validation tests as appropriate!).

This repo uses [PyTest](https://docs.pytest.org/en/stable/) for it's testing framework.

PyTest is not part of Python's standard library and thus needs to be installed before validation of changes can be done. This can be done through PIP or your environment management tool.

### Running Tests

To run all of the tests in the repository, both unit and integration tests, execute the following from the project's root directory:

```
$ pytest
```

If you would like to view the output of the tests, simply add `-s` to the previous command.

### Writing Tests

Any time you make additions to the repository or alter the code, it's important to write tests that cover this new code. That way, we can ensure that the new code doesn't break the existing code. Additionally, by adding a new test, we can do regression testing with all future development.

There are two main types of tests that you can write: unit and integration tests. 
+ Unit tests cover the smallest units of the code that you have added. If you add a new class or function, unit tests will test *only* this new addition. 
+ Integration tests cover how units integrate with each other.

For a full explanation on how to write tests, follow the [PyTest documentation](https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test). I've briefly summarized the method here:

1. Decide on whether this is going to be a unit or integration test. *Hint* it should probably be a new unit test since most integration tests are already written.
    + If you're writing a unit test, the test will be located in `test/unit/`. Similarly, if you're writing an integration test, the test will need to be located in `test/integration/`.
2. Decide on whether you need to write a whole new test suite (test Python file) or add the test to a new file.
    + If writing the test to a new file, name the file `test_<name of thing you're testing>.py` in the directory step number 1 told you to work in.
3. For each of the new tests that you want to write, define a new function called `test_<name of test>` where `<name of test>` is a description of the functionality that you're testing.

And that's it! When you follow the instructions in [Running Tests](#running-tests), PyTest automatically finds all of the tests in the repository and runs them.