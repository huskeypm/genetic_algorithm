---
title: Variables In Our GA
nav_order: 1
parent: Repository Description
---

# Variables In Our GA
{:.no_toc}

* TOC
{:toc}

+ `simulation`
    + Description: the simulation engine that runs the problem that you are wishing to optimize. A template for this can be found in `runner.py` in this repository. A description of the structure follows:
        + Needs to be in `runner.py`
        + Main method to run simulation needs to be defined as class `Runner` with the following methods:
            + `__init__` must assign `Runner.params` dictionary where the key is the parameter name and the value is the value of the parameter that you would like to use in this simulation. For example:
                ```python
                def __init__(self):
                  self.params = {
                    "kon": 0.1,
                    "koff": 0.1,
                    "bMax": 5,
                    "scale": 1
                  }
                ```
            + `dydt` that takes `ys`, `ts`, and `params`.
                + `ys` 