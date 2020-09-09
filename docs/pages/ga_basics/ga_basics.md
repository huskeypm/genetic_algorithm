---
title: Genetic Algorithm Basics
nav_order: 2
---

# Genetic Algorithm Basics
{:.no_toc}

* TOC 
{:toc}

## Overview

This page goes over the basics of genetic algorithms. What are they, what are they used for, and how do we use them?

Note: Much of this information was heavily borrowed from the Wikipedia article for genetic algorithms.

## What Are Genetic Algorithms?

To help answer this question, let's borrow some of the information from [Wikipedia's Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) article, a genetic algorithm (GA) is an algorithm inspired by natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms rely on biologically-inspired operations like mutation, crossing over of "alleles", and selection of the fittest to "survive". Don't feel bad if this seems intimidating. All that you need to know is that GAs are a technologically clever way to approach optimization problems.

In GAs a population of candidate solutions (called individuals, creatures, or phenotypes) to an optimization problem is *evolved* toward better solutions. Each candidate solution has a set of properties (its chromosomes, genotype, or alleles) which are mutated and altered - just like in real life - to adapt to fitting the optimization problem better.

### What Are The Steps Of A Genetic Algorithm?

We'll give a quick overview of the process of using our GA before we dive into each step in detail. The GA breaks down into the following steps:

+ Before using the genetic algorithm, reframe your optimization problem into a function that accepts parameters as floating point numbers. <font color=red>revise me</font> This will likely be easy since you're already working with numerical models of cellular function! This is usually as easy as making sure that all of your parameters are specified via a YAML file.

1. Generate a population of candidate solutions (called individuals, creatures, or phenotypes)
