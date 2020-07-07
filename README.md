# TransUpdate

![](https://img.shields.io/badge/Status-Developing-brightgreen.svg)

Developing in progress.

The project is part of AcaFinder.

Online Learning Framework for Updating KG embedding based on TransE, TransD, TransH and TransR.

## Main Files
- KG_data.py: load and save KG triples, embedding and model parameters
- transX.py: include train and test model for TransE, TransD, TransH and TransR
- transuttnew.py: online learning for new triples embedding task based on transX
- transRel.py: online learning with more complex neighbor aggregator

## Idea
Optimize cross entropy between different probability distributions .

