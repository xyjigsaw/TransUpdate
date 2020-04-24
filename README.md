# TransUpdate

Developing in progress

Online Learning Framework for Updating KG embedding based on TransE, TransD, TransH and TransR.

## Main Files
- KG_data.py: load and save KG triples, embedding and model parameters
- transX.py: include train and test model for TransE, TransD, TransH and TransR
- transuttnew.py: online learning for new triples embedding task based on transX
- transRel.py: online learning with more complex neighbor aggregator
