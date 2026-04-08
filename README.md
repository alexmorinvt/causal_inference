# Causal Inference Experimentation

### Example Running
`./run_model.sh -m mean_difference -p src/MeanDifference.py`

### Notes
- causalbench subrepo is present for reference. There is no need to run `git submodule init`
- the folders output, storage, and manual_data are reserved.
- this will not run immediately - there is a nontrivial amount of editing to the pip installed causalbench files that is required to run this, specifically, 
    - 

### Custom model list
- mean_difference, src/MeanDifference.py
- random, src/RandomModel.py
- mean_difference, src/DataSaverModel.py
    - This model saves the expression data to the manual_data folder

