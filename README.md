# TTbarAllHadUproot
Uproot-based analysis of the ttbar all-hadronic analysis.

For this analysis, create an environment which comes with the latest versions of Coffea and Awkward1.

For a dask environment that can run this analysis on lpc, you should follow the setup instructions from https://github.com/lgray/lpc_dask .
Be sure to git clone both this repository and https://github.com/lgray/lpc_dask in the same working directory.  
# How to run
From this working directory, run the uproot job that will produce coffea output files by entering this in the terminal: 

```python -i TTbarAllHadUproot/TTbarResCoffeaOutputs.py```

