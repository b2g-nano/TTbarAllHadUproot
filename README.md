# TTbarAllHadUproot
Uproot-based analysis of the ttbar all-hadronic analysis.

For this analysis, create an environment which comes with the latest versions of Coffea and Awkward1.

For LPC:
``` 
ssh <username>@cmslpc-sl7.fnal.gov 
voms-proxy init -voms cms
```

For a coffea-dask environment that can run this analysis on lpc, you should follow the setup/installation instructions from https://github.com/CoffeaTeam/lpcjobqueue.  For convience sake, the installation instructions there are listed below.  

Run these commands only if creating this environment for the first time
```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```

After the initial setup steps of the coffea-dask environment (stated above) you can enter the singularity shell with the following command (wherever you saved your bootstrap.sh)
```
./shell
```

# How to run
From your working directory (and one layer outside of this repo), you can run the uproot job that will produce coffea output files by entering this in the terminal: 
```python TTbarAllHadUproot/TTbarResCoffeaOutputs.py```

# How it works
The processor is where all of the analysis is defined.  The processor is aptly named `TTbarResProcessor.py`.  

The file `TTbarResCoffeaOutputs.py` runs the file according to the selected options at the beginning of the file.  When this is run, the analysis is performed and the outputs defined in the processor are stored in a `.coffea` file, which can be found in the corresponding directory `CoffeaOutputs` or `CoffeaOutputsForCombine`.  The first directory `CoffeaOutputs` has outputs that were made while doing numerous tests to ensure the processor was giving what is expected.

You can choose the datasets you want for the first and second uproot run by uncommenting the dataset names in the corresponding dictionaries in the file `Filesets.py`.  
(*NOTE* A better, more elogent way of doing this may be implemented soon.  Stay tuned for this upate)

The mistag rates are stored in lookup tables (or LUTs for short).  The file `TTbarResCoffeaOutpts` creates the LUTs after the first run by importing the module `TTbarResLookUpTables.py`.  In this module, you can define whether you want to create new mistag LUTs with the given datasets from `Filesets.py`.  If you choose to create new LUTs (runLUTS == True), you need to also specify whether or not you would like to create new LUTs from the TTbar MC dataset.  For example: if you only run JetHT and create LUTs from this dataset alone, you would have to load a pre-existing TTbar `.coffea` output to gaurantee that the correction for TTbar subtraction is implemented.  If OldTTbar == False in the module, then no TTbar contamination will be corrected for JetHT (if you are also making new JetHT LUTs) and the TTbar MC LUTs will not be created, unless you have included the TTbar datasets to be run in the first uproot job.

In `TTbarResCoffeaOutputs.py` the processor is ran a second time, where the second run applies the mistag weights stored as LUTs from the first run.  
