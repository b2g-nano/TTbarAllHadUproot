# TTbarAllHadUproot
Uproot-based analysis of the ttbar all-hadronic analysis that uses a COFFEA based workflow.

For this analysis, create an environment which comes with the latest versions of Coffea and Awkward1.

### For LPC:
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

> ./shell

### For Coffea-Casa:

[MovingToCoffeaCasa.pdf](https://github.com/b2g-nano/TTbarAllHadUproot/files/9181161/MovingToCoffeaCasa.pdf)
![MovingToCoffeaCasa-01](https://user-images.githubusercontent.com/42876001/191271139-2770b104-516a-4567-b7f7-bdbdecd8e242.png)
![MovingToCoffeaCasa-02](https://user-images.githubusercontent.com/42876001/191271177-f69f1ea3-c611-464a-879d-5472c9ee99e2.png)
![MovingToCoffeaCasa-03](https://user-images.githubusercontent.com/42876001/191271205-b8049339-2f39-4766-a7aa-39f0890014bc.png)
![MovingToCoffeaCasa-04](https://user-images.githubusercontent.com/42876001/191271231-ebf07498-a296-4332-a767-70e5f4872791.png)
![MovingToCoffeaCasa-05](https://user-images.githubusercontent.com/42876001/191271265-99206ce9-0e69-4cac-b3a0-5302e9e9a00a.png)
![MovingToCoffeaCasa-06](https://user-images.githubusercontent.com/42876001/191271332-eba3a8a1-65d9-4c9b-bd2e-ad2964255438.png)
![MovingToCoffeaCasa-07](https://user-images.githubusercontent.com/42876001/191271349-81346536-16f6-40e5-84ff-4bf96109e4c7.png)
![MovingToCoffeaCasa-08](https://user-images.githubusercontent.com/42876001/191271372-5204d53e-206e-4de1-aa5c-9caa7fbbac58.png)
![MovingToCoffeaCasa-09](https://user-images.githubusercontent.com/42876001/191271390-6b700e5d-0ca2-4b9a-bb41-06538ba64dcb.png)
![MovingToCoffeaCasa-10](https://user-images.githubusercontent.com/42876001/191271739-42a9389f-1edd-41df-8dab-4fdf9a874f07.png)
![MovingToCoffeaCasa-11](https://user-images.githubusercontent.com/42876001/191271786-f7718a2f-cb51-4f99-b8d9-03d3411930d0.png)
![MovingToCoffeaCasa-12](https://user-images.githubusercontent.com/42876001/191271930-80b87200-0e73-41a2-8048-f190b0896e63.png)
![CoffeaCasa](https://user-images.githubusercontent.com/42876001/191276229-6acd0320-0032-4d13-a8c6-e582efbb76fb.png)
![MovingToCoffeaCasa-14](https://user-images.githubusercontent.com/42876001/191272013-f9dfd5f2-1de3-4bd3-9267-edad9b3fe374.png)
![MovingToCoffeaCasa-15](https://user-images.githubusercontent.com/42876001/191272034-4551948d-93d1-446d-815a-cbdac844bc05.png)
***
# Brief Intro and Basic Idea for Using COFFEA

![Coffea1024_2](https://user-images.githubusercontent.com/42876001/180783601-77e4a680-25c8-41e6-8e44-046df88b631a.jpg)
![Coffea1024_3](https://user-images.githubusercontent.com/42876001/180783621-562ebddc-63f6-4676-91d0-7c352425a0ad.jpg)
***
# How to run
From within this repo, you can run the uproot job that will produce coffea output files.  To see a list of arguments needed to run this program please enter the following in the terminal: 

> python Run.py --help

The output should look something like this:
```
usage: Run.py [-h] [-t | -m | -T | -F RUNFLAVOREFF [RUNFLAVOREFF ...] | -M RUNMMO [RUNMMO ...] | -d RUNDATASET [RUNDATASET ...]] (-C | -L) [-l | -med | -med2016] [-a {yes,no}] [-y {2016,2017,2018,0}]
              [--uproot {1,2}] [--letters LETTERS [LETTERS ...]] [--chunks CHUNKS] [--chunksize CHUNKSIZE] [--save] [--saveMistag] [--saveTrig] [--saveFlav] [--dask] [--newCluster] [--timeout TIMEOUT] [--useEff]
              [--tpt] [--useHist] [--step {1,2,3,4}]
              [--bTagSyst {central,up,down} | --tTagSyst {central,up,down} | --ttXSSyst {central,up,down} | --lumSyst {central,up,down} | --jec | --jer | --pdf | --pileup {central,up,down}]

-----------------------------------------------------------------------------
Run the TTbarAllHadProcessor script.  
All objects for each dataset ran can be saved as its own .coffea output file.
-----------------------------------------------------------------------------

optional arguments:
  -h, --help            show this help message and exit
  -t, --runtesting      Only run a select few root files defined in the code.
  -m, --runmistag       Make data mistag rate where ttbar contamination is removed (as well as ttbar mistag rate)
  -T, --runtrigeff      Create trigger efficiency hist coffea output objects for chosen condition
  -F RUNFLAVOREFF [RUNFLAVOREFF ...], --runflavoreff RUNFLAVOREFF [RUNFLAVOREFF ...]
                        Create flavor efficiency hist coffea output objects for chosen MC datasets
  -M RUNMMO [RUNMMO ...], --runMMO RUNMMO [RUNMMO ...]
                        Run Mistag-weight and Mass modification Only (no other systematics for uproot 2)
  -d RUNDATASET [RUNDATASET ...], --rundataset RUNDATASET [RUNDATASET ...]
                        List of datasets to be ran/loaded
  -C, --casa            Use Coffea-Casa redirector: root://xcache/
  -L, --lpc             Use CMSLPC redirector: root://cmsxrootd.fnal.gov/
  -l, --loose           Apply loose bTag discriminant cut
  -med, --medium        Apply medium bTag discriminant cut
  -med2016, --medium2016
                        Apply medium bTag discriminant cut from 2016 AN
  -a {yes,no}, --APV {yes,no}
                        Do datasets have APV?
  -y {2016,2017,2018,0}, --year {2016,2017,2018,0}
                        Year(s) of data/MC of the datasets you want to run uproot with. Choose 0 for all years simultaneously.
  --uproot {1,2}        1st run or 2nd run of uproot job. If not specified, both the 1st and 2nd job will be run one after the other.
  --letters LETTERS [LETTERS ...]
                        Choose letter(s) of jetHT to run over
  --chunks CHUNKS       Number of chunks of data to run for given dataset(s)
  --chunksize CHUNKSIZE
                        Size of each chunk to run for given dataset(s)
  --save                Choose to save the uproot job as a coffea output for later analysis
  --saveMistag          Save mistag rate calculated from running either --uproot 1 or --mistag
  --saveTrig            Save uproot job with trigger analysis outputs (Only if -T selected)
  --saveFlav            Save uproot job with flavor efficiency outputs (Only if -F selected)
  --dask                Try the dask executor (experimental) for some fast processing!
  --newCluster          Use Manually Defined Cluster (Must Disable Default Cluster First if Running in CoffeaCasa)
  --timeout TIMEOUT     How many seconds should dask wait for scheduler to connect
  --useEff              Use MC bTag efficiencies for bTagging systematics
  --tpt                 Apply top pT re-weighting for uproot 2
  --useHist             use scikit-hep/hist for histograms
  --step {1,2,3,4}      Easily run a certain step of the workflow
  --bTagSyst {central,up,down}
                        Choose Unc.
  --tTagSyst {central,up,down}
                        Choose Unc.
  --ttXSSyst {central,up,down}
                        ttbar cross section systematics. Choose Unc.
  --lumSyst {central,up,down}
                        Luminosity systematics. Choose Unc.
  --jec                 apply jec systematic weights
  --jer                 apply jer systematic weights
  --pdf                 apply pdf systematic weights
  --pileup {central,up,down}
                        Choose Unc.

                                Available List of Dataset Strings:
                                Key:
                                -------------------------------------------------------------------------------
                                <x> = integer from [1, 5]
                                <y> = integer either 0 or 5 
                                <x> = <y> = 5 is not an available string to be included in dataset string names
                                -------------------------------------------------------------------------------
                                QCD
                                DM<x><y>00, DM
                                RSGluon<x><y>00, RSGluon
                                TTbar
                                JetHT
                                SingleMu

                                    **NOTE**
                                    =========================
                                    JetHT 2016 letters: B - H
                                    JetHT 2017 letters: B - F
                                    JetHT 2018 letters: A - D
                                    =========================

    Example of a usual workflow on Coffea-Casa to make the relevant coffea outputs:

    0.) Make Outputs for Flavor and Trigger Efficiencies
./Run.py -C -med -F QCD TTbar DM RSGluon -a no -y 2016 --dask --saveFlav
./Run.py -C -med -T -a no -y 2016 --dask --saveTrig

    1.) Create Mistag Rates that will be used to estimate NTMJ background
./Run.py -C --step 1
python Run.py -C -med -m -a no -y 2016 --saveMistag

    2.) Make Outputs for the first Uproot Job with no weights applied (outside of MC weights that come with the nanoAOD)
./Run.py -C --step 2
python Run.py -C -med -d QCD TTbar JetHT DM RSGluon -a no -y 2016 --uproot 1 --save

    3.) Make Outputs for the second Uproot Job with only mistag rate applied to JetHT and TTbar, and mass modification of JetHT and TTbar in pre-tag region
./Run.py -C --step 3
python Run.py -C -med -M QCD TTbar JetHT DM RSGluon -a no -y 2016 --save

    4.) Make Outputs for the second Uproot Job with systematics, on top of mistag rate application and mass modification
./Run.py -C --step 4
python Run.py -C -med -d QCD TTbar JetHT DM RSGluon -a no -y 2016 --uproot 2 --bTagSyst central --useEff --save
```
***
# How it works
The processor is where all of the analysis is defined.  The processor is aptly named `TTbarResProcessor.py`.  

The file `Run.py` runs the file according to the selected options at the beginning of the file.  When this is run, the analysis is performed and the outputs defined in the processor can be stored in a `.coffea` file, which can be found in the corresponding directory `CoffeaOutputs` or `CoffeaOutputsForCombine`.  The first directory `CoffeaOutputs` has outputs that were made while doing numerous tests to ensure the processor was giving what is expected.

For starters, if you are running the code on the LPC or Coffea-Casa, you must specify either `--lpc` (`-L`) or `--casa` (`-C`) respectively.  This is important so that the correct redirector is used for locating the desired datasets.  It also sets specific options for running the dask executor that vary between these two environments.

Next, specify the btagging working point, WP, that you want to run the processor with.  There are two (technically three) choices to pick from.  You can either choose to run with the loose or medium WP, `--loose` (`-l`) or `--medium` (`-med`). For testing purposes, you can also choose to run the processor with the same medium WP that was defined in the 2016 Analysis Note ([AN2016_459_v8.pdf](https://github.com/b2g-nano/TTbarAllHadUproot/files/9182018/AN2016_459_v8.pdf)), by specifying the `-med2016` option.  The output files created from using the loose and medium WPs will be saved in a directory that is labeled with whatever WP you've picked.  Coffea outputs made from `-med2016` option will not get it's own dedicated directory path label(s). 

You can choose the datasets you want for the first and second uproot run by specifying `--rundataset` or `-d` followed by the names of the datasets you'd like to run.  When running the code with this `-d` option (selecting the datasets you want from the terminal) it is mandatory to give the names of the dataset according to the key listed in the help message's epilogue.  For any run option selected to run the program (`-d`, `-m` or `-t`) you must also specify the year, `---year` or `-y`, and whether or not the datasets have APV or not, `--APV` or `-a`.  All other arguments are optional, but should still be carefully considered depending on what you want to do.
***
## Example 1:
Suppose you would like to run the 1.5 TeV Zprime to DM and 2.0 TeV RS Gluon Ultra Legacy 16 files with no APV included.  You just want an idea of the order of magnitude of events that goes into each analysis category.  For this run, let's assume you don't need/want to save this coffea output to either avoid clutter in the directory or overwriting a preexisting coffea output with better stats.  Also, there is no need to apply mistag/mod-mass/systematic corrections to this run, as this is just a run out of curiosity; you only want to see the output of the cutflow onto the terminal.  In this case, you only need to run the first uproot job and you can ignore the second run to save time.

For such a task, the code can be ran with the following arguments like this:

> python Run.py -d DM1500 RSGluon2000 -a no -y 2016 --uproot 1 --chunks 10 --chunksize 1000

This runs the first uproot job with the two desired datasets according to the APV status and year (and also mass in this example).  The choice of chunks and chunksize gives roughly 10<sup>1</sup> times 10<sup>3</sup> (10,000) events
***
## Other Examples:

TBA soon :)
***
# Workflow

![Coffea1024_5](https://user-images.githubusercontent.com/42876001/180783658-0ddf6d5b-75b7-4a31-82b8-9dae55925323.jpg)

## --- Import ---
----------------
As this step implies, insure that the necessary packages, primarily coffea, awkward1 and numpy, are imported and ready to use.  Not much detail is assumed to be required for this step of the workflow.

## --- Processor ---
##### Each processor is defined as a class object with an initializer.  The initializer also initializes the accumulator, which defines the coffea objects to be filled with variables of interest.  Of course, other user specific definitions can also be defined after the initializer function.  The accumulator function is defined to return its initialized self.  Then, most importantly, the process itself is defined, where all of the event selection for the analysis takes place, as well as filling the coffea objects (namely histograms).  Finally, postprocessing is defined to simply return the accumulator.
----------------
#### Main Analysis
1. Preliminary Cuts/Selections
    - $HT_{Cut}\ >\ 950\ GeV$
    - Loose Jet ID
    - $p_T\ >\ 400\ GeV$ and $|y|\ <\ 2.4$
    - Two AK8 Jets
      - Randomly assign these two jets as ttbar candidate 0 and 1 to avoid bias
      - Select events with at least one pair of ttbar candidates
    - $\Delta\Phi > 2.1$ between two ttbar candidates  
    - TTbar candidates with two subjets each (bjets interpreted another way)
2. Analysis Categories; Combinations of regions defined by number of ttags and btags and $|\Delta y|$ window 
    - Define Rapidity Regions
      - central region: 
        - $|\Delta y|\ < 1.$  
      - forward region: 
        - $|\Delta y|\ > 1.$ 
    - Define Top Tag Regions (Either with CMSTTV2 (CMS Top Tagger Version 2) or DeepAK8 Tagger)
      - CMSTTV2 Top Tag:
        - $\tau_{3/2\} <\ 0.65$ 
        - $105\ GeV\ <\ m_{SD}\ <\ 210\ GeV$
      - CMSTTV2 Anti-Tag:
        - $\tau_{3/2\} >\ 0.65$ 
        - $105\ GeV\ <\ m_{SD}\ <\ 210\ GeV$
      - DeepAK8 Top Tag:
        - deepTag_TvsQCD $>\ 0.435$  
      - DeepAK8 Anti-Tag:
        - deepTag_TvsQCD $<\ 0.435$ (???)
    - Define b Tag Regions
      - For each ttbar candidate find the subjet with the largest DeepCSV value; btag $DeepB_{max}$
      - Medium Working Point b Tag:
        - btag $DeepB_{\max}\ >\ 0.5847$
      - Loose Working Point b Tag:
        - btag $DeepB_{max}\ >\ 0.1918$
3. Scale Factors
    - b Tag SF's used to either:
      - a. Create an additional event weight (independent of MC flavor tag efficiency)
      - b. Update b-tag status of ttbar candidates (dependent on MC flavor tag efficiency)
4. Loop Through Analysis Categories (Hist objects are filled with desired variables acccording to dataset and category, along with the event weights)
    - Uproot 1 Option (-d [LIST OF DATASETS] --uproot 1 ...)
      - No additional weights and/or corrections are applied apart from the generator event weights (if any)
      - Fill histograms
    - Mistag Run Option (-m ...)
      - Same as Uproot 1 Option, but specifically done with TTbar and JetHT datasets
      - Standard model ttbar contamination is removed from JetHT mistag-rate
    - Uproot 2 Option (-d [LIST OF DATASETS] --uproot 2 --SystematicOption {central, up, down} ...)
      - Re-weight events by ttbar contamination subtracted JetHT mistag rate
      - Mass Modification Procedure
      - Include any weights from corrections/re-weighting/uncertainties...
      - Fill histograms 
    - Mass Modification Only Option (-M [LIST OF DATASETS] ...)
      - Re-weight events by ttbar contamination subtracted JetHT mistag rate
      - Mass Modification Procedure
      - No other systematic corrections included in this run option
      - Fill histograms
#### MC Flavor Efficiency Analysis (-F [LIST OF DATASETS] ...)
1. Preliminary Cuts/Selections
    - $HT_{Cut}\ >\ 950\ GeV$
    - Loose Jet ID
    - $p_T\ >\ 400\ GeV$ and $|y|\ <\ 2.4$
    - Two AK8 Jets
      - Randomly assign these two jets as ttbar candidate 0 and 1 to avoid bias
      - Select events with at least one pair of ttbar candidates
    - $\Delta\Phi > 2.1$ between two ttbar candidates  
    - TTbar candidates with two subjets each (bjets interpreted another way)
2. Get Flavor Efficiency Info
    - Efficiency defined as the rate of a given subjet flavor passing our b-tag requirement
    - Fill histograms as functions of subjet $\eta$ and $p_T$
#### Trigger Efficiency Analysis (-T ...)
1. Combination of 2016 Triggers
    - HLT_PFHT900
    - HLT_AK8PFHT700_TrimR0p1PT0p03Mass50
    - HLT_AK8PFJet450
    - HLT_AK8PFJet360_TrimMass30
      - Common Control Triggers (for denominator of efficiency calculation):
        - HLT_Mu50
        - HLT_IsoMu24
2. Cuts/Selections
    - Loose Jet ID
    - $p_T\ >\ 400\ GeV$ and $|y|\ <\ 2.4$
    - Two AK8 Jets
      - Randomly assign these two jets as ttbar candidate 0 and 1 to avoid bias
      - Select events with at least one pair of ttbar candidates
    - $\Delta\Phi > 2.1$ between two ttbar candidates  
    - TTbar candidates with two subjets each (bjets interpreted another way)
3. Analysis Categories; With and without softdrop mass window
    - $105\ GeV\ <\ m_{SD}\ <\ 210\ GeV$
4. For Both Analysis Categories:
    - For Jet $p_T\ >\ 30\ GeV$ and $|\eta|\ <\ 3.0$:
      - $\mathit{Jet}\ H_T\ =\ \sum{p_{T_i}}$
    - Efficiency defined as the rate of jets that pass combination of triggers
    - Fill histograms as function of Jet $H_T$
## --- Uproot Job ---
##### The script `Run.py` imports the desired processor from `TTbarResProcessor.py`, along with all other required scripts
----------------
1. Import processor(s)
2. Import the desired datasets from `Filesets.py` script, that reads the files in from the `nanoAODv9Files` directory
3. Setup Dask if desired (Highly Recommended for Fast Processing Speed when processing whole dataset(s))

![Coffea1024_4](https://user-images.githubusercontent.com/42876001/180783637-694246d5-e3cc-498f-a96f-b7c2477c4f7e.jpg)

4. Perform uproot job
    - Define a dictionary that maps string names to the datasets' files
    - Call `run_uproot_job` from the coffea processor
    - Give the run_uproot_job with the names and files from the dictionary defined in this first step.  
```python
for name,files in filesets_to_run.items(): 
    output = processor.run_uproot_job({name:files},
                                      treename='Events',
                                      processor_instance=MCFlavorEfficiencyProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={
                                          #'client': client,
                                          'skipbadfiles':False,
                                          'schema': BaseSchema, #NanoAODSchema,
                                          'workers': 2},
  ```
***
