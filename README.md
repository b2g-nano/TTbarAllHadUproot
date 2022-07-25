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
***
# Brief Intro and Basic Idea for Using COFFEA

![Coffea1024_2](https://user-images.githubusercontent.com/42876001/180783601-77e4a680-25c8-41e6-8e44-046df88b631a.jpg)
![Coffea1024_3](https://user-images.githubusercontent.com/42876001/180783621-562ebddc-63f6-4676-91d0-7c352425a0ad.jpg)
***
# How to run
From within this repo, you can run the uproot job that will produce coffea output files.  To see a list of arguments needed to run this program please enter the following in the terminal: 

> python TTbarResCoffeaOutputs.py --help

The output should look something like this:
```
usage: TTbarResCoffeaOutputs.py [-h] (-t | -m | -T | -F RUNFLAVOREFF [RUNFLAVOREFF ...] | -M RUNMMO [RUNMMO ...] | -d RUNDATASET [RUNDATASET ...]) (-C | -L) (-l | -med) -a {yes,no} -y
                                {2016,2017,2018,0} [--uproot {1,2}] [--chunks CHUNKS] [--chunksize CHUNKSIZE] [--save] [--saveMistag] [--saveTrig] [--saveFlav] [--dask] [--useEff] [--tpt]
                                [--bTagSyst {central,up,down} | --tTagSyst {central,up,down} | --ttXSSyst {central,up,down} | --lumSyst {central,up,down} | --jec {central,up,down} | --jer {central,up,down} | --pileup {central,up,down}]

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
  -a {yes,no}, --APV {yes,no}
                        Do datasets have APV?
  -y {2016,2017,2018,0}, --year {2016,2017,2018,0}
                        Year(s) of data/MC of the datasets you want to run uproot with. Choose 0 for all years simultaneously.
  --uproot {1,2}        1st run or 2nd run of uproot job. If not specified, both the 1st and 2nd job will be run one after the other.
  --chunks CHUNKS       Number of chunks of data to run for given dataset(s)
  --chunksize CHUNKSIZE
                        Size of each chunk to run for given dataset(s)
  --save                Choose to save the uproot job as a coffea output for later analysis
  --saveMistag          Save mistag rate calculated from running either --uproot 1 or --mistag
  --saveTrig            Save uproot job with trigger analysis outputs (Only if -T selected)
  --saveFlav            Save uproot job with flavor efficiency outputs (Only if -F selected)
  --dask                Try the dask executor (experimental) for some fast processing!
  --useEff              Use MC bTag efficiencies for bTagging systematics
  --tpt                 Apply top pT re-weighting for uproot 2
  --bTagSyst {central,up,down}
                        Choose Unc.
  --tTagSyst {central,up,down}
                        Choose Unc.
  --ttXSSyst {central,up,down}
                        ttbar cross section systematics. Choose Unc.
  --lumSyst {central,up,down}
                        Luminosity systematics. Choose Unc.
  --jec {central,up,down}
                        Choose Unc.
  --jer {central,up,down}
                        Choose Unc.
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
                                DM<x><y>00
                                RSGluon<x><y>00
                                TTbar
                                JetHT
                                SingleMu
                                NOTE** UL17 and UL18 samples TBA
```
***
# How it works
The processor is where all of the analysis is defined.  The processor is aptly named `TTbarResProcessor.py`.  

The file `TTbarResCoffeaOutputs.py` runs the file according to the selected options at the beginning of the file.  When this is run, the analysis is performed and the outputs defined in the processor can be stored in a `.coffea` file, which can be found in the corresponding directory `CoffeaOutputs` or `CoffeaOutputsForCombine`.  The first directory `CoffeaOutputs` has outputs that were made while doing numerous tests to ensure the processor was giving what is expected.

For starters, if you are running the code on the LPC or Coffea-Casa, you must specify either `--lpc` (`-L`) or `--casa` (`-C`) respectively.  This is important so that the correct redirector is used for locating the desired datasets.  It also sets specific options for running the dask executor that vary between these two environments.

Next, specify the btagging working point, WP, that you want to run the processor with.  There are two (technically three) choices to pick from.  You can either choose to run with the loose or medium WP, `--loose` (`-l`) or `--medium` (`-med`). For testing purposes, you can also choose to run the processor with the same medium WP that was defined in the 2016 Analysis Note (), by specifying the `-med2016` option.  The output files created from using the loose and medium WPs will be saved in a directory that is labeled with whatever WP you've picked.  Coffea outputs made from `-med2016` option will not get it's own dedicated directory path label(s). 

You can choose the datasets you want for the first and second uproot run by specifying `--rundataset` or `-d` followed by the names of the datasets you'd like to run.  When running the code with this `-d` option (selecting the datasets you want from the terminal) it is mandatory to give the names of the dataset according to the key listed in the help message's epilogue.  For any run option selected to run the program (`-d`, `-m` or `-t`) you must also specify the year, `---year` or `-y`, and whether or not the datasets have APV or not, `--APV` or `-a`.  All other arguments are optional, but should still be carefully considered depending on what you want to do.
***
## Example 1:
Suppose you would like to run the 1.5 TeV Zprime to DM and 2.0 TeV RS Gluon Ultra Legacy 16 files with no APV included.  You just want an idea of the order of magnitude of events that goes into each analysis category.  For this run, let's assume you don't need/want to save this coffea output to either avoid clutter in the directory or overwriting a preexisting coffea output with better stats.  Also, there is no need to apply mistag/mod-mass/systematic corrections to this run, as this is just a run out of curiosity; you only want to see the output of the cutflow onto the terminal.  In this case, you only need to run the first uproot job and you can ignore the second run to save time.

For such a task, the code can be ran with the following arguments like this:

> python TTbarResCoffeaOutputs.py -d DM1500 RSGluon2000 -a no -y 2016 --uproot 1 --chunks 10 --chunksize 1000

This runs the first uproot job with the two desired datasets according to the APV status and year (and also mass in this example).  The choice of chunks and chunksize gives roughly 10<sup>1</sup> times 10<sup>3</sup> (10,000) events
***
## Other Examples:

TBA soon :)
***
# Workflow

![Coffea1024_4](https://user-images.githubusercontent.com/42876001/180783637-694246d5-e3cc-498f-a96f-b7c2477c4f7e.jpg)
![Coffea1024_5](https://user-images.githubusercontent.com/42876001/180783658-0ddf6d5b-75b7-4a31-82b8-9dae55925323.jpg)
***
