#!/usr/bin/env python
# coding: utf-8

# `TTbarResCoffeaOutputs` Notebook to produce Coffea output files for an all hadronic $t\bar{t}$ analysis.  The outputs will be found in the corresponding **CoffeaOutputs** directory.

import time
import copy
import itertools
import scipy.stats as ss
import awkward as ak
import numpy as np
import glob as glob
import pandas as pd
import argparse as ap
from coffea import hist, processor, nanoevents, util
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoAODSchema, BaseSchema
from numpy.random import RandomState
from dask.distributed import Client#, Scheduler, SchedulerPlugin
from lpcjobqueue import LPCCondorCluster

ak.behavior.update(candidate.behavior)

#    -----------------------------------------------
#    PPPPPP     A    RRRRRR    SSSSS EEEEEEE RRRRRR      
#    P     P   A A   R     R  S      E       R     R     
#    P     P  A   A  R     R S       E       R     R     
#    PPPPPP   AAAAA  RRRRRR   SSSSS  EEEEEEE RRRRRR      
#    P       A     A R   R         S E       R   R       
#    P       A     A R    R       S  E       R    R      
#    P       A     A R     R SSSSS   EEEEEEE R     R 
#    -----------------------------------------------

# Parser = ap.ArgumentParser(prog='TTbarResCoffeaOutputs.py', description='something')
Parser = ap.ArgumentParser(prog='TTbarResCoffeaOutputs.py', formatter_class=ap.RawDescriptionHelpFormatter, description='''\
-----------------------------------------------------------------------------
Run the TTbarAllHadProcessor script.  
All objects for each dataset ran can be saved as its own .coffea output file.
-----------------------------------------------------------------------------''', 
                                epilog='''\
                                Available List of Dataset Strings:
                                Key:
                                -------------------------------------------------------------------------------
                                <sim.year> = 16, 17, or 18
                                <x> = integer from [1, 5]
                                <y> = integer either 0 or 5 
                                <x> = <y> = 5 is not an available string to be included in dataset string names
                                <year> = 2016, 2017, or 2018
                                -------------------------------------------------------------------------------
                                UL<sim.year>_QCD
                                UL<sim.year>_DM<x><y>00
                                UL<sim.year>_RSGluon<x><y>00
                                UL<sim.year>_TTbar
                                JetHT<year>_Data
                                JetHT
                                NOTE** UL17 and UL18 TBA''')
# ---- Necessary arguments ---- #
Parser.add_argument('--dataset', type=str, required=True, nargs='+', help='List of datasets to be ran/loaded')

# ---- Other arguments ---- #
Parser.add_argument('--testing', type=bool, default=False, help='Only run a select few root files defined in the code')
Parser.add_argument('--uproot', type=int, choices=[1, 2], help='1st run or 2nd run of uproot job.  Enter either 1 or 2 accordingly.  If not specified, both the 1st and 2nd job will be run one after the other.')
Parser.add_argument('--chunks', type=int, help='Number of chunks of data to run for given dataset(s)')
Parser.add_argument('--chunksize', type=int, help='Size of each chunk to run for given dataset(s)')
Parser.add_argument('--save', type=bool, default=False, help='Choose to save the uproot job as a coffea output for later analysis')
Parser.add_argument('--loadMistag', type=bool, default=False, help='Load mistag rate Look Up Tables instead of recreating them after 1st uproot job.')
Parser.add_argument('--loadTTbar', type=bool, default=False, help='Load 1st uproot run ttbar coffea file while running 1st JetHT uproot job.  This ensures that ttbar contamination can be removed from JetHT mistag rate without needing to run the 1st uproot ttbar MC job a second time.  Note that you must already have a 1st run ttbar coffea file to use this option.')
Parser.add_argument('--dask', type=bool, default=False, help='Try the dask executor (experimental) for some fast processing!')

UncertaintyGroup = Parser.add_mutually_exclusive_group()
UncertaintyGroup.add_argument('-b', '--bTagSyst', type=str, choices=['central', 'up', 'down'], help='Either \'central\', \'up\', \'down\'')
UncertaintyGroup.add_argument('-t', '--tTagSyst', type=str, choices=['central', 'up', 'down'], help='Either \'central\', \'up\', \'down\'')
UncertaintyGroup.add_argument('--jec', type=str, choices=['central', 'up', 'down'], help='Either \'central\', \'up\', \'down\'')
UncertaintyGroup.add_argument('--jer', type=str, choices=['central', 'up', 'down'], help='Either \'central\', \'up\', \'down\'')
UncertaintyGroup.add_argument('--pileup', type=str, choices=['central', 'up', 'down'], help='Either \'central\', \'up\', \'down\'')

args = Parser.parse_args()

if (args.chunks and not args.chunksize) or (args.chunksize and not args.chunks):
    Parser.error('If either chunks or chunksize is specified, please specify both to run this program.')
    quit()

#    -------------------------------------------------------
#      OOO   PPPPPP  TTTTTTT IIIIIII   OOO   N     N   SSSSS     
#     O   O  P     P    T       I     O   O  NN    N  S          
#    O     O P     P    T       I    O     O N N   N S           
#    O     O PPPPPP     T       I    O     O N  N  N  SSSSS      
#    O     O P          T       I    O     O N   N N       S     
#     O   O  P          T       I     O   O  N    NN      S      
#      OOO   P          T    IIIIIII   OOO   N     N SSSSS   
#    -------------------------------------------------------
    
Testing = args.testing
#    -------------------------------------------------------    #
LoadingUnweightedFiles = False 
OnlyCreateLookupTables = False 
if args.uproot == 1:
    OnlyCreateLookupTables = True # stop the code after LUTs are displayed on the terminal; after 1st uproot job
elif args.uproot == 2:
    LoadingUnweightedFiles = True # Load the 1st uproot job's coffea outputs if you only want to run the 2nd uproot job.
else: # Default for running both 1st and 2nd uproot job
    LoadingUnweightedFiles = False 
    OnlyCreateLookupTables = False 
#    -------------------------------------------------------    #    
RunAllRootFiles = False 
if not args.chunks:
    RunAllRootFiles = True
#    -------------------------------------------------------    #    
UsingDaskExecutor = args.dask
#    -------------------------------------------------------    #
SaveFirstRun = False
SaveSecondRun = False
if args.save:
    SaveFirstRun = True # Make a coffea output file of the first uproot job (without the systematics and corrections)
    SaveSecondRun = True # Make a coffea output file of the second uproot job (with the systematics and corrections)
#    -------------------------------------------------------    #    
SystType = "central" 
UncType = ""
if args.bTagSyst:
    UncType = "btagUnc"
    SystType = args.bTagSyst # string for btag SF evaluator --> "central", "up", or "down"
if args.tTagSyst:
    UncType = "ttagUnc"
    SystType = args.tTagSyst # string for ttag SF correction --> "central", "up", or "down"



from TTbarResProcessor import TTbarResProcessor

if not Testing:
    filesets_to_run = {}
    from Filesets import filesets # Filesets.py reads in .root file address locations and stores all in dictionary called 'filesets'
    for a in args.dataset: # for any dataset included as user argument...
        filesets_to_run[a] = filesets[a] # include that dataset read in from Filesets module
    
else:
    TestRootFiles = [#"TTbarAllHadUproot/SMttbar_nEvents10.root",
                     #"TTbarAllHadUproot/SMttbar_nEvents10000.root",
                     #"TTbarAllHadUproot/SMttbar_nEvents60000.root",
                     "TTbarAllHadUproot/ttbar_Mtt-1000toInf_nEvents50000.root"]
                     #"TTbarAllHadUproot/QCD_pt600to800_customNano_numEvents79200.root"]
    filesets = {
        'TestSample_ttbarRES':TestRootFiles
    }
    filesets_forweights = filesets
    
# print(filesets)

Chunk = [args.chunksize, args.chunks] # [chunksize, maxchunks]

# class TextFileReaderPlugin(SchedulerPlugin):
#     def __init__(self, filename):
#         self.file = open(filename)

# plugin = TextFileReaderPlugin('TTbarAllHadUproot/QCD_UL16_APVv2.txt') 

maindir = 'TTbarAllHadUproot/'
# textfiles = [
#     maindir+'QCD_UL16_APVv2.txt',
#     maindir+'TTJets_BiasedSamples.txt',
#     maindir+'TTJets_Mtt-1000toInf_UL16.txt',
#     maindir+'TTJets_Mtt-700to1000_UL16.txt',
#     maindir+'ZprimeDMToTTbar_UL16.txt',
#     maindir+'RSGluonToTT.txt',
#     maindir+'JetHT_Data.txt'
# ]

if UsingDaskExecutor == True:
    if __name__ == "__main__":
        tic = time.time()
        cluster = LPCCondorCluster(
            ship_env = True,
#             transfer_input_files = textfiles
        )
        # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=10)
        client = Client(cluster)
#         scheduler = Scheduler()
        client.restart()
#         client.upload_file('TTbarAllHadUproot/QCD_UL16_APVv2.txt')
        client.upload_file('TTbarAllHadUproot/Filesets.py')
        client.upload_file('TTbarAllHadUproot/TTbarResLookUpTables.py')
        client.upload_file('TTbarAllHadUproot/TTbarResProcessor.py')
#         scheduler.add_plugin(plugin)

tstart = time.time()

outputs_unweighted = {}

seed = 1234577890
prng = RandomState(seed)

print('Filesets to run should only be the datasets specified by the user:\n')
print(filesets_to_run)
print()

for name,files in filesets_to_run.items(): 
    if not LoadingUnweightedFiles:
        print('Processing', name, '...')
        if not RunAllRootFiles:
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=False,
                                                                                       ModMass=False, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=True,
                                                                                       prng=prng),
                                                  executor=processor.futures_executor,
                                                  executor_args={
                                                      #'client': client,
                                                      'skipbadfiles':True,
                                                      'schema': BaseSchema, #NanoAODSchema,
                                                      'workers': 2},
                                                  chunksize=Chunk[0], maxchunks=Chunk[1])
            else: # use dask
                chosen_exec = 'dask'
                client.wait_for_workers(1)
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=False,
                                                                                       ModMass=False, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=True,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema},
                                                  chunksize=Chunk[0], maxchunks=Chunk[1])
                client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)
            if SaveFirstRun:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/UnweightedOutputs/TTbarResCoffea_' 
                          + name 
                          + '_'   
                          + '.coffea')
            
            
        else: # Run all Root Files
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=False,
                                                                                       ModMass=False, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=True,
                                                                                       prng=prng),
                                                  executor=processor.futures_executor,
                                                  executor_args={
                                                      #'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema, #NanoAODSchema,
                                                      'workers': 2})

            else: # use dask
                chosen_exec = 'dask'
                client.wait_for_workers(1)
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=False,
                                                                                       ModMass=False, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=True,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema})
                client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)
            if SaveFirstRun:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/UnweightedOutputs/TTbarResCoffea_' 
                          + name 
                          + '_'  
                          + '.coffea')
            

    else: # Load files
        output = util.load('TTbarAllHadUproot/CoffeaOutputs/UnweightedOutputs/TTbarResCoffea_' 
                           + name 
                           + '_unweighted_output.coffea')

        outputs_unweighted[name] = output
        print(name + ' unweighted output loaded')
        elapsed = time.time() - tstart

    print('Elapsed time = ', elapsed, ' sec.')
    print('Elapsed time = ', elapsed/60., ' min.')
    print('Elapsed time = ', elapsed/3600., ' hrs.') 

for name,output in outputs_unweighted.items(): 
    print("-------Unweighted " + name + "--------")
    for i,j in output['cutflow'].items():        
        print( '%20s : %12d' % (i,j) )

# First, run the `TTbarResLookUpTables` module by simply importing it.  If it works, it will print out varies pandas dataframes with information about the mistag rates and finally print the `luts` multi-dictionary

import TTbarResLookUpTables

# Next, import that multi-dictionary `luts`, as it is needed for the processor to create output files.  These new output files will have the necessary datasets weighted by their corresponding mistag rate

from TTbarResLookUpTables import CreateLUTS

mistag_luts = CreateLUTS(filesets_to_run, outputs_unweighted, args.loadMistag, args.loadTTbar)

#from Filesets import filesets_forweights

# Ensure that the necessary files have been included in the `TTbarResLookUpTables` process before running the next processor, as the mistag procedure is found within that module.  For details about the categories used to write the mistag procedure, refer to the `TTbarResProcessor` module.

""" Runs Processor, Weights Datasets with Corresponding Mistag Weight, Implements Mass Modification Procedure """

tstart = time.time()

outputs_weighted = {}

seed = 1234577890
prng = RandomState(seed)

outputs_weighted = {}

for name,files in filesets_to_run.items(): 
    if not OnlyCreateLookupTables:
        print('Processing', name)
        if not RunAllRootFiles:
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                       lu=mistag_luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       ApplybtagSF=False,
                                                                                       sysType=SystType,
                                                                                       UseEfficiencies=True,
                                                                                       prng=prng),
                                                  #executor=processor.iterative_executor,
                                                  executor=processor.futures_executor,
                                                  executor_args={
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema, #NanoAODSchema,
                                                      'workers': 2},
                                                  chunksize=Chunk[0], maxchunks=Chunk[1])
            else:
                chosen_exec = 'dask'
                client.wait_for_workers(1)
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                       lu=mistag_luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       ApplybtagSF=True,
                                                                                       sysType=SystType,
                                                                                       UseEfficiencies=True,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema},
                                                  chunksize=Chunk[0], maxchunks=Chunk[1])
                client.restart()
            elapsed = time.time() - tstart
            outputs_weighted[name] = output
            print(output)
            if SaveSecondRun:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/WeightedModMassOutputs/TTbarResCoffea_' 
                          + name 
                          + '_weighted_'
                          + UncType + '_' 
                          + SystType
                          + '.coffea')
            
            
        else: # Run all Root Files
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                       lu=mistag_luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       ApplybtagSF=True,
                                                                                       sysType=SystType,
                                                                                       UseEfficiencies=False,
                                                                                       prng=prng),
                                                  #executor=processor.iterative_executor,
                                                  executor=processor.futures_executor,
                                                  executor_args={
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema, #NanoAODSchema,
                                                      'workers': 2})

            else:
                chosen_exec = 'dask'
                client.wait_for_workers(1)
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                       lu=mistag_luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       ApplybtagSF=True,
                                                                                       sysType=SystType,
                                                                                       UseEfficiencies=True,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema})
                client.restart()
            elapsed = time.time() - tstart
            outputs_weighted[name] = output
            print(output)
            if SaveSecondRun:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/WeightedModMassOutputs/TTbarResCoffea_' 
                          + name 
                          + '_weighted_'
                          + UncType + '_' 
                          + SystType
                          + '.coffea')
    else:
        continue

    print('Elapsed time = ', elapsed, ' sec.')
    print('Elapsed time = ', elapsed/60., ' min.')
    print('Elapsed time = ', elapsed/3600., ' hrs.') 

if not OnlyCreateLookupTables:
    for name,output in outputs_weighted.items(): 
        print("-------Weighted " + name + "--------")
        for i,j in output['cutflow'].items():        
            print( '%20s : %12d' % (i,j) )
    print("\n\nWe\'re done here!!")
else:
    print('\n\nWe\'re done here!!')

#quit()
