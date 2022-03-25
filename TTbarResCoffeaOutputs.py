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
from coffea import hist, processor, nanoevents, util
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoAODSchema, BaseSchema
from numpy.random import RandomState
from dask.distributed import Client#, Scheduler, SchedulerPlugin
from lpcjobqueue import LPCCondorCluster

ak.behavior.update(candidate.behavior)

Testing = False
LoadingUnweightedFiles = True # don't run processor the first time; just run the processor with the systematics and corrections 
RunAllRootFiles = False # if not, processor will only run over the number of chunks defined by the user
UsingDaskExecutor = False
SaveFirstRun = False # Make a coffea output file of the first uproot job (without the systematics and corrections)
OnlyCreateLookupTables = False # don't run processor the second time; just run the processor the first time without systematics and corrections
systemType = "central" # string for btag SF evaluator --> "central", "up", or "down"
#%------------------------------------------------------------------------------------------------------------------------------------
SaveSecondRun = False # Make a coffea output file of the second uproot job (with the systematics and corrections)

from TTbarResProcessor import TTbarResProcessor

if not Testing:
    from Filesets import filesets, filesets_forweights
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


Chunk = [10000, 5] # [chunksize, maxchunks]

# class TextFileReaderPlugin(SchedulerPlugin):
#     def __init__(self, filename):
#         self.file = open(filename)

# plugin = TextFileReaderPlugin('TTbarAllHadUproot/QCD_UL16_APVv2.txt') 

maindir = 'TTbarAllHadUproot/'
textfiles = [
    maindir+'QCD_UL16_APVv2.txt',
    maindir+'TTJets_BiasedSamples.txt',
    maindir+'TTJets_Mtt-1000toInf_UL16.txt',
    maindir+'TTJets_Mtt-700to1000_UL16.txt',
    maindir+'ZprimeDMToTTbar_UL16.txt',
    maindir+'RSGluonToTT.txt',
    maindir+'JetHT_Data.txt'
]

if UsingDaskExecutor == True:
    if __name__ == "__main__":
        tic = time.time()
        cluster = LPCCondorCluster(
            ship_env = True,
            transfer_input_files = textfiles
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

for name,files in filesets.items(): 
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
                                                                                       ApplySF=False,
                                                                                       UseEfficiencies=False,
                                                                                       prng=prng),
                                                  executor=processor.futures_executor,
                                                  executor_args={
                                                      #'client': client,
                                                      'skipbadfiles':False,
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
                                                                                       ApplySF=False,
                                                                                       UseEfficiencies=False,
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
                          + '_unweighted_output'  
#                           + chosen_exec 
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
                                                                                       ApplySF=False,
                                                                                       UseEfficiencies=False,
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
                                                                                       ApplySF=False,
                                                                                       UseEfficiencies=False,
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
                          + '_unweighted_output' 
#                           + chosen_exec 
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

from TTbarResLookUpTables import luts

#from Filesets import filesets_forweights

# Ensure that the necessary files have been included in the `TTbarResLookUpTables` process before running the next processor, as the mistag procedure is found within that module.  For details about the categories used to write the mistag procedure, refer to the `TTbarResProcessor` module.

""" Runs Processor, Weights Datasets with Corresponding Mistag Weight, Implements Mass Modification Procedure """

tstart = time.time()

outputs_weighted = {}

seed = 1234577890
prng = RandomState(seed)

#UsingDaskExecutor = False

# if UsingDaskExecutor == True:
#     if __name__ == "__main__":
#         tic = time.time()
#         cluster = LPCCondorCluster()
#         # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
#         cluster.adapt(minimum=1, maximum=10)
#         client = Client(cluster)
#         client.restart()
#         client.upload_file('TTbarAllHadUproot/TTbarResProcessor.py')
#         client.upload_file('TTbarAllHadUproot/TTbarResLookUpTables.py')
#         client.upload_file('TTbarAllHadUproot/Filesets.py')

outputs_weighted = {}

for name,files in filesets_forweights.items(): 
    if not OnlyCreateLookupTables:
        print('Processing', name)
        if not RunAllRootFiles:
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                       lu=luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=False,
                                                                                       ApplySF=True,
                                                                                       sysType=systemType,
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
                                                                                       lu=luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=False,
                                                                                       ApplySF=True,
                                                                                       sysType=systemType,
                                                                                       UseEfficiencies=False,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema},
                                                  chunksize=Chunk[0], maxchunks=Chunk[1])
#                 client.restart()
            elapsed = time.time() - tstart
            outputs_weighted[name] = output
            print(output)
            if SaveSecondRun:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/WeightedModMassOutputs/TTbarResCoffea_' 
                          + name 
                          + '_weighted_output'
                          + '_BTagSysType_' 
                          + systemType
#                           + chosen_exec
                          + '.coffea')
            
            
        else: # Run all Root Files
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                       lu=luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=False,
                                                                                       ApplySF=True,
                                                                                       sysType=systemType,
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
#                 client.register_worker_plugin(plugin)
#                 scheduler.add_client(client)
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                       lu=luts,
                                                                                       ModMass=True, 
                                                                                       RandomDebugMode=False,
                                                                                       CalcEff_MC=False,
                                                                                       ApplySF=True,
                                                                                       sysType=systemType,
                                                                                       UseEfficiencies=False,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema})
#                 client.restart()
            elapsed = time.time() - tstart
            outputs_weighted[name] = output
            print(output)
            if SaveSecondRun:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/WeightedModMassOutputs/TTbarResCoffea_' 
                          + name 
                          + '_weighted_output'
                          + '_BTagSysType_' 
                          + systemType
#                           + chosen_exec
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
