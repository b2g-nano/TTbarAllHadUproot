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
from dask.distributed import Client
from lpcjobqueue import LPCCondorCluster

ak.behavior.update(candidate.behavior)

from TTbarResProcessor import TTbarResProcessor
from Filesets import filesets

LoadingUnweightedFiles = False
UsingDaskExecutor = False

if UsingDaskExecutor == True:
    if __name__ == "__main__":
        tic = time.time()
        cluster = LPCCondorCluster()
        # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=10)
        client = Client(cluster)
        client.upload_file('TTbarAllHadUproot/TTbarResProcessor.py')

tstart = time.time()

outputs_unweighted = {}

seed = 1234577890
prng = RandomState(seed)
#Chunk = [10000, 100] # [chunksize, maxchunks]

for name,files in filesets.items(): 
    if not LoadingUnweightedFiles:
        print('Processing', name)
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
                                              #executor=processor.iterative_executor,
                                              executor=processor.futures_executor,
                                              executor_args={
                                                  'skipbadfiles':False,
                                                  'schema': BaseSchema, #NanoAODSchema,
                                                  'workers': 2})#,
                                              #chunksize=Chunk[0], maxchunks=Chunk[1])
        else:
            chosen_exec = 'dask'
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
                                                  'schema': BaseSchema, #NanoAODSchema,
                                                  'workers': 2})#,
                                              #chunksize=Chunk[0], maxchunks=Chunk[1])

        elapsed = time.time() - tstart
        outputs_unweighted[name] = output
        print(output)
        util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/UnweightedOutputs/TTbarResCoffea_' 
                  + name 
                  + '_unweighted_output' 
#                   + chosen_exec 
                  + '.coffea')

    else:
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

from Filesets import filesets_forweights

# Ensure that the necessary files have been included in the `TTbarResLookUpTables` process before running the next processor, as the mistag procedure is found within that module.  For details about the categories used to write the mistag procedure, refer to the `TTbarResProcessor` module.

""" Runs Processor, Weights Datasets with Corresponding Mistag Weight, Implements Mass Modification Procedure """

tstart = time.time()

seed = 1234577890
outputs_weighted = {}
prng = RandomState(seed)
#Chunk = [100000, 100] # [chunksize, maxchunks]

UsingDaskExecutor = False
OnlyCreateLookupTables = True
for name,files in filesets.items(): 
    if not OnlyCreateLookupTables:
        print('Processing', name)
        if not UsingDaskExecutor:
            chosen_exec = 'futures'
            output = processor.run_uproot_job({name:files},
                                              treename='Events',
                                              processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                   lu=luts,
                                                                                   ModMass=False, 
                                                                                   RandomDebugMode=False,
                                                                                   CalcEff_MC=False,
                                                                                   ApplySF=True,
                                                                                   UseEfficiencies=False,
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
            output = processor.run_uproot_job({name:files},
                                              treename='Events',
                                              processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                                   lu=luts,
                                                                                   ModMass=True, 
                                                                                   RandomDebugMode=False,
                                                                                   CalcEff_MC=False,
                                                                                   ApplySF=True,
                                                                                   UseEfficiencies=False,
                                                                                   prng=prng),
                                              executor=processor.dask_executor,
                                              executor_args={
                                                  'client': client,
                                                  'skipbadfiles':False,
                                                  'schema': BaseSchema, #NanoAODSchema,
                                                  'workers': 2},
                                              chunksize=Chunk[0], maxchunks=Chunk[1])

        elapsed = time.time() - tstart
        outputs_unweighted[name] = output
        print(output)
#         util.save(output, 'TTbarAllHadUproot/CoffeaOutputs/WeightedModMassOutputs/TTbarResCoffea_' 
#                   + name 
#                   + '_ModMass_weighted_output'
#                   + chosen_exec
#                   + '.coffea')
    else:
        continue

print('Elapsed time = ', elapsed, ' sec.')
print('Elapsed time = ', elapsed/60., ' min.')
print('Elapsed time = ', elapsed/3600., ' hrs.') 

if not OnlyCreateLookupTables:
    for name,output in outputs_weighted.items(): 
        print("-------Unweighted " + name + "--------")
        for i,j in output['cutflow'].items():        
            print( '%20s : %12d' % (i,j) )
else:
    print('We\'re done here!!')

#quit()
