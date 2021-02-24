#!/usr/bin/env python
# coding: utf-8

# `TTbarResCoffeaOutputs` Notebook to produce Coffea output files for an all hadronic $t\bar{t}$ analysis.  The outputs will be found in the corresponding **CoffeaOutputs** directory.

import time
import copy
import scipy.stats as ss
from coffea import hist, processor, nanoevents, util
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoAODSchema, BaseSchema
#import coffea.processor as processor
#from coffea import util
import awkward as ak
import numpy as np
import glob as glob
import itertools
import pandas as pd
from numpy.random import RandomState

from dask.distributed import Client
from lpc_dask import HTCondorCluster
import socket
import time

ak.behavior.update(candidate.behavior)

extra = ['--worker-port 10002:10100']

hostname = socket.gethostname()

cluster = HTCondorCluster(scheduler_options = {'host': f'{hostname}:10000'},
                          cores=1, 
                          memory="4GB", 
                          disk="2GB", 
                          python='python',
                          nanny=False,
                          extra=extra
)

cluster.scale(jobs=10)

client = Client(cluster)

from TTbarResProcessor import TTbarResProcessor

from Filesets import filesets

tstart = time.time()

outputs_unweighted = {}

seed = 1234577890
prng = RandomState(seed)
Chunk = [100000, 100] # [chunksize, maxchunks]

for name,files in filesets.items(): 
    
    print(name)
    output = processor.run_uproot_job({name:files},
                                      treename='Events',
                                      processor_instance=TTbarResProcessor(UseLookUpTables=False,
                                                                           ModMass=False,
                                                                           RandomDebugMode=False,
                                                                           prng=prng),
                                      #executor=processor.dask_executor,
                                      #executor=processor.iterative_executor,
                                      executor=processor.futures_executor,
                                      executor_args={
                                          'client': client,
                                          'savemetrics': True,
                                          #'nano':False, 
                                          #'flatten':False, 
                                          'skipbadfiles':False,
                                          'schema': BaseSchema, #NanoAODSchema,
                                          'align_clusters': True,
                                          'workers': 2},
                                      chunksize=Chunk[0], maxchunks=Chunk[1]
                                     )

    elapsed = time.time() - tstart
    outputs_unweighted[name] = output
    print(output)
    #util.save(output, 'CoffeaOutputs/UnweightedOutputs/TTbarResCoffea_' + name + '_unweighted_output_partial_2021_dask_run.coffea')


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
Chunk = [100000, 100] # [chunksize, maxchunks]

for name,files in filesets_forweights.items(): 
    

    print(name)
    output = processor.run_uproot_job({name:files},
                                      treename='Events',
                                      processor_instance=TTbarResProcessor(UseLookUpTables=True,
                                                                           ModMass = True,
                                                                           RandomDebugMode = False,
                                                                           lu=luts,
                                                                           prng=prng),
                                      #executor=processor.dask_executor,
                                      #executor=processor.iterative_executor,
                                      executor=processor.futures_executor,
                                      executor_args={
                                          'client': client, 
                                          #'savemetrics': True,
                                          'nano':False, 
                                          'flatten':True, 
                                          'skipbadfiles':False,
                                          'schema': processor.NanoEvents,
                                          #'align_clusters': True,
                                          'workers': 2},
                                      chunksize=Chunk[0], maxchunks=Chunk[1]
                                     )

    elapsed = time.time() - tstart
    outputs_weighted[name] = output
    print(output)
    #util.save(output, 'CoffeaOutputs/WeightedModMassOutputs/TTbarResCoffea_' + name + '_ModMass_weighted_output_partial_2021_dask_run.coffea')


print('Elapsed time = ', elapsed, ' sec.')
print('Elapsed time = ', elapsed/60., ' min.')
print('Elapsed time = ', elapsed/3600., ' hrs.') 

for name,output in outputs_weighted.items(): 
    print("-------Unweighted " + name + "--------")
    for i,j in output['cutflow'].items():        
        print( '%20s : %12d' % (i,j) )


