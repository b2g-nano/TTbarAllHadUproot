#!/usr/bin/env python
# coding: utf-8

# `TTbarResCoffeaOutputs` Notebook to produce Coffea output files for an all hadronic $t\bar{t}$ analysis.  The outputs will be found in the corresponding **CoffeaOutputs** directory.

import os
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
# from lpcjobqueue import LPCCondorCluster

ak.behavior.update(candidate.behavior)

os.chdir('../') # Runs the code from within the working directory without manually changing all directory paths!

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
                                NOTE** UL17 and UL18 samples TBA''')
# ---- Necessary arguments ---- #
StartGroup = Parser.add_mutually_exclusive_group(required=True)
StartGroup.add_argument('-t', '--runtesting', action='store_true', help='Only run a select few root files defined in the code.')
StartGroup.add_argument('-m', '--runmistag', action='store_true',help='Make data mistag rate where ttbar contamination is removed (as well as ttbar mistag rate)')
StartGroup.add_argument('-T', '--runtrigeff', action='store_true', help='Create trigger efficiency hist objects for chosen condition') 
StartGroup.add_argument('-d', '--rundataset', type=str, nargs='+', help='List of datasets to be ran/loaded')

RedirectorGroup = Parser.add_mutually_exclusive_group(required=True)
RedirectorGroup.add_argument('-C', '--casa', action='store_true', help='Use Coffea-Casa redirector: root://xcache/')
RedirectorGroup.add_argument('-L', '--lpc', action='store_true', help='Use CMSLPC redirector: root://cmsxrootd.fnal.gov/')

Parser.add_argument('-a', '--APV', type=str, required=True, choices=['yes', 'no'], help='Do datasets have APV?')
Parser.add_argument('-y', '--year', type=int, required=True, choices=[2016, 2017, 2018, 0], help='Year(s) of data/MC of the datasets you want to run uproot with.  Choose 0 for all years simultaneously.')

# ---- Other arguments ---- #
Parser.add_argument('--uproot', type=int, choices=[1, 2], help='1st run or 2nd run of uproot job.  If not specified, both the 1st and 2nd job will be run one after the other.')
Parser.add_argument('--chunks', type=int, help='Number of chunks of data to run for given dataset(s)')
Parser.add_argument('--chunksize', type=int, help='Size of each chunk to run for given dataset(s)')
Parser.add_argument('--save', action='store_true', help='Choose to save the uproot job as a coffea output for later analysis')
Parser.add_argument('--saveMistag', action='store_true', help='Save mistag rate calculated from running either --uproot 1 or --mistag')
Parser.add_argument('--saveTrig', action='store_true', help='Save uproot job with trigger analysis outputs (Only if -T selected)')
Parser.add_argument('--dask', action='store_true', help='Try the dask executor (experimental) for some fast processing!')

UncertaintyGroup = Parser.add_mutually_exclusive_group()
UncertaintyGroup.add_argument('--bTagSyst', type=str, choices=['central', 'up', 'down'], help='Choose Unc.')
UncertaintyGroup.add_argument('--tTagSyst', type=str, choices=['central', 'up', 'down'], help='Choose Unc.')
UncertaintyGroup.add_argument('--jec', type=str, choices=['central', 'up', 'down'], help='Choose Unc.')
UncertaintyGroup.add_argument('--jer', type=str, choices=['central', 'up', 'down'], help='Choose Unc.')
UncertaintyGroup.add_argument('--pileup', type=str, choices=['central', 'up', 'down'], help='Choose Unc.')

args = Parser.parse_args()

if (args.chunks and not args.chunksize) or (args.chunksize and not args.chunks):
    Parser.error('If either chunks or chunksize is specified, please specify both to run this program.')
    quit()
if args.year != 2016: # This will be removed once other years are ready
    Parser.error('Currently, 2017 and 2018 datasets are not ready for use.  Please stick to 2016 for now.  Thanks!')
    quit()
if args.runmistag and args.uproot:
    Parser.error('When running the --runmistag option do not specify --uproot.')
    quit()
isTrigEffArg = args.runtrigeff
if isTrigEffArg and args.uproot:
    Parser.error('When running any --runtrigeff option do not specify --uproot.')
    quit()
if isTrigEffArg == False and args.saveTrig:
    Parser.error('When not running some --runtrigeff option do not specify --saveTrig.')
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

Redirector = None
if args.casa:
    Redirector = 'root://xcache/'
elif args.lpc:
    Redirector = 'root://cmsxrootd.fnal.gov/'
else:
    print('Redirector not selected properly; code should have terminated earlier!  Terminating now!')
    quit()
#    -------------------------------------------------------    #
VFP = ''
if args.APV == 'yes':
    VFP = 'preVFP'
else:
    VFP = 'postVFP'
convertLabel = {
    'preVFP': 'APV',
    'postVFP': 'noAPV'
}
#    -------------------------------------------------------    #
Testing = args.runtesting
#    -------------------------------------------------------    #
LoadingUnweightedFiles = False 
OnlyCreateLookupTables = False 
if (args.uproot == 1 or args.runmistag) or isTrigEffArg:
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
#    -------------------------------------------------------    # 
Chunk = [args.chunksize, args.chunks] # [chunksize, maxchunks]
#    -------------------------------------------------------    # 

from TTbarResProcessor import TTbarResProcessor

#    -------------------------------------------------------------------------------------------------------------------
#    IIIIIII M     M PPPPPP    OOO   RRRRRR  TTTTTTT     DDDD       A    TTTTTTT    A      SSSSS EEEEEEE TTTTTTT   SSSSS     
#       I    MM   MM P     P  O   O  R     R    T        D   D     A A      T      A A    S      E          T     S          
#       I    M M M M P     P O     O R     R    T        D    D   A   A     T     A   A  S       E          T    S           
#       I    M  M  M PPPPPP  O     O RRRRRR     T        D     D  AAAAA     T     AAAAA   SSSSS  EEEEEEE    T     SSSSS      
#       I    M     M P       O     O R   R      T        D    D  A     A    T    A     A       S E          T          S     
#       I    M     M P        O   O  R    R     T        D   D   A     A    T    A     A      S  E          T         S      
#    IIIIIII M     M P         OOO   R     R    T        DDDD    A     A    T    A     A SSSSS   EEEEEEE    T    SSSSS  
#    -------------------------------------------------------------------------------------------------------------------
namingConvention = 'UL'+str(args.year-2000)+VFP # prefix to help name every MC coffea output according to the selected options
fileConvention = str(args.year) + '/' + convertLabel[VFP] + '/TTbarRes_0l_' # direct the saved coffea output to the appropriate directory
SaveLocation={ # Fill this dictionary with each type of dataset; use this dictionary when saving uproot jobs below
    namingConvention+'_TTbar': 'TT/' + fileConvention,
    namingConvention+'_QCD': 'QCD/' + fileConvention
}
if not Testing:
    filesets_to_run = {}
    from Filesets import CollectDatasets # Filesets.py reads in .root file address locations and stores all in dictionary called 'filesets'
    filesets = CollectDatasets(Redirector)
    if args.rundataset:
        for a in args.rundataset: # for any dataset included as user argument...
            if ('JetHT' in a) and (args.year != 0): 
                filesets_to_run['JetHT'+str(args.year)+'_Data'] = filesets['JetHT'+str(args.year)+'_Data'] # include JetHT dataset read in from Filesets
                SaveLocation['JetHT'+str(args.year)+'_Data'] = 'JetHT/' + str(args.year) + '/TTbarRes_0l_' # file where output will be saved
            elif ('SingleMu' in a) and (args.year != 0): 
                filesets_to_run['SingleMu'+str(args.year)+'_Data'] = filesets['SingleMu'+str(args.year)+'_Data'] # include JetHT dataset read in from Filesets
                SaveLocation['SingleMu'+str(args.year)+'_Data'] = 'SingleMu/' + str(args.year) + '/TTbarRes_0l_' # file where output will be saved
            elif args.year != 0:
                filesets_to_run[namingConvention+'_'+a] = filesets[namingConvention+'_'+a] # include MC dataset read in from Filesets
                if 'RSGluon' in a :
                    SaveLocation[namingConvention+'_'+a] = 'RSGluonToTT/' + fileConvention
                elif 'DM' in a :
                    SaveLocation[namingConvention+'_'+a] = 'ZprimeDMToTTbar/' + fileConvention
    elif args.runmistag: # if args.mistag: Only run 1st uproot job for ttbar and data to get mistag rate with tt contamination removed
        filesets_to_run[namingConvention+'_TTbar'] = filesets[namingConvention+'_TTbar']
        filesets_to_run['JetHT'+str(args.year)+'_Data'] = filesets['JetHT'+str(args.year)+'_Data']
        SaveLocation['JetHT'+str(args.year)+'_Data'] = 'JetHT/' + str(args.year) + '/TTbarRes_0l_'
    elif isTrigEffArg: # 1, 2, 3, or 4; just run over data
        filesets_to_run['SingleMu'+str(args.year)+'_Data'] = filesets['SingleMu'+str(args.year)+'_Data']
        SaveLocation['SingleMu'+str(args.year)+'_Data'] = 'SingleMu/' + str(args.year) + '/TTbarRes_0l_'
    else: # if somehow, the initial needed arguments are not used
        print("Something is wrong.  Please come and infestigate what the problem could be")
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
    
#    ---------------------------------------------------------------------------
#    DDDD       A      SSSSS K     K       SSSSS EEEEEEE TTTTTTT U     U PPPPPP      
#    D   D     A A    S      K   K        S      E          T    U     U P     P     
#    D    D   A   A  S       K K         S       E          T    U     U P     P     
#    D     D  AAAAA   SSSSS  KKk          SSSSS  EEEEEEE    T    U     U PPPPPP      
#    D    D  A     A       S K  K              S E          T    U     U P           
#    D   D   A     A      S  K   K            S  E          T     U   U  P           
#    DDDD    A     A SSSSS   K   K       SSSSS   EEEEEEE    T      UUU   P    
#    ---------------------------------------------------------------------------

ImportFiles = 'TTbarAllHadUproot/nanoAODv9Files/'

if UsingDaskExecutor == True:
    if __name__ == "__main__":
        tic = time.time()
        cluster = LPCCondorCluster(
            ship_env = True,
            transfer_input_files = ImportFiles
        )
        # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=10)
        client = Client(cluster)
        client.restart()
#         client.upload_file('TTbarAllHadUproot/Filesets.py')

#    ---------------------------------------------------------------------------
#    U     U PPPPPP  RRRRRR    OOO     OOO   TTTTTTT       OOO   N     N EEEEEEE     
#    U     U P     P R     R  O   O   O   O     T         O   O  NN    N E           
#    U     U P     P R     R O     O O     O    T        O     O N N   N E           
#    U     U PPPPPP  RRRRRR  O     O O     O    T        O     O N  N  N EEEEEEE     
#    U     U P       R   R   O     O O     O    T        O     O N   N N E           
#     U   U  P       R    R   O   O   O   O     T         O   O  N    NN E           
#      UUU   P       R     R   OOO     OOO      T          OOO   N     N EEEEEEE  
#    ---------------------------------------------------------------------------

tstart = time.time()

outputs_unweighted = {}

seed = 1234577890
prng = RandomState(seed)

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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       triggerAnalysisObjects = isTrigEffArg,
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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       triggerAnalysisObjects = isTrigEffArg,
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
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/'
                          + SaveLocation[name]
                          + name    
                          + '.coffea')
            if isTrigEffArg and args.saveTrig:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForTriggerAnalysis/'
                      + SaveLocation[name]
                      + name    
                      + '_TriggerAnalysis' 
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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       triggerAnalysisObjects = isTrigEffArg,
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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       triggerAnalysisObjects = isTrigEffArg,
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
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/'
                          + SaveLocation[name]
                          + name   
                          + '.coffea')
            if isTrigEffArg and args.saveTrig:
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForTriggerAnalysis/'
                      + SaveLocation[name]
                      + name    
                      + '_TriggerAnalysis' 
                      + '.coffea')
            

    else: # Load files
        output = util.load('TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/'
                           + SaveLocation[name]
                           + name 
                           + '.coffea')

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

#    -----------------------------------------------------------------------------------
#    GGGGGGG EEEEEEE TTTTTTT     M     M IIIIIII   SSSSS TTTTTTT    A    GGGGGGG   SSSSS     
#    G       E          T        MM   MM    I     S         T      A A   G        S          
#    G       E          T        M M M M    I    S          T     A   A  G       S           
#    G  GGGG EEEEEEE    T        M  M  M    I     SSSSS     T     AAAAA  G  GGGG  SSSSS      
#    G     G E          T        M     M    I          S    T    A     A G     G       S     
#    G     G E          T        M     M    I         S     T    A     A G     G      S      
#     GGGGG  EEEEEEE    T        M     M IIIIIII SSSSS      T    A     A  GGGGG  SSSSS  
#    -----------------------------------------------------------------------------------

import TTbarResLookUpTables

from TTbarResLookUpTables import CreateLUTS

mistag_luts = CreateLUTS(filesets_to_run, outputs_unweighted, args.year, VFP, args.runmistag, args.saveMistag)

""" Second uproot job runs the processor with the mistag rates (and flavor effs if desired) and Mass-Modification Procedure """

#    ---------------------------------------------------------------------------
#    U     U PPPPPP  RRRRRR    OOO     OOO   TTTTTTT     TTTTTTT W     W   OOO       
#    U     U P     P R     R  O   O   O   O     T           T    W     W  O   O      
#    U     U P     P R     R O     O O     O    T           T    W     W O     O     
#    U     U PPPPPP  RRRRRR  O     O O     O    T           T    W  W  W O     O     
#    U     U P       R   R   O     O O     O    T           T    W W W W O     O     
#     U   U  P       R    R   O   O   O   O     T           T    WW   WW  O   O      
#      UUU   P       R     R   OOO     OOO      T           T    W     W   OOO    
#    ---------------------------------------------------------------------------

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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_SecondRun/'
                          + SaveLocation[name]
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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_SecondRun/'
                          + SaveLocation[name]
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
