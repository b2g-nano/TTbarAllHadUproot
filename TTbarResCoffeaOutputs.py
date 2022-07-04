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
import mplhep as hep
import matplotlib.colors as colors
# from hist.intervals import ratio_uncertainty

ak.behavior.update(candidate.behavior)
maindirectory = os.getcwd()
os.chdir('../') # Runs the code from within the working directory without manually changing all directory paths!

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST

    try:
        os.makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(mypath):
            pass
        else: raise
        
def plotratio2d(numerator, denominator, ax=None, cmap='Blues', cbar=True):
    NumeratorAxes = numerator.axes()
    DenominatorAxes = denominator.axes()
    
    # integer number of bins in this axis #
    NumeratorAxis1_BinNumber = NumeratorAxes[0].size - 3 # Subtract 3 to remove overflow
    NumeratorAxis2_BinNumber = NumeratorAxes[1].size - 3
    
    DenominatorAxis1_BinNumber = DenominatorAxes[0].size - 3 
    DenominatorAxis2_BinNumber = DenominatorAxes[1].size - 3 
    
    if(NumeratorAxis1_BinNumber != DenominatorAxis1_BinNumber 
       or NumeratorAxis2_BinNumber != DenominatorAxis2_BinNumber):
        raise Exception('Numerator and Denominator axes are different sizes; Cannot perform division.')
    else:
        Numerator = numerator.to_hist()
        Denominator = denominator.to_hist()

        ratio = Numerator / Denominator.values()
        
        return hep.hist2dplot(ratio, ax=ax, cmap=cmap, norm=colors.Normalize(0.,1.), cbar=cbar)

def FlavEffList(Flavor, Output, Dataset):
    """
    Flavor ---> string: either 'b', 'c', or 'udsg'
    Output ---> Coffea Object: Output that is returned from running processor
    Dataset ---> string: the dataset string (ex QCD, RSGluon1000, etc...) corresponding to Output
    """
    SaveDirectory = maindirectory + '/FlavorTagEfficiencies/' + Flavor + 'tagEfficiencyTables/'
    mkdir_p(SaveDirectory)
    for subjet in ['s01', 's02', 's11', 's12']:

        eff_numerator = Output[Flavor + '_eff_numerator_' + subjet + '_manualbins'].integrate('dataset', Dataset)
        eff_denominator = Output[Flavor + '_eff_denominator_' + subjet + '_manualbins'].integrate('dataset', Dataset)

        eff = plotratio2d(eff_numerator, eff_denominator) #ColormeshArtists object
        eff_data = eff[0].get_array().data # This is what goes into pandas dataframe
        eff_data = np.nan_to_num(eff_data, nan=0.0) # If eff bin is empty, call it zero

        # ---- Define pt and eta bins from the numerator or denominator hist objects ---- #
        pt_bins = []
        eta_bins = []

        for iden in eff_numerator.identifiers('subjetpt'):
            pt_bins.append(iden)
        for iden in eff_numerator.identifiers('subjeteta'):
            eta_bins.append(iden)

        # ---- Define the Efficiency List as a Pandas Dataframe ---- #
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        EfficiencyList = pd.DataFrame(
                            eff_data,
                            pd.MultiIndex.from_product( [pt_bins, eta_bins], names=['pt', 'eta'] ),
                            ['efficiency']
                        )

        # ---- Save the Efficiency List as .csv ---- #
        filename = dataset + '_' + subjet + '_' + Flavor + 'tageff_large_bins.csv'
        EfficiencyList.to_csv(SaveDirectory+filename)
        print('\nSaved ' + filename)
        
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
StartGroup.add_argument('-T', '--runtrigeff', action='store_true', help='Create trigger efficiency hist coffea output objects for chosen condition') 
StartGroup.add_argument('-F', '--runflavoreff', type=str, nargs='+', help='Create flavor efficiency hist coffea output objects for chosen MC datasets')
StartGroup.add_argument('-d', '--rundataset', type=str, nargs='+', help='List of datasets to be ran/loaded')

RedirectorGroup = Parser.add_mutually_exclusive_group(required=True)
RedirectorGroup.add_argument('-C', '--casa', action='store_true', help='Use Coffea-Casa redirector: root://xcache/')
RedirectorGroup.add_argument('-L', '--lpc', action='store_true', help='Use CMSLPC redirector: root://cmsxrootd.fnal.gov/')

BDiscriminatorGroup = Parser.add_mutually_exclusive_group(required=True)
BDiscriminatorGroup.add_argument('-l', '--loose', action='store_true', help='Apply loose bTag discriminant cut')
BDiscriminatorGroup.add_argument('-med', '--medium', action='store_true', help='Apply medium bTag discriminant cut')

Parser.add_argument('-a', '--APV', type=str, required=True, choices=['yes', 'no'], help='Do datasets have APV?')
Parser.add_argument('-y', '--year', type=int, required=True, choices=[2016, 2017, 2018, 0], help='Year(s) of data/MC of the datasets you want to run uproot with.  Choose 0 for all years simultaneously.')

# ---- Other arguments ---- #
Parser.add_argument('--uproot', type=int, choices=[1, 2], help='1st run or 2nd run of uproot job.  If not specified, both the 1st and 2nd job will be run one after the other.')
Parser.add_argument('--chunks', type=int, help='Number of chunks of data to run for given dataset(s)')
Parser.add_argument('--chunksize', type=int, help='Size of each chunk to run for given dataset(s)')
Parser.add_argument('--save', action='store_true', help='Choose to save the uproot job as a coffea output for later analysis')
Parser.add_argument('--saveMistag', action='store_true', help='Save mistag rate calculated from running either --uproot 1 or --mistag')
Parser.add_argument('--saveTrig', action='store_true', help='Save uproot job with trigger analysis outputs (Only if -T selected)')
Parser.add_argument('--saveFlav', action='store_true', help='Save uproot job with flavor efficiency outputs (Only if -F selected)')
Parser.add_argument('--dask', action='store_true', help='Try the dask executor (experimental) for some fast processing!')
Parser.add_argument('--useEff', action='store_true', help='Use MC bTag efficiencies for bTagging systematics')

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
    Parser.error('When running --runtrigeff option do not specify --uproot.')
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
BDisc = 0.
if args.loose:
    BDisc = 0.1918
else: # args.medium
    BDisc = 0.5847
#    -------------------------------------------------------    #
Testing = args.runtesting
#    -------------------------------------------------------    #
LoadingUnweightedFiles = False 
OnlyCreateLookupTables = False 
if (args.uproot == 1 or args.runmistag) or (isTrigEffArg or args.runflavoreff):
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
method=''
if not args.useEff:
    method='_method2' # Use bTagging systematic method without MC efficiencies and label output accordingly
#    -------------------------------------------------------    #   
SystType = "central" 
UncType = ""
SFfile = ""
if args.bTagSyst:
    UncType = "btagUnc"
    SystType = args.bTagSyst # string for btag SF evaluator --> "central", "up", or "down"
    SFfile = 'TTbarAllHadUproot/CorrectionFiles/SFs/bquark/subjet_btagging.json.gz'
elif args.tTagSyst:
    UncType = "ttagUnc"
    SystType = args.tTagSyst # string for ttag SF correction --> "central", "up", or "down"
elif args.jec:
    UncType = "jec"
    SystType = args.jec # string for ttag SF correction --> "central", "up", or "down"
elif args.jer:
    UncType = "jer"
    SystType = args.jer # string for ttag SF correction --> "central", "up", or "down"
elif args.pileup:
    UncType = "pileup"
    SystType = args.pileup # string for ttag SF correction --> "central", "up", or "down"
SystOpts = ((args.bTagSyst or args.tTagSyst) or (args.jec or args.jer)) or args.pileup
if (not OnlyCreateLookupTables) and (not SystOpts):
    Parser.error('Only run second uproot job with a Systematic application (like --bTagSyst, --jer, etc.)')
    quit()
#    -------------------------------------------------------    # 
Chunk = [args.chunksize, args.chunks] # [chunksize, maxchunks]
#    -------------------------------------------------------    # 

from TTbarResProcessor import TTbarResProcessor
from TTbarResProcessor import TriggerAnalysisProcessor
from TTbarResProcessor import MCFlavorEfficiencyProcessor

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
    elif args.runflavoreff:
        for a in args.runflavoreff: # for any dataset included as user argument...
            if args.year != 0:
                filesets_to_run[namingConvention+'_'+a] = filesets[namingConvention+'_'+a] # include MC dataset read in from Filesets
                if 'RSGluon' in a :
                    SaveLocation[namingConvention+'_'+a] = 'RSGluonToTT/' + fileConvention
                elif 'DM' in a :
                    SaveLocation[namingConvention+'_'+a] = 'ZprimeDMToTTbar/' + fileConvention
    elif args.runmistag: # if args.mistag: Only run 1st uproot job for ttbar and data to get mistag rate with tt contamination removed
        filesets_to_run[namingConvention+'_TTbar'] = filesets[namingConvention+'_TTbar']
        filesets_to_run['JetHT'+str(args.year)+'_Data'] = filesets['JetHT'+str(args.year)+'_Data']
        SaveLocation['JetHT'+str(args.year)+'_Data'] = 'JetHT/' + str(args.year) + '/TTbarRes_0l_'
    elif isTrigEffArg: # just run over data
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

# from coffea_casa import CoffeaCasaCluster
from dask.distributed import Client #, Scheduler, SchedulerPlugin
# ImportFiles = ['TTbarAllHadUproot/CorrectionFiles']
client = None

if UsingDaskExecutor == True and args.casa:
    if __name__ == "__main__":       
        # cluster = CoffeaCasaCluster(
        #     job_extra = {
        #         'transfer_input_files': ImportFiles
        #     }
        # )
        client = Client('tls://ac-2emalik-2ewilliams-40cern-2ech.dask.coffea.casa:8786')
        # client = Client(cluster)
        # client.restart()
        client.upload_file('TTbarAllHadUproot/Filesets.py')
        client.upload_file('TTbarAllHadUproot/TTbarResProcessor.py')
        client.upload_file('TTbarAllHadUproot/TTbarResLookUpTables.py')
        
        
elif UsingDaskExecutor == True and args.lpc:
    # from lpcjobqueue import LPCCondorCluster
    if __name__ == "__main__":  
        tic = time.time()
        cluster = LPCCondorCluster(
            ship_env = True,
            transfer_input_files = ImportFiles
        )
        # minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=10)
        client = Client(cluster)
        # client.restart()
        client.upload_file('TTbarAllHadUproot/Filesets.py')
        client.upload_file('TTbarAllHadUproot/TTbarResProcessor.py')
        client.upload_file('TTbarAllHadUproot/TTbarResLookUpTables.py')
        
        
        
        
#    ----------------------------------------------------------------------------------------------------  
#    U     U PPPPPP  RRRRRR    OOO     OOO   TTTTTTT     FFFFFFF L          A    V     V   OOO   RRRRRR      
#    U     U P     P R     R  O   O   O   O     T        F       L         A A   V     V  O   O  R     R     
#    U     U P     P R     R O     O O     O    T        F       L        A   A  V     V O     O R     R     
#    U     U PPPPPP  RRRRRR  O     O O     O    T        FFFFFFF L        AAAAA  V     V O     O RRRRRR      
#    U     U P       R   R   O     O O     O    T        F       L       A     A  V   V  O     O R   R       
#     U   U  P       R    R   O   O   O   O     T        F       L       A     A   V V    O   O  R    R      
#      UUU   P       R     R   OOO     OOO      T        F       LLLLLLL A     A    V      OOO   R     R   
#    ----------------------------------------------------------------------------------------------------  
        
if args.runflavoreff:
    tstart = time.time()

    outputs_unweighted = {}

    seed = 1234577890
    prng = RandomState(seed)

    for name,files in filesets_to_run.items(): 
        print('Processing', name, '...')
        if not RunAllRootFiles:
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=MCFlavorEfficiencyProcessor(RandomDebugMode=False,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       bdisc=BDisc,
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
                                                  processor_instance=MCFlavorEfficiencyProcessor(RandomDebugMode=False,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       bdisc=BDisc,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema},
                                                  chunksize=Chunk[0], maxchunks=Chunk[1])
                # client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)

            if args.saveFlav:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForMCFlavorAnalysis/'
                          + SaveLocation[name])
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForMCFlavorAnalysis/'
                      + SaveLocation[name]
                      + name    
                      + '_MCFlavorAnalysis' 
                      + '.coffea')


        else: # Run all Root Files
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=MCFlavorEfficiencyProcessor(RandomDebugMode=False,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       bdisc=BDisc,
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
                                                  processor_instance=MCFlavorEfficiencyProcessor(RandomDebugMode=False,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       bdisc=BDisc,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema})
                # client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)

            if args.saveFlav:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForMCFlavorAnalysis/'
                          + SaveLocation[name])
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForMCFlavorAnalysis/'
                      + SaveLocation[name]
                      + name    
                      + '_MCFlavorAnalysis' 
                      + '.coffea')


        print('Elapsed time = ', elapsed, ' sec.')
        print('Elapsed time = ', elapsed/60., ' min.')
        print('Elapsed time = ', elapsed/3600., ' hrs.') 

    for dataset,output in outputs_unweighted.items(): 
        print("-------Unweighted " + dataset + "--------")
        for i,j in output['cutflow'].items():        
            print( '%20s : %1s' % (i,j) ) 
            
        if args.saveFlav:
            FlavEffList('b', output, dataset)
            FlavEffList('c', output, dataset)
            FlavEffList('udsg', output, dataset)
            
    exit() # No need to go further if performing trigger analysis
        
        
        
        
        
        
        


#    -----------------------------------------------------------------------------------------------------------
#    U     U PPPPPP  RRRRRR    OOO     OOO   TTTTTTT     TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR      
#    U     U P     P R     R  O   O   O   O     T           T    R     R    I    G       G       E       R     R     
#    U     U P     P R     R O     O O     O    T           T    R     R    I    G       G       E       R     R     
#    U     U PPPPPP  RRRRRR  O     O O     O    T           T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR      
#    U     U P       R   R   O     O O     O    T           T    R   R      I    G     G G     G E       R   R       
#     U   U  P       R    R   O   O   O   O     T           T    R    R     I    G     G G     G E       R    R      
#      UUU   P       R     R   OOO     OOO      T           T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R       
#    -----------------------------------------------------------------------------------------------------------
        
                      
if isTrigEffArg:
    tstart = time.time()

    outputs_unweighted = {}

    seed = 1234577890
    prng = RandomState(seed)

    for name,files in filesets_to_run.items(): 
        print('Processing', name, '...')
        if not RunAllRootFiles:
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TriggerAnalysisProcessor(RandomDebugMode=False,
                                                                                       bdisc = BDisc,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                                                  processor_instance=TriggerAnalysisProcessor(RandomDebugMode=False,
                                                                                       bdisc = BDisc,
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
                # client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)

            if args.saveTrig:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForTriggerAnalysis/'
                          + SaveLocation[name])
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
                                                  processor_instance=TriggerAnalysisProcessor(RandomDebugMode=False,
                                                                                       bdisc = BDisc,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                                                  processor_instance=TriggerAnalysisProcessor(RandomDebugMode=False,
                                                                                       bdisc = BDisc,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema})
                # client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)

            if args.saveTrig:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForTriggerAnalysis/'
                          + SaveLocation[name])
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForTriggerAnalysis/'
                      + SaveLocation[name]
                      + name    
                      + '_TriggerAnalysis' 
                      + '.coffea')


        print('Elapsed time = ', elapsed, ' sec.')
        print('Elapsed time = ', elapsed/60., ' min.')
        print('Elapsed time = ', elapsed/3600., ' hrs.') 

    for name,output in outputs_unweighted.items(): 
        print("-------Unweighted " + name + "--------")
        for i,j in output['cutflow'].items():        
            print( '%20s : %1s' % (i,j) )        
        
    exit() # No need to go further if performing trigger analysis
    
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
                                                                                       bdisc = BDisc,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                                                                                       bdisc = BDisc,
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
                # client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)
            if SaveFirstRun:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/'
                          + SaveLocation[name])
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/'
                          + SaveLocation[name]
                          + name    
                          + '.coffea')
            
            
        else: # Run all Root Files
            if not UsingDaskExecutor:
                chosen_exec = 'futures'
                output = processor.run_uproot_job({name:files},
                                                  treename='Events',
                                                  processor_instance=TTbarResProcessor(UseLookUpTables=False,
                                                                                       ModMass=False, 
                                                                                       RandomDebugMode=False,
                                                                                       bdisc = BDisc,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
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
                                                                                       bdisc = BDisc,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema})
                # client.restart()

            elapsed = time.time() - tstart
            outputs_unweighted[name] = output
            print(output)
            if SaveFirstRun:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/'
                          + SaveLocation[name])
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/'
                          + SaveLocation[name]
                          + name   
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
        print( '%20s : %1s' % (i,j) )

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

from TTbarResLookUpTables import CreateLUTS, LoadDataLUTS

each_mistag_luts = CreateLUTS(filesets_to_run, outputs_unweighted, args.year, VFP, args.runmistag, args.saveMistag)
mistag_luts = LoadDataLUTS(args.year) # Specifically get data mistag rates with ttContam. corrections

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
                                                                                       ApplybtagSF=True,
                                                                                       sysType=SystType,
                                                                                       ScaleFactorFile=SFfile,
                                                                                       UseEfficiencies=args.useEff,
                                                                                       bdisc = BDisc,
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
                                                                                       ScaleFactorFile=SFfile,
                                                                                       UseEfficiencies=args.useEff,
                                                                                       bdisc = BDisc,
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
                # client.restart()
            elapsed = time.time() - tstart
            outputs_weighted[name] = output
            print(output)
            if SaveSecondRun:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_SecondRun/'
                          + SaveLocation[name])
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_SecondRun/'
                          + SaveLocation[name]
                          + name 
                          + '_weighted_'
                          + UncType + '_' 
                          + SystType
                          + method
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
                                                                                       ScaleFactorFile=SFfile,
                                                                                       UseEfficiencies=args.useEff,
                                                                                       bdisc = BDisc,
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
                                                                                       ScaleFactorFile=SFfile,
                                                                                       UseEfficiencies=args.useEff,
                                                                                       bdisc = BDisc,
                                                                                       year=args.year,
                                                                                       apv=convertLabel[VFP],
                                                                                       vfp=VFP,
                                                                                       prng=prng),
                                                  executor=processor.dask_executor,
                                                  executor_args={
                                                      'client': client,
                                                      'skipbadfiles':False,
                                                      'schema': BaseSchema})
                # client.restart()
            elapsed = time.time() - tstart
            outputs_weighted[name] = output
            print(output)
            if SaveSecondRun:
                mkdir_p('TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_SecondRun/'
                          + SaveLocation[name])
                util.save(output, 'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_SecondRun/'
                          + SaveLocation[name]
                          + name 
                          + '_weighted_'
                          + UncType + '_' 
                          + SystType
                          + method
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
            print( '%20s : %1s' % (i,j) )
    print("\n\nWe\'re done here!!")
else:
    print('\n\nWe\'re done here!!')

#quit()
