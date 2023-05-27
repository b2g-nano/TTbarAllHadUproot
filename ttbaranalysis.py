# ttbaranalysis.py

from coffea import util
from coffea.nanoevents import NanoAODSchema, BaseSchema
import coffea.processor as processor

import argparse
import time
import json
import os

from dask.distributed import Client, performance_report
from lpcjobqueue import LPCCondorCluster

import warnings
warnings.filterwarnings("ignore")


savedir = 'outputs/'

from ttbarprocessor import TTbarResProcessor

if __name__ == "__main__":
    
    tic = time.time()
        
    parser = argparse.ArgumentParser(
                    prog='ttbaranalysis.py',
                    description='Run ttbarprocessor',
                    epilog='help')
    
    
    
    # datasets to run
    parser.add_argument('-d', '--dataset', choices=['JetHT', 'QCD', 'TTbar'], default='QCD')
    parser.add_argument('--iov', choices=['2016APV', '2016', '2017', '2018'], default='2016APV')
    parser.add_argument('--era', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], action='append', default=[])

    # analysis options
    parser.add_argument('--bkgest', action='store_true')

    # run options
    parser.add_argument('--dask', action='store_true')
    parser.add_argument('--env', choices=['casa', 'lpc', 'winterfell', 'C', 'L', 'W'], default='lpc')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    print('\n------args------')
    for argname, value in vars(args).items(): print(argname, '=', value)
    print('----------------\n')
    
    
    # paramters
    sample = args.dataset
    IOV = args.iov
    useDeepAK8 = True
    

    # get root files
    
    if args.env == 'casa' or args.env == 'C': redirector = 'root://xcache/'
    elif args.env == 'winterfell' or args.env == 'W': redirector = '/mnt/data/cms/'
    else: redirector = 'root://cmsxrootd.fnal.gov/' # default LPC
 
    jsonfiles = {
        "JetHT": 'data/JetHT.json',
        "QCD": 'data/QCD.json',
        "TTbar": 'data/TTbar.json',
    }
    
    inputfile = jsonfiles[sample]
    files = []
    with open(inputfile) as json_file:
        
        data = json.load(json_file)
        
        
        # select files to run over
        filedict = {}
        if 'QCD' in sample:
            filedict[''] = data[IOV]
        else:
            if len(args.era) > 0:
                for era in args.era:
                    if era in data[IOV].keys():
                        filedict[era] = data[IOV][era]
                    else:
                        print(f'{era} not in {IOV}')
            else:
                filedict = data[IOV]
        
        
        # run uproot job
        for subsection, files in filedict.items():
    
    
            # add redirector; select file for testing
            files = [redirector + f for f in files]
            if args.test: files = [files[0]]
            fileset = {sample: files}            

            # coffea output file name
            testString = ''
            bkgString = ''
            subString = subsection.replace('700to', '_700to').replace('1000to','_1000to')
            if args.test: testString = '_test'
            if args.bkgest: bkgString = '_bkgest'

            savefilename = f'{savedir}{sample}_{IOV}{subString}{bkgString}{testString}.coffea'                   
            print(f'running {sample}{subString}')
            
            if args.dask:
                
                if args.env != 'lpc' and args.env != 'L':
                    print('dask currently set up for LPC only')
                    break
                
                uploadfiles_for_dask = [
                    'ttbarprocessor.py',
                    'corrections/corrections.py',
                    'corrections/btagCorrections.py',
                    'corrections/functions.py',
                ]

                cluster = LPCCondorCluster(memory='6GB', transfer_input_files=uploadfiles_for_dask)
                cluster.adapt(minimum=1, maximum=100)
                client = Client(cluster)
                client = Client()
                
                print("Waiting for at least one worker...")
                client.wait_for_workers(1)
                
                exe_args = {
                    "client": client,
                    "skipbadfiles": True,
                    "savemetrics": True,
                    "schema": NanoAODSchema,
                }

                hists, metrics = processor.run_uproot_job(
                    fileset,
                    treename="Events",
                    processor_instance=TTbarResProcessor(iov=IOV,
                                                         bkgEst=args.bkgest,
                                                         useDeepAK8=useDeepAK8,),
                    executor=processor.dask_executor,
                    executor_args=exe_args,
                    chunksize=1000000,
                )

            else:

                exe_args = {
                    "skipbadfiles": True,
                    "savemetrics": True,
                    "schema": NanoAODSchema,
                    "workers":4
                }

                hists, metrics = processor.run_uproot_job(
                    fileset,
                    treename="Events",
                    processor_instance=TTbarResProcessor(iov=IOV,
                                                         bkgEst=args.bkgest,
                                                         useDeepAK8=useDeepAK8,),
                    executor=processor.futures_executor,
                    executor_args=exe_args,
                    chunksize=100000,
                )


            util.save(hists, savefilename)
            print('saving', savefilename)

    elapsed = time.time() - tic
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Events/s: {metrics['entries'] / elapsed:.0f}")



