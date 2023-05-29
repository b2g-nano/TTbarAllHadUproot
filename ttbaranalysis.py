#!/usr/bin/env
# ttbaranalysis.py

from coffea import util
from coffea.nanoevents import NanoAODSchema, BaseSchema
import coffea.processor as processor

import itertools
import argparse
import time
import json
import os

from dask.distributed import Client, performance_report

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
    parser.add_argument('-d', '--dataset', choices=['JetHT', 'QCD', 'TTbar'], action='append', default=['QCD', 'TTbar', 'JetHT'])
    parser.add_argument('--iov', choices=['2016APV', '2016', '2017', '2018'], default='2016')
    parser.add_argument('--era', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], action='append', default=[])

    # analysis options
    parser.add_argument('--bkgest', action='store_true')

    # run options
    parser.add_argument('--dask', action='store_true')
    parser.add_argument('--env', choices=['casa', 'lpc', 'winterfell', 'C', 'L', 'W'], default='lpc')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()
    
    # remove defaults if --dataset given
    if len(args.dataset) > 3: args.dataset = args.dataset[3:]
    
    
    print('\n------args------')
    for argname, value in vars(args).items(): print(argname, '=', value)
    print('----------------\n')
    
    
    # paramters
    samples = args.dataset
    IOV = args.iov
    useDeepAK8 = True
    
    
    
    # analysis categories #
    
    # analysis categories #
    # ttagcats = ["AT&Pt", "at", "pret", "0t", "1t", ">=1t", "2t", ">=0t"]
    ttagcats = ["at", "pret", "2t"]
    btagcats = ["0b", "1b", "2b"]
    ycats = ['cen', 'fwd']
    anacats = [ t+b+y for t,b,y in itertools.product( ttagcats, btagcats, ycats) ]



    label_dict = {i: label for i, label in enumerate(anacats)}
    label_to_int_dict = {label: i for i, label in enumerate(anacats)}


    
 


    # get root files
    
    if args.env == 'casa' or args.env == 'C': redirector = 'root://xcache/'
    elif args.env == 'winterfell' or args.env == 'W': redirector = '/mnt/data/cms/'
    else: redirector = 'root://cmsxrootd.fnal.gov/' # default LPC
 
    jsonfiles = {
        "JetHT": 'data/nanoAOD/JetHT.json',
        "QCD": 'data/nanoAOD/QCD.json',
        "TTbar": 'data/nanoAOD/TTbar.json',
    }
    
    
    upload_to_dask = [
                        'data',
                        'python',
                        'ttbarprocessor.py',
                    ]
    
    if args.dask and (args.env == 'lpc' or args.env == 'L'):

        from lpcjobqueue import LPCCondorCluster
        cluster = LPCCondorCluster(memory='6GB', transfer_input_files=upload_to_dask)
        cluster.adapt(minimum=1, maximum=100)
    
        
    for sample in samples:
    
        inputfile = jsonfiles[sample]
        files = []
        with open(inputfile) as json_file:

            data = json.load(json_file)


            # select files to run over
            filedict = {}
            if 'QCD' in sample:
                filedict[''] = data[IOV]
            else:
                
                # if eras specified, add individually
                if len(args.era) > 0:
                    for era in args.era:
                        if era in data[IOV].keys():
                            filedict[era] = data[IOV][era]
                        else:
                            print(f'{era} not in {IOV}')
                            
                # if eras not specified, get all files in dataset
                else:
                    filedict = data[IOV]


            # run uproot job
            for subsection, files in filedict.items():


                # add redirector; select file for testing
                files = [redirector + f for f in files]
                if args.test: files = [files[int(len(files)/2)]]
                fileset = {sample: files}            

                # coffea output file name
                testString = ''
                bkgString = ''
                subString = subsection.replace('700to', '_700to').replace('1000to','_1000to')
                if args.test: testString = '_test'
                if args.bkgest: bkgString = '_bkgest'

                savefilename = f'{savedir}{sample}_{IOV}{subString}{bkgString}{testString}.coffea'                   
                print(f'running {sample}{subString}')


                # run using futures executor
                if not args.dask:

                    output, metrics = processor.run_uproot_job(
                        fileset,
                        treename="Events",
                        processor_instance=TTbarResProcessor(iov=IOV,
                                                             bkgEst=args.bkgest,
                                                             useDeepAK8=useDeepAK8,
                                                             anacats=anacats,
                                                            ),
                        executor=processor.futures_executor,
                        executor_args={
                                "skipbadfiles": True,
                                "savemetrics": True,
                                "schema": NanoAODSchema,
                                "workers":4
                                },
                        chunksize=100000,
                    )


                # run using dask
                else:

                    # files and directories for dask
                    upload_to_dask = [
                        'data',
                        'python',
                        'ttbarprocessor.py',
                    ]

                    

                    


                    with Client(cluster) as client:

                        run_instance = processor.Runner(
                            metadata_cache={},
                            executor=processor.DaskExecutor(client=client, retries=12,),
                            schema=NanoAODSchema,
                            savemetrics=True,
                            skipbadfiles=True,
                            chunksize=1000000,
                        )


#                         print("Waiting for at least one worker...")
#                         client.wait_for_workers(1)

                        output, metrics = run_instance(fileset,
                                                      treename="Events",
                                                      processor_instance=TTbarResProcessor(
                                                          iov=IOV,
                                                          bkgEst=args.bkgest,
                                                          useDeepAK8=useDeepAK8,
                                                          anacats=anacats,
                                                          ),
                                                     )

                
                output['analysisCategories'] = {i:label for i,label in enumerate(anacats)}
                util.save(output, savefilename)
                print('saving', savefilename)

#                 # save copy for running mass modification
#                 if 'QCD' in sample and not args.bkgest:
#                     util.save(hists, savefilename.replace(savedir, 'data/corrections/backgroundEstimate/'))
#                     print('saving copy to', savefilename.replace(savedir, 'data/corrections/backgroundEstimate/'))




    elapsed = time.time() - tic
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
    
    if args.dask: cluster.close()



