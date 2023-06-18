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
    parser.add_argument('-d', '--dataset', choices=['JetHT', 'QCD', 'TTbar', 'ZPrime', 'ZPrimeDM', 'RSGluon'], 
                        action='append', default=['QCD', 'TTbar', 'JetHT'])
    parser.add_argument('--iov', choices=['2016APV', '2016', '2017', '2018'], default='2016')
    parser.add_argument('--era', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], action='append', default=[], help='--era A --era B --era C for multiple eras, runs all eras if not specificed')

    # analysis options
    parser.add_argument('--bkgest', action='store_true')
    parser.add_argument('--syst', action='store_true')


    # run options
    parser.add_argument('--dask', action='store_true')
    parser.add_argument('--env', choices=['casa', 'lpc', 'winterfell', 'C', 'L', 'W'], default='lpc')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-n', '--nocluster', action='store_true', help='use client=Client() if LPCCondorCluster is slow')

    args = parser.parse_args()
    
    # remove defaults if --dataset given
    if len(args.dataset) > 3: args.dataset = args.dataset[3:]
    
    if args.dask and (args.env == 'lpc' or args.env == 'L'):
        from lpcjobqueue import LPCCondorCluster
    
    
    # paramters
    samples = args.dataset
    IOV = args.iov
    useDeepAK8 = True
    
    
    # systematics #
    
    
    systematics = [
        'nominal',
        'pileup',
        'prefiring',
        'pdf',
        'btag',
    ]
    
    
    
    # analysis categories #
    
    # analysis categories #
    # ttagcats = ["AT&Pt", "at", "pret", "0t", "1t", ">=1t", "2t", ">=0t"]
    ttagcats = ["at", "pret", "2t"]
    btagcats = ["0b", "1b", "2b"]
    ycats = ['cen', 'fwd']
    anacats = [ t+b+y for t,b,y in itertools.product( ttagcats, btagcats, ycats) ]



    label_dict = {i: label for i, label in enumerate(anacats)}
    label_to_int_dict = {label: i for i, label in enumerate(anacats)}
        
        
    # save analysis info #
    with open('out.log', 'w') as f:        
        print('\n------args------', file=f)
        for argname, value in vars(args).items(): print(argname, '=', value, file=f)
        print('----------------\n', file=f)
        print('categories =', label_dict, file=f)
        print('\n', file=f)
        print('systematics =', systematics, file=f)
     
    # display analysis info
    for argname, value in vars(args).items(): print(argname, '=', value)

    

    # get root files
    
    if args.env == 'casa' or args.env == 'C': redirector = 'root://xcache/'
    elif args.env == 'winterfell' or args.env == 'W': redirector = '/mnt/data/cms/'
    else: redirector = 'root://cmsxrootd.fnal.gov/' # default LPC
 
    jsonfiles = {
        "JetHT": 'data/nanoAOD/JetHT.json',
        "QCD": 'data/nanoAOD/QCD.json',
        "TTbar": 'data/nanoAOD/TTbar.json',
        "ZPrime": 'data/nanoAOD/ZPrime.json',
        "ZPrimeDM": 'data/nanoAOD/ZPrimeDM.json',
        "RSGluon": 'data/nanoAOD/RSGluon.json',
    }
    
    
    upload_to_dask = [
                        'data',
                        'python',
                        'ttbarprocessor.py',
                    ]
    
    
    
    
        
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
                if args.bkgest: subString += '_bkgest'
                if args.test: subString += '_test'
                

                
                savefilename = f'{savedir}{sample}_{IOV}{subString}.coffea'
                if 'DM' in sample: savefilename = f'{savedir}ZPrime{subString}_{IOV}.coffea'
                if 'RSGluon' in sample: savefilename = f'{savedir}RSGluon{subString}_{IOV}.coffea'
                print(f'running {sample} {subsection}')


                # run using futures executor
                if not args.dask:

                    output, metrics = processor.run_uproot_job(
                        fileset,
                        treename="Events",
                        processor_instance=TTbarResProcessor(
                                                             iov=IOV,
                                                             bkgEst=args.bkgest,
                                                             syst=args.syst,
                                                             useDeepAK8=useDeepAK8,
                                                             anacats=anacats,
                                                             systematics=systematics,

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
                    
                    
                    
                    
                    if args.dask and (args.env == 'lpc' or args.env == 'L'):
                        
                        
                        if args.nocluster:
                            cluster = None
                        else:
                            cluster = LPCCondorCluster(memory='12GB', transfer_input_files=upload_to_dask)
                            cluster.adapt(minimum=1, maximum=100)
                            
                        client=Client(cluster)
        

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


                        if args.nocluster:
                            worker_toc = time.time()
                            print("Waiting for 4 workers...")
                            client.wait_for_workers(4)
                            worker_tic = time.time()
                            
                        else:
                            worker_toc = time.time()
                            print("Waiting for at least one worker...")
                            client.wait_for_workers(1)
                            worker_tic = time.time()
                        
                        print(f'time to wait for worker = {int(worker_tic - worker_toc)}s')

                        output, metrics = run_instance(fileset,
                                                      treename="Events",
                                                      processor_instance=TTbarResProcessor(
                                                          iov=IOV,
                                                          bkgEst=args.bkgest,
                                                          syst=args.syst,
                                                          useDeepAK8=useDeepAK8,
                                                          anacats=anacats,
                                                          systematics=systematics,
                                                          ),
                                                     )
                        
                        client.shutdown()
                        del cluster

                
                output['analysisCategories'] = {i:label for i,label in enumerate(anacats)}
                util.save(output, savefilename)
                print('saving', savefilename)

               



    elapsed = time.time() - tic
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
    


