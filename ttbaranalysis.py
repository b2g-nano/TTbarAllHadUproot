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
                    epilog='for a test of MC QCD, MC TTbar, JetHT run  "python ttbaranalysis.py --test"')
    
    # datasets to run
    parser.add_argument('-d', '--dataset', choices=['JetHT', 'QCD', 'TTbar', 'ZPrime', 'ZPrimeDM', 'RSGluon'], 
                        action='append', default=['QCD', 'TTbar', 'JetHT'])
    parser.add_argument('--iov', choices=['2016APV', '2016', '2017', '2018'], default='2016')
    
    # choose specific eras, pt bins, mass points
    parser.add_argument('--era', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], action='append', default=[], help='--era A --era B --era C for multiple eras, runs all eras if not specificed')
    parser.add_argument('-p', '--pt', choices=['700to1000', '1000toInf'], action='append', default=[], help='pt bins for TTbar datasets')
    parser.add_argument('-m', '--mass', action='append', default=[], help='mass points for signal')

    # analysis options
    parser.add_argument('--bkgest', action='store_true', help='run with background estimate')
    parser.add_argument('--noSyst', action='store_true', help='run without systematics')


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
    
    
    ##### parameters #####

    samples = args.dataset
    IOV = args.iov
    useDeepAK8 = True
    dask_memory = '3GB' # priority decreases for >2GB memory
    chunksize_dask = 100000
    chunksize_futures = 10000
    maxchunks = 10 if args.test else None
    
    
    
    # transfer function parameters
    f = open('data/corrections/rpf_params_QCD_rpf_fits_1x0.txt').read().split('\n')
    params = {
        'param': [float(param.split('+/-')[0]) for param in [line.split(':')[1] for line in f]],
        'error': [float(param.split('+/-')[1]) for param in [line.split(':')[1] for line in f]],
        'function': '1x0',
    }
    del f

        
    
    ##### systematics and analysis categories #####
    
    #############################################################################################
    #                                                                                           #   
    # systematics options: nominal, jes, jer, pileup, pdf, q2, btag, prefiring, hem             # 
    #                                                                                           #   
    # top tag category options:                                                                 #                     
    #                                                                                           #
    # at        antitag - jet0 is antitagged                                                    #            
    # pret      pretag  - jet0 is top tagged, jet1 top tag is unknown                           #                 
    # 0t        jet0 and jet1 are not top tagged                                                #                
    # 1t        jet0 is top tagged and jet1 is not top tagged                                   #                    
    # 2t        jet0 and jet1 are top tagged                                                    #            
    # AT&P      antitag and probe - jet0 is antitagged and jet1 is top tagged                   #               
    # >=1t      at least 1 top tag in event                                                     #           
    # >=2t      at least 2 top tags in event                                                    #            
    #                                                                                           #             
    #############################################################################################
    
    systematics = [
        'nominal',
        'jes',
        'jer',
        'pileup',
        'pdf',
        'q2',
        'btag',
    ]
    
    if ('2016' in IOV) or ('2017' in IOV): systematics.append('prefiring')
    elif '2018' in IOV: systematics.append('hem')

     
    # make analysis categories 
    ttagcats = ["at", "pret", "2t"]
    btagcats = ["0b", "1b", "2b"]
    ycats = ['cen', 'fwd']
    
    anacats = [ t+b+y for t,b,y in itertools.product( ttagcats, btagcats, ycats) ]
    label_map = {i: label for i, label in enumerate(anacats)}
        
    
    
    # save analysis info #
    with open('out.log', 'w') as f:        
        print('\n------args------', file=f)
        for argname, value in vars(args).items(): print(argname, '=', value, file=f)
        print('----------------\n', file=f)
        print('categories =', label_map, file=f)
        print('\n', file=f)
        if not args.noSyst: print('systematics =', systematics, file=f)
     
    # display analysis info
    print('\n------args------')
    for argname, value in vars(args).items(): print(argname, '=', value)
    if not args.noSyst: print('systematics =', systematics)
    print('----------------\n')

    
    
    ##### root files #####
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
    
    # directories and files for dask
    upload_to_dask = ['data', 'python', 'ttbarprocessor.py']

    
    
    ##### get fileset and run processor #####

    for sample in samples:
    
        inputfile = jsonfiles[sample]
        files = []
        with open(inputfile) as json_file:
            
            
            subsections = args.era + args.mass + args.pt

            data = json.load(json_file)

            # select files to run over
            filedict = {}
            
            # if IOV split into subsections (eras, masses, pt bins)
            try: 
                data[IOV].keys()
                
                # if eras, pt bins, or mass points specified, add files individually
                if len(subsections) > 0:
                    for s in subsections:
                        if s in data[IOV].keys():
                            filedict[s] = data[IOV][s]
                        else:
                            print(f'{s} not in {sample} {IOV}')
                            
                            
                # if eras not specified, get all files for dataset
                else:
                    filedict = data[IOV]
                
            # no subsections in IOV    
            except:
                filedict[''] = data[IOV]
                

            # run uproot job
            for subsection, files in filedict.items():

                # add redirector; select file for testing
                files = [redirector + f for f in files]
                if args.test: files = [files[int(len(files)/2)]]
                fileset = {sample: files}            

                # coffea output file name
                subString = subsection.replace('700to', '_700to').replace('1000to','_1000to')
                if args.bkgest: subString += '_bkgest'
                if args.test: subString += '_test'
                                
                savefilename = f'{savedir}{sample}_{IOV}{subString}.coffea'
                if 'ZPrime' in sample or 'RSGluon' in sample:
                    subString = subString.replace(subsection, '')
                    savefilename = f'{savedir}{sample}{subsection}_{IOV}{subString}.coffea'
                print(f'running {sample} {subsection}')


                # run using futures executor
                if not args.dask:

                    output, metrics = processor.run_uproot_job(
                        fileset,
                        treename="Events",
                        processor_instance=TTbarResProcessor(
                                                             iov=IOV,
                                                             bkgEst=args.bkgest,
                                                             noSyst=args.noSyst,
                                                             useDeepAK8=useDeepAK8,
                                                             anacats=anacats,
                                                             systematics=systematics,
                                                             rpf_params = params,

                                                            ),
                        executor=processor.futures_executor,
                        executor_args={
                                "skipbadfiles": True,
                                "savemetrics": True,
                                "schema": NanoAODSchema,
                                "workers":4
                                },
                        chunksize=chunksize_futures,
                        maxchunks=maxchunks,
                    )


                # run using dask
                else:
                    
                    if args.dask and (args.env == 'lpc' or args.env == 'L'):
                        
                        if args.nocluster:
                            cluster = None
                        else:
                            cluster = LPCCondorCluster(memory=dask_memory, transfer_input_files=upload_to_dask)
                            cluster.adapt(minimum=1, maximum=100)
                    else:
                        cluster = None
        

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
                            chunksize=chunksize_dask,
                            maxchunks=maxchunks,
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
                                                          noSyst=args.noSyst,
                                                          useDeepAK8=useDeepAK8,
                                                          anacats=anacats,
                                                          systematics=systematics,
                                                          rpf_params = params,
                                                          ),
                                                     )
                        
                        client.shutdown()
                        del cluster

                
                output['analysisCategories'] = label_map
                util.save(output, savefilename)
                print('saving', savefilename)

               



    elapsed = time.time() - tic
    print(f"\nFinished in {elapsed:.1f}s")
    print(f"Events/s: {metrics['entries'] / elapsed:.0f}")
    



