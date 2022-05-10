#!/usr/bin/python

import subprocess
import os,sys
from optparse import OptionParser

parser = OptionParser(usage="usage: %prog [options] \nrun with --help to get list of options")
parser.add_option("-q","--query",dest="query",default="",type="string",help="DAS query you want to make.")
parser.add_option("-n","--name",dest="name",default="",type="string",help="Choose the name of the output file.")
(options, args) = parser.parse_args()

datasets = [

    "/RSGluonToTT*/RunIISummer20UL16*NanoAODAPV*v9-106X*/NANOAODSIM"
    
]

def doDatasetQuery(query): #query for the querys you are interested in
    print("Doing the das query for {}".format(query))
    datasets = []
    p = subprocess.Popen("dasgoclient -query=\"dataset={0}\"".format(query),shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()
    datasets = out.split("\n")
    datasets = datasets[:-1] #for some reason the last element of the list is an empty string, remove it
    return datasets

def doFileQuery(outputfile, dataset): #query for the files in each dataset
    files = []
    p = subprocess.Popen("dasgoclient -query=\"file dataset={0}\"".format(dataset),shell=True,stdout=subprocess.PIPE)
    out,err = p.communicate()
    files = out.split("\n")
    files = files[:-1] #for some reason the last element of the list is an empty string, remove it
    for element in files: outputfile.write(element+"\n")
    print("Done.")


if __name__ == "__main__":
    welcome_str_1 = "Did you do \'cmsenv\' and \'voms-proxy-init --voms cms\' before running this script?"
    welcome_str_2 = "If not, this script won't work"
    print("="*len(welcome_str_1))
    print("="*len(welcome_str_1))
    print(welcome_str_1)
    print(welcome_str_2)
    print("="*len(welcome_str_1))
    print("="*len(welcome_str_1))
    
    datasets = doDatasetQuery(options.query)
    #print("datasets:", datasets)
    with open(options.name+".txt", "w") as outputfile:
        for idx, dataset in enumerate(datasets):
            print "Processing {0}".format(dataset)
            doFileQuery(outputfile, dataset)
    
