#!/bin/bash

textfile=$1
directory=$2

# String path removes /mnt/data/cms from the full directory

stringpath=${directory:13}

cd $directory

for line in *
do
    echo ${stringpath}$line >> $textfile
    
done