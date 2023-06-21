#!/bin/bash


files=`ls -1`

for file in $files
do

    if [[ "$file" == *"Uncertainty"* ]]; then

        newfile=`echo $file | sed s/\.txt/\.junc\.txt/`
        echo 'moving '$file' to '$newfile
        mv $file $newfile
        
    else
        newfile=`echo $file | sed s/\.txt/\.jec\.txt/`
        echo 'moving '$file' to '$newfile
        mv $file $newfile
    fi
done

