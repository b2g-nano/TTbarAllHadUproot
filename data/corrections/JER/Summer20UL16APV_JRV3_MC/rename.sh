#!/bin/bash


files=`ls -1`

for file in $files
do

    if [[ "$file" == *"Resolution"* ]]; then

        newfile=`echo $file | sed s/\.jer.txt/\.jr\.txt/`
        echo 'moving '$file' to '$newfile
        mv $file $newfile
        
#     else
#         newfile=`echo $file | sed s/\.jersf\.txt/\.txt/`
#         echo 'moving '$file' to '$newfile
#         # mv $file $newfile
    fi
done

