import pandas as pd
import numpy as np
import argparse as ap

Parser = ap.ArgumentParser(prog='bTagSFConvert.py', formatter_class=ap.RawDescriptionHelpFormatter, description='Convert the format of btag SF file to OLD format')

Parser.add_argument('-f', '--file', type=str, required=True, help='name/path to .csv file to be converted')
Parser.add_argument('-s', '--space', action='store_true', help='add space after each comma per line')


args = Parser.parse_args()

def FormatCSV(filename):
    """
    filename ---> String (bTag SF .csv file)
    """
    ############### Ensures that file's workingpoint is written as an integer ##################
    
    workingpoints = {
        'L':0,
        'M':1,
        'T':2,
        'R':3
    }
    
    SF_stuff = [line.split(',') for line in open(filename)] # "Stuff" in the file
    SF_stuff_array = np.array(SF_stuff) # Convert the lines into numpy array (for slicing)
    
    OperatingPoints = SF_stuff_array[1:,0] # Select only the first element of each line ("Operating Points")
    IntegerPoints = [workingpoints[letter] for letter in OperatingPoints] # Convert the strings to integers
    
    dataframe = pd.read_csv(filename) # Prep to swap out the "string" column for the new "integer" column
    dataframe.OperatingPoint = IntegerPoints # Complete the swap
    
    print(filename[:-4]+'_converted.csv')
    ConvertedFile = dataframe.to_csv(filename[:-4]+'_converted.csv', index = False) # Output new converted csv file

    return ConvertedFile

FormatCSV(args.file)

if args.space:
    
    def SpaceAdder(filename):
        oldfile = args.file
        convertedfile = oldfile[:-4]+'_converted.csv'

        dataframe = pd.read_csv(convertedfile)
        spacedfile = dataframe.to_csv(convertedfile[:-4]+'_spaced.csv',  index = False) #  Output new file
        print(convertedfile[:-4]+'_spaced.csv')
        
        array = dataframe.to_numpy()
        np.savetxt(convertedfile[:-4]+'_spaced.csv', array, fmt='%s', delimiter=', ', 
                   header='DeepCSV;OperatingPoint, measurementType, sysType, jetFlavor, etaMin, etaMax, ptMin, ptMax, discrMin, discrMax, formula', 
                   comments='') # forces the extra space after commas
        
        return spacedfile
    
    
    SpaceAdder(args.file[:-4]+'_converted.csv')
# print(args.file + ' has been converted\n\n')