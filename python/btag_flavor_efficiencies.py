import numpy as np
import pandas as pd
import awkward as ak


def BtagUpdater(subjet, Eff_filename_list, ScaleFactorFilename, FittingPoint, OperatingPoint, bdisc):  
    """
    subjet (Flattened Awkward Array)       ---> One of the Four preselected subjet awkward arrays (e.g. SubJet01)
    Eff_filename_list (Array of strings)   ---> List of imported b-tagging efficiency files of the selected subjet (corresponding to the hadron flavour of the subjet)
    ScaleFactorFilename (string)           ---> CSV file containing info to evaluate scale factors with
    FittingPoint (string)                  ---> "loose"  , "medium", "tight"
    OperatingPoint (string)                ---> "central", "up"    , "down"
    """
    # ---- Declare flattened pT and Eta variables ---- #
    pT = np.asarray(ak.flatten(subjet.p4.pt))
    Eta = np.asarray(ak.flatten(subjet.p4.eta))

    # ---- Import Flavor Efficiency Tables as Dataframes ---- #
    subjet_flav_index = np.arange(ak.to_numpy(subjet.hadronFlavour).size)
    df_list = [ pd.read_csv(Eff_filename_list[i]) for i in subjet_flav_index ] # List of efficiency dataframes; imported to extract list of eff_vals
    # print(df_list[0])
    eff_vals_list = [ df_list[i]['efficiency'].values for i in subjet_flav_index ] # efficiency values for each file read in; one file per element of subjet array
    # print(eff_vals_list[0])

    # ---- Match subjet pt and eta to appropriate bins ---- #
    pt_BinKeys = np.arange(np.array(manual_subjetpt_bins).size - 1) # the -1 ensures proper size for bin labeling
    eta_BinKeys = np.arange(np.array(manual_subjeteta_bins).size - 1) # the -1 ensures proper size for bin labeling
    pt_Bins = np.array(manual_subjetpt_bins)
    eta_Bins = np.array(manual_subjeteta_bins)

    # ---- Usable pt and eta bin indices ---- #
    pt_indices = np.digitize(pT, pt_Bins, right=True) - 1 # minus one because digitize labels first element as 1 instead of 0
    eta_indices = np.digitize(Eta, eta_Bins, right=True) - 1

    pt_indices = np.where(pt_indices == pt_BinKeys.size, pt_indices-1, pt_indices) # if value is larger than largest bin, bin number will be defaulted to largest bin
    eta_indices = np.where(eta_indices == eta_BinKeys.size, eta_indices-1, eta_indices)

    pt_indices = np.where(pt_indices < 0, 0, pt_indices) # if value is less than smallest bin, bin number will be defaulted to smallest bin (zeroth)
    eta_indices = np.where(eta_indices < 0, 0, eta_indices)

    # ---- Pair the indices together ---- #
    index_pairs = np.vstack((pt_indices, eta_indices)).T  # Pairs of pt and eta bin indices to be mapped to corresponding efficiency bin number
    index_pairs_tuples = [tuple(e) for e in index_pairs] # This can be indexed easily for reading from dictionary

    # ---- Get Efficiencies from  ---- #
    eff_BinKeys_comb = CartesianProduct(pt_BinKeys, eta_BinKeys) #List of Combined pt and eta keys (should be 40 of them)
    effBinKeys = np.arange( len(eff_BinKeys_comb) )
    EffKeys_Dict = dict(zip([tuple(eff_BinKeys_comb[i]) for i in effBinKeys], effBinKeys)) # Mapping combined pt and eta keys to a single integer (for boradcasting)
    Eff_indices = [EffKeys_Dict[index_pairs_tuples[i]] for i in range(pt_indices.size)] # Indices for selecting efficiency values from the lists for each subjet index

    eff_val = np.asarray([ eff_vals_list[i][Eff_indices[i]] for i in subjet_flav_index ])
    # print(eff_val)

    """
                                !! NOTE !!
            Some efficiency values (eff_val array elements) may be zero
            and must be taken into account when dividing by the efficiency
    """

    ###############  Btag Update Method ##################
    #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods
    #https://github.com/rappoccio/usercode/blob/Dev_53x/EDSHyFT/plugins/BTagSFUtil_tprime.h

    coin = np.random.uniform(0,1,len(subjet)) # used for randomly deciding which jets' btag status to update or not
    subjet_btag_status = np.asarray((subjet.btagCSVV2 > bdisc)) # do subjets pass the btagger requirement

    '''
*******************************************************************************************************************
                Correction Library Logic for Applying Subjet Scale Factors
                -----------------------------------------------------------
        1.) Declare CorrectionSet object by importing the desired JSON file
                CorrectionlibObject = correctionlib.CorrectionSet.from_file(<name and/or path of file (.json.gz)>)
        2.) Flatten the subjet's flavor, pt and eta arrays
        3.) Convert these arrays to Numpy arrays (as Correctionlib hates Awkward Arrays)
        4.) Split flavor array into two arrays of the same length:
            i.) Pretend the entire array were only heavy quarks (b and c) by replacing light quraks with c quarks
                [4 4 0 5 4 0 0 5 5] ---> [4 4 4 5 4 4 4 5 5]
            ii.) Pretend the entire array were only light quarks by replacing b and c with light quarks
                [4 4 0 5 4 0 0 5 5] ---> [0 0 0 0 0 0 0 0 0]
        5.) Check the "name" of the tagging corrections at the beginning of the JSON file
                {
                  "schema_version": 2,
                  "description": "This json file contains the corrections for deepCSV subjet tagging. ",
                  "corrections": [
                    {
                      "name": "deepCSV_subjet",
        6.) Fill in the required "inputs" in the order shown in the JSON file
                "inputs": [
                    {
                      "name": "systematic",
                      "type": "string"
                    },
                    {
                      "name": "method",
                      "type": "string",
                      "description": "incl for light jets, lt for b/c jets"
                    },
                    {
                      "name": "working_point",
                      "type": "string",
                      "description": "L/M"
                    },
                    {
                      "name": "flavor",
                      "type": "int",
                      "description": "hadron flavor definition: 5=b, 4=c, 0=udsg"
                    },
                    {
                      "name": "abseta",
                      "type": "real"
                    },
                    {
                      "name": "pt",
                      "type": "real"
                    }
                  ],
        7.) Create one Scale Factor array by comparing both 'pretend' arrays with the original flavor array
            if flavor array is 0, use scale factors evaluated with 'pretend' light quark array
            otherwise, use scale factors evaluated with 'pretend' heavy quark array
                For Original Flavor Array [4 4 0 5 4 0 0 5 5]:
                0 ---> use scale factor element made from [0 0 0 0 0 0 0 0 0]
                4 ---> use scale factor element made from [4 4 4 5 4 4 4 5 5]
                5 ---> use scale factor element made from [4 4 4 5 4 4 4 5 5]

*******************************************************************************************************************
    ''' 
    # Step 1.)
    btag_sf = correctionlib.CorrectionSet.from_file(ScaleFactorFilename)
    # Step 2.) and 3.)
    hadronFlavour = ak.to_numpy(ak.flatten(subjet.hadronFlavour))
    eta = ak.to_numpy(ak.flatten(subjet.eta))
    pt = ak.to_numpy(ak.flatten(subjet.pt))
    # ---- Ensure eta and pt fall within the allowed binning for corrections ---- #
    Min_etaval = 0.
    Max_etaval = 2.5
    Min_ptval = 30.
    Max_ptval = 450.

    eta = np.where(abs(eta)>=Max_etaval, Max_etaval-0.500, eta)
    pt = np.where(pt<=Min_ptval, Min_ptval+1.00, pt)
    pt = np.where(pt>=Max_ptval, Max_ptval-1.00, pt)

    # Step 4.)
    allHeavy = np.where(hadronFlavour == 0, 4, hadronFlavour)
    allLight = np.zeros_like(allHeavy) 
    # Step 5.) and 6.)
    BSF_allHeavy = btag_sf['deepCSV_subjet'].evaluate(OperatingPoint, 'lt', FittingPoint, allHeavy, abs(eta), pt)
    BSF_allLight = btag_sf['deepCSV_subjet'].evaluate(OperatingPoint, 'incl', FittingPoint, allLight, abs(eta), pt)
    # Step 7.)
    BSF = np.where(hadronFlavour == 0, BSF_allLight, BSF_allHeavy) # btag scale factors
    # print(BSF)

    """
*******************************************************************************************************************        
                    Does the Subjet Pass the Discriminator Cut?
                   ---------------------------------------------
                  True                                      False
                  ----                                      -----
    | SF = 1  |  SF < 1  |  SF > 1  |         |  SF = 1  |  SF < 1  |  SF > 1  |

    |    O    |Downgrade?|    O     |         |     X    |    X     | Upgrade? |
               ----------                                             --------
    |         |True|False|          |         |          |          |True|False|
                ---  ---                                              ---  ---
    |         |  X |  O  |          |         |          |          |  O |  X  |

    --------------------------------------------------------------------------------

    KEY:
         O ---> btagged subjet     (boolean 'value' = True)
         X ---> non btagged subjet (boolean 'value' = False)

    Track all conditions where elements of 'btag_update' will be true (4 conditions marked with 'O')
*******************************************************************************************************************        
    """ 

    f_less = abs(1. - BSF) # fraction of subjets to be downgraded
    f_greater = np.where(eff_val > 0., abs(f_less/(1. - 1./eff_val)), 0.) # fraction of subjets to be upgraded  

    condition1 = (ak.flatten(subjet_btag_status) == True) & (BSF == 1.)
    condition2 = (ak.flatten(subjet_btag_status) == True) & ((BSF < 1.0) & (coin < BSF)) 
    condition3 = (ak.flatten(subjet_btag_status) == True) & (BSF > 1.)
    condition4 = (ak.flatten(subjet_btag_status) == False) & ((BSF > 1.) & (coin < f_greater))

    subjet_new_btag_status = np.where((condition1 ^ condition2) ^ (condition3 ^ condition4), True, False)   

    return subjet_new_btag_status

def GetFlavorEfficiency(Subjet, Flavor): # Return "Flavor" efficiency numerator and denominator
    '''
    Subjet --> awkward array object after preselection i.e. SubJetXY
    Flavor --> integer i.e 5, 4, or 0 (b, c, or udsg)
    '''
    # --- Define pT and Eta for Both Candidates' Subjets (for simplicity) --- #
    pT = ak.flatten(Subjet.pt) # pT of subjet in ttbarcand 
    eta = np.abs(ak.flatten(Subjet.eta)) # eta of 1st subjet in ttbarcand 
    flav = np.abs(ak.flatten(Subjet.hadronFlavour)) # either 'normal' or 'anti' quark

    subjet_btagged = (Subjet.btagCSVV2 > bdisc)

    Eff_Num_pT = np.where(subjet_btagged & (flav == Flavor), pT, -1) # if not collecting pT of subjet, then put non exisitent bin, i.e. -1
    Eff_Num_eta = np.where(subjet_btagged & (flav == Flavor), eta, -1) # if not collecting eta of subjet, then put non exisitent bin, i.e. 5

    Eff_Num_pT = ak.flatten(Eff_Num_pT) # extra step needed for numerator to gaurantee proper shape for filling hists
    Eff_Num_eta = ak.flatten(Eff_Num_eta)

    Eff_Denom_pT = np.where(flav == Flavor, pT, -1)
    Eff_Denom_eta = np.where(flav == Flavor, eta, -1)

    EffStuff = {
        'Num_pT' : Eff_Num_pT,
        'Num_eta' : Eff_Num_eta,
        'Denom_pT' : Eff_Denom_pT,
        'Denom_eta' : Eff_Denom_eta,
    }

    return EffStuff
