#!/usr/bin/env python 
# coding: utf-8

import os 
import copy
import scipy.stats as ss
from coffea import hist, processor, nanoevents
from coffea import util
from coffea.btag_tools import BTagScaleFactor
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState
import correctionlib
# from correctionlib.schemav2 import Correction

import awkward as ak
#from coffea.nanoevents.methods import nanoaod
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector

#ak.behavior.update(nanoaod.behavior)
ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)

# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]

# --- Define 'Manual pT bins' to use for mc flavor efficiency plots for higher stats per bin--- # 
manual_subjetpt_bins = [0, 300, 600, 1200] # Used on 6/17/22 for ttbar (3 bins)
manual_subjeteta_bins = [0., 0.6, 1.2, 2.4] # Used on 6/17/22 for ttbar (3 bins)
manual_jetht_bins = [200, 800, 840, 880, 920, 960, 1000, 1200, 1400, 1600, 1800, 2000]

"""Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. """
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., minMSD=105., maxMSD=210.,
                 tau32Cut=0.65, ak8PtMin=400., bdisc=0.8484, deepAK8Cut=0.435, BDirect='',
                 year=None, apv='', vfp='', UseLookUpTables=False, lu=None, extraDaskDirectory='',
                 ModMass=False, RandomDebugMode=False, UseEfficiencies=False, xsSystematicWeight=1., lumSystematicWeight=1.,
                 ApplybtagSF=False, ScaleFactorFile='', ApplyttagSF=False, ApplyTopReweight=False, 
                 ApplyJER=False, ApplyJEC=False, sysType=None):
        
        self.prng = prng
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.BDirect = BDirect
        self.deepAK8Cut = deepAK8Cut
        self.year = year
        self.apv = apv
        self.vfp = vfp
        self.extraDaskDirectory = extraDaskDirectory
        self.UseLookUpTables = UseLookUpTables
        self.ModMass = ModMass
        self.RandomDebugMode = RandomDebugMode
        self.ScaleFactorFile = ScaleFactorFile
        self.ApplybtagSF = ApplybtagSF # Only apply scale factors when MC efficiencies are being imported in second run of processor
        self.ApplyttagSF = ApplyttagSF
        self.ApplyTopReweight = ApplyTopReweight
        self.ApplyJEC = ApplyJEC
        self.ApplyJER = ApplyJER
        self.sysType = sysType # string for btag SF evaluator --> "central", "up", or "down"
        self.UseEfficiencies = UseEfficiencies
        self.xsSystematicWeight = xsSystematicWeight
        self.lumSystematicWeight = lumSystematicWeight
        self.lu = lu # Look Up Tables
        
        # --- anti-tag+probe, anti-tag, pre-tag, 0, 1, >=1, 2 ttags, any t-tag (>=0t) --- #
        self.ttagcats = ["AT&Pt", "at", "pret", "0t", "1t", ">=1t", "2t", ">=0t"] 
        
        # --- 0, 1, or 2 b-tags --- #
        self.btagcats = ["0b", "1b", "2b"]
        
        # --- Central and forward --- #
        self.ycats = ['cen', 'fwd']
        
        # --- Combine categories like "0bcen", "0bfwd", etc: --- #
        self.anacats = [ t+b+y for t,b,y in itertools.product( self.ttagcats, self.btagcats, self.ycats) ]
        #print(self.anacats)
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cats_axis = hist.Cat("anacat", "Analysis Category")
        
        jetmass_axis = hist.Bin("jetmass", r"Jet $m$ [GeV]", 50, 0, 500)
        jetpt_axis = hist.Bin("jetpt", r"Jet $p_{T}$ [GeV]", 50, 0, 5000)
        ttbarmass_axis = hist.Bin("ttbarmass", r"$m_{t\bar{t}}$ [GeV]", 50, 0, 5000)
        jeteta_axis = hist.Bin("jeteta", r"Jet $\eta$", 50, -2.4, 2.4)
        jetphi_axis = hist.Bin("jetphi", r"Jet $\phi$", 50, -np.pi, np.pi)
        jety_axis = hist.Bin("jety", r"Jet $y$", 50, -3, 3)
        jetdy_axis = hist.Bin("jetdy", r"Jet $\Delta y$", 50, 0, 5)
        manual_axis = hist.Bin("jetp", r"Jet Momentum [GeV]", manual_bins)
        tagger_axis = hist.Bin("tagger", r"deepTag", 50, 0, 1)
        tau32_axis = hist.Bin("tau32", r"$\tau_3/\tau_2$", 50, 0, 2)
        
        subjetmass_axis = hist.Bin("subjetmass", r"SubJet $m$ [GeV]", 50, 0, 500)
        subjetpt_axis = hist.Bin("subjetpt", r"SubJet $p_{T}$ [GeV]", 25, 0, 2000)
        subjeteta_axis = hist.Bin("subjeteta", r"SubJet $\eta$", 25, 0, 2.4)
        subjetphi_axis = hist.Bin("subjetphi", r"SubJet $\phi$", 50, -np.pi, np.pi)

        self._accumulator = processor.dict_accumulator({
#    ===================================================================================================================
#    K     K IIIIIII N     N EEEEEEE M     M    A    TTTTTTT IIIIIII   CCCC      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#    K   K      I    NN    N E       MM   MM   A A      T       I     C          H     H    I     S         T     S          
#    K K        I    N N   N E       M M M M  A   A     T       I    C           H     H    I    S          T    S           
#    KKk        I    N  N  N EEEEEEE M  M  M  AAAAA     T       I    C           HHHHHHH    I     SSSSS     T     SSSSS      
#    K  K       I    N   N N E       M     M A     A    T       I    C           H     H    I          S    T          S     
#    K   K      I    N    NN E       M     M A     A    T       I     C          H     H    I         S     T         S      
#    K   K   IIIIIII N     N EEEEEEE M     M A     A    T    IIIIIII   CCCC      H     H IIIIIII SSSSS      T    SSSSS 
#    ===================================================================================================================
            
            'ttbarmass': hist.Hist("Counts", dataset_axis, cats_axis, ttbarmass_axis),
            
            'jetmass':         hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass':          hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass_precat':   hist.Hist("Counts", dataset_axis, jetpt_axis, jetmass_axis), # What was this for again?
            
            'jetpt':     hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis),
            'jeteta':    hist.Hist("Counts", dataset_axis, cats_axis, jeteta_axis),
            'jetphi':    hist.Hist("Counts", dataset_axis, cats_axis, jetphi_axis),
            
            'probept':   hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis),
            'probep':    hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            
            'jety':      hist.Hist("Counts", dataset_axis, cats_axis, jety_axis),
            'jetdy':     hist.Hist("Counts", dataset_axis, cats_axis, jetdy_axis),
            
            'deepTag_TvsQCD':   hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis, tagger_axis),
            'deepTagMD_TvsQCD': hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis, tagger_axis),
            
            'tau32':          hist.Hist("Counts", dataset_axis, cats_axis, tau32_axis),
            'tau32_2D':       hist.Hist("Counts", dataset_axis, cats_axis, jetpt_axis, tau32_axis),
            'tau32_precat': hist.Hist("Counts", dataset_axis, jetpt_axis, tau32_axis),
            
            'subjetmass':   hist.Hist("Counts", dataset_axis, cats_axis, subjetmass_axis), # not yet used
            'subjetpt':     hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            'subjeteta':    hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            'subjetphi':    hist.Hist("Counts", dataset_axis, cats_axis, subjetphi_axis), # not yet used
            
            'numerator':   hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            'denominator': hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            
            #********************************************************************************************************************#
            
            'cutflow': processor.defaultdict_accumulator(int),
            
        })
        
#   =======================================================================
#   FFFFFFF U     U N     N   CCCC  TTTTTTT IIIIIII   OOO   N     N   SSSSS     
#   F       U     U NN    N  C         T       I     O   O  NN    N  S          
#   F       U     U N N   N C          T       I    O     O N N   N S           
#   FFFFFFF U     U N  N  N C          T       I    O     O N  N  N  SSSSS      
#   F       U     U N   N N C          T       I    O     O N   N N       S     
#   F        U   U  N    NN  C         T       I     O   O  N    NN      S      
#   F         UUU   N     N   CCCC     T    IIIIIII   OOO   N     N SSSSS
#   =======================================================================

    #https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/11146645#11146645
    def CartesianProduct(self, *arrays): 
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
    
    def BtagUpdater(self, subjet, Eff_filename_list, ScaleFactorFilename, FittingPoint, OperatingPoint):  
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
        eff_BinKeys_comb = self.CartesianProduct(pt_BinKeys, eta_BinKeys) #List of Combined pt and eta keys (should be 40 of them)
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
        subjet_btag_status = np.asarray((subjet.btagCSVV2 > self.bdisc)) # do subjets pass the btagger requirement
        
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
    
    def GetFlavorEfficiency(self, Subjet, Flavor): # Return "Flavor" efficiency numerator and denominator
        '''
        Subjet --> awkward array object after preselection i.e. SubJetXY
        Flavor --> integer i.e 5, 4, or 0 (b, c, or udsg)
        '''
        # --- Define pT and Eta for Both Candidates' Subjets (for simplicity) --- #
        pT = ak.flatten(Subjet.pt) # pT of subjet in ttbarcand 
        eta = np.abs(ak.flatten(Subjet.eta)) # eta of 1st subjet in ttbarcand 
        flav = np.abs(ak.flatten(Subjet.hadronFlavour)) # either 'normal' or 'anti' quark
        
        subjet_btagged = (Subjet.btagCSVV2 > self.bdisc)
        
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
            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        
        # ---- Define dataset ---- #
        dataset = events.metadata['dataset']

#    ===========================================================================================    
#    N     N    A    N     N   OOO      A      OOO   DDDD          OOO   BBBBBB  JJJJJJJ   SSSSS     
#    NN    N   A A   NN    N  O   O    A A    O   O  D   D        O   O  B     B    J     S          
#    N N   N  A   A  N N   N O     O  A   A  O     O D    D      O     O B     B    J    S           
#    N  N  N  AAAAA  N  N  N O     O  AAAAA  O     O D     D     O     O BBBBBB     J     SSSSS      
#    N   N N A     A N   N N O     O A     A O     O D    D      O     O B     B J  J          S     
#    N    NN A     A N    NN  O   O  A     A  O   O  D   D        O   O  B     B J  J         S      
#    N     N A     A N     N   OOO   A     A   OOO   DDDD          OOO   BBBBBB   JJ     SSSSS 
#    =========================================================================================== 

        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
        
        # ---- Define AK8 Jets as FatJets ---- #
        #FatJets = events.FatJet # Everything should already be defined in here.  example) df['FatJet_pt] -> events.FatJet.pt
        FatJets = ak.zip({
            "run": events.run,
            "nFatJet": events.nFatJet,
            "pt": events.FatJet_pt,
            "eta": events.FatJet_eta,
            "phi": events.FatJet_phi,
            "mass": events.FatJet_mass,
            "area": events.FatJet_area,
            "msoftdrop": events.FatJet_msoftdrop,
            "jetId": events.FatJet_jetId,
            "tau1": events.FatJet_tau1,
            "tau2": events.FatJet_tau2,
            "tau3": events.FatJet_tau3,
            "tau4": events.FatJet_tau4,
            "n3b1": events.FatJet_n3b1,
            "btagDeepB": events.FatJet_btagDeepB,
            "btagCSVV2": events.FatJet_btagCSVV2, # Use as a prior probability of containing a bjet?
            "deepTag_TvsQCD": events.FatJet_deepTag_TvsQCD,
            "deepTagMD_TvsQCD": events.FatJet_deepTagMD_TvsQCD,
            "subJetIdx1": events.FatJet_subJetIdx1,
            "subJetIdx2": events.FatJet_subJetIdx2,
            "p4": ak.zip({
                "pt": events.FatJet_pt,
                "eta": events.FatJet_eta,
                "phi": events.FatJet_phi,
                "mass": events.FatJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define AK4 jets as Jets ---- #
        Jets = ak.zip({
            "run": events.run,
            "pt": events.Jet_pt,
            "eta": events.Jet_eta,
            "phi": events.Jet_phi,
            "mass": events.Jet_mass,
            "area": events.Jet_area,
            "p4": ak.zip({
                "pt": events.Jet_pt,
                "eta": events.Jet_eta,
                "phi": events.Jet_phi,
                "mass": events.Jet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define SubJets ---- #
        SubJets = ak.zip({
            "run": events.run,
            "pt": events.SubJet_pt,
            "eta": events.SubJet_eta,
            "phi": events.SubJet_phi,
            "mass": events.SubJet_mass,
            "btagDeepB": events.SubJet_btagDeepB,
            "btagCSVV2": events.SubJet_btagCSVV2,
            "p4": ak.zip({
                "pt": events.SubJet_pt,
                "eta": events.SubJet_eta,
                "phi": events.SubJet_phi,
                "mass": events.SubJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })
        
        # ---- Define Generator Particles and other needed event properties for MC ---- #
        if isData == False: # If MC is used...
            GenParts = ak.zip({
                "run": events.run,
                "pdgId": events.GenPart_pdgId,
                "pt": events.GenPart_pt,
                "eta": events.GenPart_eta,
                "phi": events.GenPart_phi,
                "mass": events.GenPart_mass,
                "p4": ak.zip({
                    "pt": events.GenPart_pt,
                    "eta": events.GenPart_eta,
                    "phi": events.GenPart_phi,
                    "mass": events.GenPart_mass,
                    }, with_name="Vector3D"),
                })
            
            Jets['hadronFlavour'] = events.Jet_hadronFlavour
            SubJets['hadronFlavour'] = events.SubJet_hadronFlavour
            
#    ================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR    SSSSS     
#       T    R     R    I    G       G       E       R     R  S          
#       T    R     R    I    G       G       E       R     R S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR   SSSSS      
#       T    R   R      I    G     G G     G E       R   R         S     
#       T    R    R     I    G     G G     G E       R    R       S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R SSSSS  
#    ================================================================
        
#         Dataset_info = events.fields # All nanoaod events
#         listOfTriggers = np.array([name for name in Dataset_info if 'HLT' in name]) # Find event name info that have HLT to find all relevant triggers
        
#         isHLT_PF = np.array(['HLT_PF' in i for i in listOfTriggers])
#         isHLT_AK8 = np.array(['HLT_AK8' in i for i in listOfTriggers])
        
#         HLT_PF_triggers = listOfTriggers[isHLT_PF]
#         HLT_AK8_triggers = listOfTriggers[isHLT_AK8]
        
        
        # trigDenom = events.HLT_Mu50 | events.HLT_IsoMu24 # WHY!!!!!????
        # print(events.HLT_Mu50)
        # print(events.HLT_IsoMu24)
        # print(trigDenom)
        # print('-----------------------------------------------')
        
        # if self.year == 2016:
        #     triggers2016_1 = events.HLT_PFHT900
        #     triggers2016_2 = events.HLT_AK8PFHT700_TrimR0p1PT0p03Mass50
        #     triggers2016_3 = events.HLT_AK8PFJet450
        #     triggers2016_4 = events.HLT_AK8PFJet360_TrimMass30

#    ===================================================================================
#    PPPPPP  RRRRRR  EEEEEEE L       IIIIIII M     M       CCCC  U     U TTTTTTT   SSSSS     
#    P     P R     R E       L          I    MM   MM      C      U     U    T     S          
#    P     P R     R E       L          I    M M M M     C       U     U    T    S           
#    PPPPPP  RRRRRR  EEEEEEE L          I    M  M  M     C       U     U    T     SSSSS      
#    P       R   R   E       L          I    M     M     C       U     U    T          S     
#    P       R    R  E       L          I    M     M      C       U   U     T         S      
#    P       R     R EEEEEEE LLLLLLL IIIIIII M     M       CCCC    UUU      T    SSSSS
#    ===================================================================================
        
        # ---- Get event weights from dataset ---- #
        if isData: # If data is used...
            # print('if isData command works')
            evtweights = np.ones(ak.to_awkward0(FatJets).size) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = events.Generator_weight
        # ---- Show all events ---- #
        output['cutflow']['all events'] += ak.to_awkward0(FatJets).size
        
        # ---- Setup Trigger Analysis Conditions in higher scope ---- #
        # condition1 = None
        # condition2 = None
        # condition3 = None
        # condition4 = None
        # firstCondition = None
        
        # ---- Apply Trigger(s) ---- # 
        # if self.year == 2016: # Include other years later...
        #     applyTrigs = (triggers2016_1 ^ triggers2016_2) ^ (triggers2016_3 ^ triggers2016_4)
        #     firstCondition = (triggers2016_1) 

        # ---- Defining Trigger Analysis Conditions ---- #
        # condition1 = firstCondition & trigDenom # WHY?! Figure out later...
        # condition2 = (firstCondition | triggers2016_2) & trigDenom
        # condition3 = ((firstCondition | triggers2016_2) | triggers2016_3) & trigDenom
        # condition4 = ((firstCondition | triggers2016_2) | (triggers2016_3 | triggers2016_4)) & trigDenom
            
        # Do NOT apply triggers yet...
        # if not self.triggerAnalysisObjects: # apply all necessary triggers as a first step if not performing trigger analysis
        #     FatJets = FatJets[applyTrigs]
        #     Jets = Jets[applyTrigs]
        #     SubJets = SubJets[applyTrigs] 
        #     evtweights = evtweights[applyTrigs]

        # ---- Apply HT Cut ---- #
        # ---- This gives the analysis 99.8% efficiency (see 2016 AN) ---- #
        hT = ak.to_awkward0(Jets.pt).sum()
        passhT = (hT > self.htCut)
        FatJets = FatJets[passhT]
        Jets = Jets[passhT] # this used to not be here
        SubJets = SubJets[passhT]
        evtweights = evtweights[passhT]
        output['cutflow']['HT Cut'] += ak.to_awkward0(passhT).sum()
        if isData == False: # If MC is used...
            # print('if not isData command works')
            GenParts = GenParts[passhT]
        
            
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['Loose Jet ID'] += ak.to_awkward0(jet_id).any().sum()
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['pT,y Cut'] += ak.to_awkward0(jetkincut_index).any().sum()
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) == 2)
        FatJets = FatJets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin] # this used to not be here
        evtweights = evtweights[twoFatJetsKin]
        if not isData: # If MC is used...
            GenParts = GenParts[twoFatJetsKin]
        output['cutflow']['two FatJets'] += ak.to_awkward0(twoFatJetsKin).sum()
        
        # ---- Randomly Assign AK8 Jets as TTbar Candidates 0 and 1 --- #
        Counts = np.ones(len(FatJets), dtype='i') # Number 1 for each FatJet
        
        if self.RandomDebugMode == True: # 'Sudo' randomizer to test for consistent results
            highPhi = FatJets.phi[:,0] > FatJets.phi[:,1]
            highRandIndex = np.where(highPhi, 0, 1) # 1D array of 0's and 1's
            index = ak.unflatten( highRandIndex, Counts ) # Subtly confusing logic of what this does is shown below
            """
                For example:
                
                    FatJets = [[FatJet0, FatJet1], [AnotherFatJet0, AnotherFatJet1], ..., [LastFatJet0, LastFatJet1]]
                    Counts = [1, 1, 1, 1, 1, 1]
                    highRandIndex = [1, 1, 0, 1, 0, 0]
                    
                unflattening highRandIndex with Counts will group "Counts" number of highRandIndex elements together 
                in a new higher order array:
                
                    index = [[1], [1], [0], [1], [0], [0]]
                    1 - index = [[0], [0], [1], [0], [1], [1]]
                    
               where the index is used to slice (select) either the first FatJet (FatJets[:,0]) or the second (FatJets[:,1]):
        
                    FatJets[index] = [FatJet1, AnotherFatJet1, ..., LastFatJet0]
                    FatJets[1-index] = [FatJet0, AnotherFatJet0, ..., LastFatJet1]
            """
        else: # Truly randomize
            index = ak.unflatten( self.prng.randint(2, size=len(FatJets)), Counts )
        
        jet0 = FatJets[index] #J0
        jet1 = FatJets[1 - index] #J1
        
        ttbarcands = ak.cartesian([jet0, jet1]) # Re-group the randomized pairs in a similar fashion to how they were
        
        """ NOTE that ak.cartesian gives a shape with one more layer than FatJets """
        # ---- Make sure we have at least 1 TTbar candidate pair and re-broadcast releveant arrays  ---- #
        oneTTbar = (ak.num(ttbarcands, axis=-1) >= 1)
        output['cutflow']['>= oneTTbar'] += ak.to_awkward0(oneTTbar).sum()
        ttbarcands = ttbarcands[oneTTbar]
        FatJets = FatJets[oneTTbar]
        Jets = Jets[oneTTbar] # this used to not be here
        SubJets = SubJets[oneTTbar]
        evtweights = evtweights[oneTTbar]
        if not isData: # If MC is used...
            GenParts = GenParts[oneTTbar]
            
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        output['cutflow']['dPhi Cut'] += ak.to_awkward0(dPhiCut).sum()
        ttbarcands = ttbarcands[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        Jets = Jets[dPhiCut] # this used to not be here
        SubJets = SubJets[dPhiCut] 
        evtweights = evtweights[dPhiCut]
        if not isData: # If MC is used...
            GenParts = GenParts[dPhiCut]
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
        output['cutflow']['Good Subjets'] += ak.to_awkward0(GoodSubjets).sum()
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        SubJets = SubJets[GoodSubjets]
        Jets = Jets[GoodSubjets] # this used to not be here
        if not isData: # If MC is used...
            GenParts = GenParts[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
        
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # ttbarcandidate 0's first subjet 
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # ttbarcandidate 0's second subjet
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # ttbarcandidate 1's first subjet 
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # ttbarcandidate 1's second subjet
        
        # ---- Define Rapidity Regions ---- #
        """ NOTE that ttbarcands.i0.p4.energy no longer works after ttbarcands is defined as an old awkward array """
        s0_energy = ttbarcands.slot0.p4.energy
        s1_energy = ttbarcands.slot1.p4.energy
        s0_pz = ttbarcands.slot0.p4.pz
        s1_pz = ttbarcands.slot1.p4.pz
        ttbarcands_s0_rapidity = 0.5*np.log( (s0_energy+s0_pz)/(s0_energy-s0_pz) ) # rapidity as function of eta
        ttbarcands_s1_rapidity = 0.5*np.log( (s1_energy+s1_pz)/(s1_energy-s1_pz) ) # rapidity as function of eta
        cen = np.abs(ttbarcands_s0_rapidity - ttbarcands_s1_rapidity) < 1.0
        fwd = (~cen)
        

#    ============================================================
#    TTTTTTT     TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#       T           T      A A   G       G       E       R     R 
#       T           T     A   A  G       G       E       R     R 
#       T           T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#       T           T    A     A G     G G     G E       R   R   
#       T           T    A     A G     G G     G E       R    R   
#       T           T    A     A  GGGGG   GGGGG  EEEEEEE R     R
#    ============================================================

        # ----------- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ----------- #
        # ---- NOTE: Must Change This to DeepAK8 Top Tag Discriminator Cut ----#
        # ---- Maybe we should ignore tau32 cut(s) when performing trigger analysis ---- #
        tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
        tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )
        
        taucut_s0 = tau32_s0 < self.tau32Cut
        taucut_s1 = tau32_s1 < self.tau32Cut
        
        mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
        mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

        ttag_s0 = (taucut_s0) & (mcut_s0)
        ttag_s1 = (taucut_s1) & (mcut_s1)
        antitag = (~taucut_s0) & (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)

        # ----------- DeepAK8 Tagger (Discriminator Cut) ----------- #
        # ttag_s0 = ttbarcands.slot0.deepTag_TvsQCD > self.deepAK8Cut
        # ttag_s1 = ttbarcands.slot1.deepTag_TvsQCD > self.deepAK8Cut
        # antitag = ttbarcands.slot0.deepTag_TvsQCD < self.deepAK8Cut # The Probe jet will always be ttbarcands.slot1 (at)
        
        # ---- Define "Top Tag" Regions ---- #
        antitag_probe = np.logical_and(antitag, ttag_s1) # Found an antitag and ttagged probe pair for mistag rate (Probet)
        pretag =  ttag_s0 # Only jet0 (pret)
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t)
        ttag1 =   ttag_s0 ^ ttag_s1 # Exclusively one top tagged (1t)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttag2 =   ttag_s0 & ttag_s1 # Both jets top tagged (2t)
        Alltags = ttag0 | ttagI #Either no tag or at least one tag (0t+1t+2t)
        
        
#    ============================================================        
#    BBBBBB      TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#    B     B        T      A A   G       G       E       R     R 
#    B     B        T     A   A  G       G       E       R     R 
#    BBBBBB         T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#    B     B        T    A     A G     G G     G E       R   R   
#    B     B        T    A     A G     G G     G E       R    R   
#    BBBBBB         T    A     A  GGGGG   GGGGG  EEEEEEE R     R
#    ============================================================
        
        # ---- Pick FatJet that passes btag discriminator cut based on its subjet with the highest btag value ---- #
        # -------------- NOTE: B-discriminator cut must be changed to match BTV POG Recommendations -------------- #
        btag_s0 = ( np.maximum(SubJet01.btagCSVV2 , SubJet02.btagCSVV2) > self.bdisc )
        btag_s1 = ( np.maximum(SubJet11.btagCSVV2 , SubJet12.btagCSVV2) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_s0) & (~btag_s1) #(0b)
        btag1 = btag_s0 ^ btag_s1 #(1b)
        btag2 = btag_s0 & btag_s1 #(2b)
        
#    ===========================================================
#       A    PPPPPP  PPPPPP  L       Y     Y       SSSSS FFFFFFF     
#      A A   P     P P     P L        Y   Y       S      F           
#     A   A  P     P P     P L         Y Y       S       F           
#     AAAAA  PPPPPP  PPPPPP  L          Y         SSSSS  FFFFFFF     
#    A     A P       P       L          Y              S F           
#    A     A P       P       L          Y             S  F           
#    A     A P       P       LLLLLLL    Y        SSSSS   F
#    ===========================================================
        
        if not isData:
            
            Btag_wgts = {} # To be filled with "btag_wgts" corrections below (Needs to be defined for higher scope)
            if self.ApplybtagSF == True: # Apply b Tag Scale Factors and redefine btag_s0 and btag_s1
                if self.UseEfficiencies == False: # Define weights solely from BSF to weight each btag category region
                    
                    # **************************************************************************************** #
                    # --------------------------- Method 1c) Apply Event Weights ----------------------------- #
                    # -------------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods -------------- #
                    # **************************************************************************************** #
                
                    # ---- Temporarily define the 'outline' of the collection of weights to calculate ---- # 
                    btag_wgts = {'0b':np.array([None]),
                                 '1b':np.array([None, None]),
                                 '2b':np.array([None, None, None])}

                    """
                    ******************************************************************************************************
                    btag_wgts['mb'][n] --> w(n|m) --> "Probability" of n number of b-tags given m number of b-tagged jets
                    -----------------------------------------------------------------------------------------
                    w(0|0) = 1

                    w(0|1), w(1|1) = 1 - BSF, 
                                   = BSF

                    w(0|2), w(1|2), w(2|2) = (1 - BSF_s0)(1 - BSF_s1), 
                                           = (1 - BSF_s0)BSF_s1 + BSF_s0(1 - BSF_s1),
                                           = (BSF_s0)(BSF_s1)

                    w(1|0), w(2|0), w(2|1) = Undef.
                    ******************************************************************************************************
                    """

                    # ---- Use the leading subjet again to get the scale factors ---- #
                    LeadingSubjet_s0 = np.where(SubJet01.btagCSVV2>SubJet02.btagCSVV2, SubJet01, SubJet02)
                    LeadingSubjet_s1 = np.where(SubJet11.btagCSVV2>SubJet12.btagCSVV2, SubJet11, SubJet12)

                    # ---- Define the BSF for each of the two fatjets ---- #
                    SF_filename = self.ScaleFactorFile    
                    Fitting = "M"
                    if self.bdisc < 0.5:
                        Fitting = "L"
                     
                    btag_sf = correctionlib.CorrectionSet.from_file(SF_filename)
                    
                    s0_hadronFlavour = ak.to_numpy(ak.flatten(LeadingSubjet_s0.hadronFlavour))
                    s1_hadronFlavour = ak.to_numpy(ak.flatten(LeadingSubjet_s1.hadronFlavour))
                    
                    s0_eta = ak.to_numpy(ak.flatten(LeadingSubjet_s0.eta))
                    s1_eta = ak.to_numpy(ak.flatten(LeadingSubjet_s1.eta))
                    
                    s0_pt = ak.to_numpy(ak.flatten(LeadingSubjet_s0.pt))
                    s1_pt = ak.to_numpy(ak.flatten(LeadingSubjet_s1.pt))
                    
                    s0_allHeavy = np.where(s0_hadronFlavour == 0, 4, s0_hadronFlavour)
                    s0_allLight = np.zeros_like(s0_allHeavy) 
                    
                    s1_allHeavy = np.where(s1_hadronFlavour == 0, 4, s1_hadronFlavour)
                    s1_allLight = np.zeros_like(s1_allHeavy) 
                    
                    # ---- Ensure eta and pt fall within the allowed binning for corrections ---- #
                    Min_etaval = 0.
                    Max_etaval = 2.5
                    Min_ptval = 30.
                    Max_ptval = 450.
                    
                    s0_eta = np.where(abs(s0_eta)>=Max_etaval, Max_etaval-0.01, s0_eta)
                    s1_eta = np.where(abs(s1_eta)>=Max_etaval, Max_etaval-0.01, s1_eta)
                    
                    s0_pt = np.where(abs(s0_pt)<=Min_ptval, Min_ptval+1.00, s0_pt)
                    s1_pt = np.where(abs(s1_pt)<=Min_ptval, Min_ptval+1.00, s1_pt)
                    s0_pt = np.where(abs(s0_pt)>=Max_ptval, Max_ptval-1.00, s0_pt)
                    s1_pt = np.where(abs(s1_pt)>=Max_ptval, Max_ptval-1.00, s1_pt)
                    
                    try:
                        BSF_s0_allHeavy = btag_sf['deepCSV_subjet'].evaluate(self.sysType, 'lt', Fitting, s0_allHeavy, abs(s0_eta), s0_pt)
                    except RuntimeError as re:
                        print('flavor (with light mask): \n', s0_allHeavy)
                        print('eta: \n', s0_eta)
                        print('pt: \n', s0_pt)
                        print('These subjets\' all heavy SFs evaluation failed')
                        print(re)
                    try:
                        BSF_s1_allHeavy = btag_sf['deepCSV_subjet'].evaluate(self.sysType, 'lt', Fitting, s1_allHeavy, abs(s1_eta), s1_pt)
                    except RuntimeError as RE:
                        print('flavor (with light mask): \n', s1_allHeavy)
                        print('eta: \n', s1_eta)
                        print('pt: \n', s1_pt)
                        print('These subjets\' all heavy SFs evaluation failed')
                        print(RE)
                    
                    BSF_s0_allLight = btag_sf['deepCSV_subjet'].evaluate(self.sysType, 'incl', Fitting, s0_allLight, abs(s0_eta), s0_pt)
                    BSF_s1_allLight = btag_sf['deepCSV_subjet'].evaluate(self.sysType, 'incl', Fitting, s1_allLight, abs(s1_eta), s1_pt)
                    
                    BSF_s0 = np.where(s0_hadronFlavour == 0, BSF_s0_allLight, BSF_s0_allHeavy)
                    BSF_s1 = np.where(s1_hadronFlavour == 0, BSF_s1_allLight, BSF_s1_allHeavy)
                    
                    # ---- w(0|0) ---- #
                    btag_wgts['0b'][0] = np.where(btag0, np.ones_like(BSF_s0), 0.)

                    # ---- w(0|1) and w(1|1) ---- # 
                    btag_wgts['1b'][0] = np.where(btag1, np.where(btag_s0, 1.-BSF_s0, 1.-BSF_s1), 0.)
                    btag_wgts['1b'][1] = np.where(btag1, np.where(btag_s0, BSF_s0, BSF_s1), 0.)

                    # ---- w(0|2), w(1|2), w(2|2) ---- # 
                    btag_wgts['2b'][0] = np.where(btag2, (1 - BSF_s0)*(1 - BSF_s1), 0.) 
                    btag_wgts['2b'][1] = np.where(btag2, (1 - BSF_s0)*BSF_s1 + BSF_s0*(1 - BSF_s1), 0.) 
                    btag_wgts['2b'][2] = np.where(btag2, BSF_s0*BSF_s1, 0.) 

                    # ---- 'Matrix Multiplied' weights to apply to each b-tag region ---- #
                    Wgts_to_0btag_region = ak.flatten(btag_wgts['0b'][0] + btag_wgts['1b'][0] + btag_wgts['2b'][0])
                    Wgts_to_1btag_region = ak.flatten(btag_wgts['1b'][1] + btag_wgts['2b'][1])
                    Wgts_to_2btag_region = ak.flatten(btag_wgts['2b'][2])

                    # ---- 'Matrix Multiplied' non-zero weights to apply to each b-tag region ---- #
                    Wgts_to_0btag_region_nonzero = np.where(Wgts_to_0btag_region==0., 1., Wgts_to_0btag_region)
                    Wgts_to_1btag_region_nonzero = np.where(Wgts_to_1btag_region==0., 1., Wgts_to_1btag_region)
                    Wgts_to_2btag_region_nonzero = np.where(Wgts_to_2btag_region==0., 1., Wgts_to_2btag_region)

                    # ---- Avoid Potential Scope Issues (Unpredictable when/if scope error occurs, so best to avoid it entirely) ---- #
                    Btag_wgts['0b'] = Wgts_to_0btag_region_nonzero
                    Btag_wgts['1b'] = Wgts_to_1btag_region_nonzero
                    Btag_wgts['2b'] = Wgts_to_2btag_region_nonzero


                else: # Upgrade or Downgrade btag status based on btag efficiency of all four subjets
                    
                    # **************************************************************************************** #
                    # --------------------------- Method 2a) Update B-tag Status ----------------------------- #
                    # -------------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods -------------- #
                    # **************************************************************************************** #
                    
                                # ---- Import MC 'flavor' efficiencies ---- #

                    # -- Scale Factor File -- #
                    SF_filename = self.ScaleFactorFile    
                    Fitting = "M"
                    if self.bdisc < 0.5:
                        Fitting = "L"
                    
                    # -- Get Efficiency .csv Files -- #
                    FlavorTagsDict = {
                        5 : 'btag',
                        4 : 'ctag',
                        0 : 'udsgtag'
                    }

                    SubjetNumDict = {
                        'SubJet01' : [SubJet01, 's01'],
                        'SubJet02' : [SubJet02, 's02'],
                        'SubJet11' : [SubJet11, 's11'],
                        'SubJet12' : [SubJet12, 's12']
                    }
                    
                    EffFileDict = {
                        'Eff_File_s01' : [], # List of eff files corresponding to 1st subjet's flavours
                        'Eff_File_s02' : [], # List of eff files corresponding to 2nd subjet's flavours
                        'Eff_File_s11' : [], # List of eff files corresponding to 3rd subjet's flavours
                        'Eff_File_s12' : []  # List of eff files corresponding to 4th subjet's flavours
                    }
                    
                    for subjet,subjet_info in SubjetNumDict.items():
                        flav_tag_list = [FlavorTagsDict[num] for num in np.abs(ak.flatten(subjet_info[0].hadronFlavour))] # List of tags i.e.) ['btag', 'udsgtag', 'ctag',...]
                        for flav_tag in flav_tag_list:
                            EffFileDict['Eff_File_'+subjet_info[1]].append('TTbarAllHadUproot/FlavorTagEfficiencies/' + BDirect + flav_tag 
                                                                           + 'EfficiencyTables/' + dataset + '_' + subjet_info[1] 
                                                                           + '_' + flav_tag + 'eff.csv')
                            
                    # -- Does Subjet pass the discriminator cut and is it updated -- #
                    SubJet01_isBtagged = self.BtagUpdater(SubJet01, EffFileDict['Eff_File_s01'], SF_filename, Fitting, self.sysType)
                    SubJet02_isBtagged = self.BtagUpdater(SubJet02, EffFileDict['Eff_File_s02'], SF_filename, Fitting, self.sysType)
                    SubJet11_isBtagged = self.BtagUpdater(SubJet11, EffFileDict['Eff_File_s11'], SF_filename, Fitting, self.sysType)
                    SubJet12_isBtagged = self.BtagUpdater(SubJet12, EffFileDict['Eff_File_s12'], SF_filename, Fitting, self.sysType)

                    # If either subjet 1 or 2 in FatJet 0 and 1 is btagged after update, then that FatJet is considered btagged #
                    btag_s0 = (SubJet01_isBtagged) | (SubJet02_isBtagged)  
                    btag_s1 = (SubJet11_isBtagged) | (SubJet12_isBtagged)

                    # --- Re-Define b-Tag Regions with "Updated" Tags ---- #
                    btag0 = (~btag_s0) & (~btag_s1) #(0b)
                    btag1 = btag_s0 ^ btag_s1 #(1b)
                    btag2 = btag_s0 & btag_s1 #(2b)
                    
#    ================================================================
#       A    N     N    A    L       Y     Y   SSSSS IIIIIII   SSSSS 
#      A A   NN    N   A A   L        Y   Y   S         I     S      
#     A   A  N N   N  A   A  L         Y Y   S          I    S       
#     AAAAA  N  N  N  AAAAA  L          Y     SSSSS     I     SSSSS  
#    A     A N   N N A     A L          Y          S    I          S 
#    A     A N    NN A     A L          Y         S     I         S  
#    A     A N     N A     A LLLLLLL    Y    SSSSS   IIIIIII SSSSS   
#    ================================================================

        # ---- Get Analysis Categories ---- # 
        # ---- They are (central, forward) cross (0b,1b,2b) cross (Probet,at,pret,0t,1t,>=1t,2t,>=0t) ---- #
        regs = [cen,fwd]
        btags = [btag0,btag1,btag2]
        ttags = [antitag_probe,antitag,pretag,ttag0,ttag1,ttagI,ttag2,Alltags]
        cats = [ ak.to_awkward0(ak.flatten(t&b&y)) for t,b,y in itertools.product( ttags, btags, regs) ]
        labels_and_categories = dict(zip( self.anacats, cats ))
        #print(labels_and_categories)
        
        # ---- Variables for Kinematic Histograms ---- #
        # ---- "slot0" is the control jet, "slot1" is the probe jet ---- #
        jetpt = ak.flatten(ttbarcands.slot1.pt)
        jeteta = ak.flatten(ttbarcands.slot1.eta)
        jetphi = ak.flatten(ttbarcands.slot1.phi)
        jetmass = ak.flatten(ttbarcands.slot1.mass)
        
        SDmass = ak.flatten(ttbarcands.slot1.msoftdrop)
        Tau32 = ak.flatten((ttbarcands.slot1.tau3/ttbarcands.slot1.tau2))

        """ Add 4-vectors and get its total mass """
        ttbarp4sum = ttbarcands.slot0.p4.add(ttbarcands.slot1.p4)
        ttbarmass = ak.flatten(ttbarp4sum.mass)
        
        """ Use previously defined definitions for rapidity (until/unless better method is found) """
        jety = ak.flatten(ttbarcands_s0_rapidity)
        jetdy = np.abs(ak.flatten(ttbarcands_s0_rapidity) - ak.flatten(ttbarcands_s1_rapidity))
        
        # ---- Weights ---- #
        weights = evtweights*self.xsSystematicWeight*self.lumSystematicWeight

        # ---- Define the SumW2 for MC Datasets (Probably unnecessary now) ---- #
        output['cutflow']['sumw'] += np.sum(weights)
        output['cutflow']['sumw2'] += np.sum(weights**2)
        
        # ---- Define Momentum p of probe jet as the Mistag Rate variable; M(p) ---- #
        # ---- Transverse Momentum pT can also be used instead; M(pT) ---- #
        pT = ak.flatten(ttbarcands.slot1.pt)
        pz = ak.flatten(ttbarcands.slot1.p4.pz)
        p = np.absolute(np.sqrt(pT**2 + pz**2))
        
        # --------------------------------------------------------------------------------- #
        # ----          Avoid momentum values greater than what's defined in           ---- #
        # ---- manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000] ---- #
        # ---- example: momentum                 BinWidth  BinNumber                   ---- #
        # ----          13339.667                NaN       NaN                         ---- #
        # --------------------------------------------------------------------------------- #
        p = np.where(p >= 10000., 9999., p)
        
        # -------------------- Define pT of control jet ------------------ #
        # ---- Use pT and pT0 to calculate ttbar pT weight used later ---- #
        pT0 = ak.flatten(ttbarcands.slot0.pt)
        topcand0_wgt = np.exp(0.0615 - 0.0005*ak.to_numpy(pT0))
        topcand1_wgt = np.exp(0.0615 - 0.0005*ak.to_numpy(pT))
        ttbar_wgt = np.sqrt(topcand0_wgt*topcand1_wgt) # used for re-weighting tttbar MC
        
        # ---- Define the Numerator and Denominator for Mistag Rate ---- #
        numerator = np.where(antitag_probe, p, -1) # If no antitag and tagged probe, move event to useless bin
        denominator = np.where(antitag, p, -1) # If no antitag, move event to useless bin
        
        numerator = ak.flatten(numerator)
        denominator = ak.flatten(denominator)
        
        df = pd.DataFrame({"momentum":p}) # DataFrame used for finding values in LookUp Tables
        file_df = None # Initial Declaration
        
        # print(self.lu[])
        
        for ilabel,icat in labels_and_categories.items():
            ###------------------------------------------------------------------------------------------###
            ### ------------------------------------ Mistag Scaling ------------------------------------ ###
            ###------------------------------------------------------------------------------------------###
            if self.UseLookUpTables == True:
                # ---- Weight dataset by mistag from data (corresponding to its year) ---- #
                # ---- Pick out proper JetHT year mistag for TTbar sim. ---- #
                
                if self.year > 0: # this UL string should only appear in MC dataset name when year is either 2016, 2017 or 2018
                    file_df = self.lu['JetHT' + str(self.year) + '_Data']['at' + str(ilabel[-5:])] # Only the corresponding JetHT year mistag rate
                elif self.year == 0: # all years; not just 2016, 17 or 18 alone
                    file_df = self.lu['JetHT_Data']['at' + str(ilabel[-5:])] # All JetHT years mistag rate
                else:
                    print('Something is wrong...\n\nNecessary JetHT LUT(s) not found')
                    quit()
               
                bin_widths = file_df['p'].values # collect bins as written in .csv file
                mtr = file_df['M(p)'].values # collect mistag rate as function of p as written in file
                wgts = mtr # Define weights based on mistag rates
                
                BinKeys = np.arange(bin_widths.size) # Use as label for BinNumber column in the new dataframe
                
                Bins = np.array(manual_bins)
                
                df['BinWidth'] = pd.cut(np.asarray(p), bins=Bins) # new dataframe column
                df['BinNumber'] = pd.cut(np.asarray(p), bins=Bins, labels=BinKeys)
                
                BinNumber = df['BinNumber'].values # Collect the Bin Numbers into a numpy array
                try:
                    BinNumber = BinNumber.astype('int64') # Insures the bin numbers are integers
                except ValueError as VE:
                    with pd.option_context('display.max_rows', None,
                                       'display.max_columns', None,
                                       'display.precision', 3,
                                       ):
                        print(df)
                    print(VE)
            
                WeightMatching = wgts[BinNumber] # Match 'wgts' with corresponding p bin using the bin number
                Weights = weights*WeightMatching # Include 'wgts' with the previously defined 'weights'
            else:
                Weights = weights # No mistag rate changes to weights
                
            ###---------------------------------------------------------------------------------------------###
            ### ----------------------------------- Mod-mass Procedure ------------------------------------ ###
            ###---------------------------------------------------------------------------------------------###
            if self.ModMass == True:
                QCD_unweighted = util.load(self.extraDaskDirectory+'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/QCD/'
                                           +self.BDirect+str(self.year)+'/'+self.apv+'/TTbarRes_0l_UL'+str(self.year-2000)+self.vfp+'_QCD.coffea') 
    
                # ---- Extract event counts from QCD MC hist in signal region ---- #
                QCD_hist = QCD_unweighted['jetmass'].integrate('anacat', '2t' + str(ilabel[-5:]))#.integrate('dataset', 'QCD')
                data = QCD_hist.values() # Dictionary of values

                QCD_data = [i for i in data.values()][0] # place every element of the dictionary into a numpy array

                # ---- Re-create Bins from QCD_hist as Numpy Array ---- #
                bins = np.arange(510) #Re-make bins from the jetmass_axis starting with the appropriate range
                QCD_bins = bins[::10] #Finish re-making bins by insuring exactly 50 bins like the jetmass_axis

                # ---- Define Mod Mass Distribution ---- #
                ModMass_hist_dist = ss.rv_histogram([QCD_data,QCD_bins])
                jet1_modp4 = copy.copy(jet1.p4) #J1's Lorentz four vector that can be safely modified
                jet1_modp4["fMass"] = ModMass_hist_dist.rvs(size=ak.to_awkward0(jet1_modp4).size) #Replace J1's mass with random value of mass from mm hist
                
                ttbarcands_modmass = ak.cartesian([jet0.p4, jet1_modp4])

                # ---- Apply Necessary Selections to new modmass version ---- #
                ttbarcands_modmass = ttbarcands_modmass[oneTTbar]
                ttbarcands_modmass = ttbarcands_modmass[dPhiCut]
                ttbarcands_modmass = ttbarcands_modmass[GoodSubjets]
                
                # ---- Manually sum the modmass p4 candidates (Coffea technicality) ---- #
                ttbarcands_modmassp4sum = ttbarcands.slot0.p4.add(ttbarcands.slot1.p4)
                
                # ---- Re-define Mass Variables for ModMass Procedure (pt, eta, phi are redundant to change) ---- #
                ttbarmass = ak.flatten(ttbarcands_modmassp4sum.mass)
                jetmass = ak.flatten(ttbarcands_modmass.slot1.mass)
                
            ###---------------------------------------------------------------------------------------------###
            ### ------------------------------ B-Tag Weighting (S.F. Only) -------------------------------- ###
            ###---------------------------------------------------------------------------------------------###
            if not isData:
                if (self.ApplybtagSF == True) and (self.UseEfficiencies == False):
                    Weights = Weights*Btag_wgts[str(ilabel[-5:-3])]
                    
            ###---------------------------------------------------------------------------------------------###
            ### ----------------------- Top pT Reweighting (S.F. as function of pT) ----------------------- ###
            ###---------------------------------------------------------------------------------------------###
            if ('TTbar' in dataset) and (self.ApplyTopReweight):
                Weights = Weights*ttbar_wgt

# ************************************************************************************************************ #    

            output['cutflow'][ilabel] += np.sum(icat)
            
            output['ttbarmass'].fill(dataset = dataset, anacat = ilabel, 
                                ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetpt'].fill(dataset = dataset, anacat = ilabel, 
                                jetpt = ak.to_numpy(jetpt[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['probept'].fill(dataset = dataset, anacat = ilabel, 
                                jetpt = ak.to_numpy(pT[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['probep'].fill(dataset = dataset, anacat = ilabel, 
                                jetp = ak.to_numpy(p[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jeteta'].fill(dataset = dataset, anacat = ilabel, 
                                jeteta = ak.to_numpy(jeteta[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetphi'].fill(dataset = dataset, anacat = ilabel, 
                                jetphi = ak.to_numpy(jetphi[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jety'].fill(dataset = dataset, anacat = ilabel, 
                                jety = ak.to_numpy(jety[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetdy'].fill(dataset = dataset, anacat = ilabel, 
                                jetdy = ak.to_numpy(jetdy[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['numerator'].fill(dataset = dataset, anacat = ilabel, 
                                jetp = ak.to_numpy(numerator[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['denominator'].fill(dataset = dataset, anacat = ilabel, 
                                jetp = ak.to_numpy(denominator[icat]),
                                weight = ak.to_numpy(Weights[icat]))
            output['jetmass'].fill(dataset = dataset, anacat = ilabel, 
                                   jetmass = ak.to_numpy(jetmass[icat]),
                                   weight = ak.to_numpy(Weights[icat]))
            output['SDmass'].fill(dataset = dataset, anacat = ilabel, 
                                   jetmass = ak.to_numpy(SDmass[icat]),
                                   weight = ak.to_numpy(Weights[icat]))
            output['tau32'].fill(dataset = dataset, anacat = ilabel,
                                          tau32 = ak.to_numpy(Tau32[icat]),
                                          weight = ak.to_numpy(Weights[icat]))
 
        return output

    def postprocess(self, accumulator):
        return accumulator
    
#===================================================================================================================================================    
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#===================================================================================================================================================     

"""Package to collect MC flavor efficiencies used for btagging systematics in main analysis (TTbarResProcessor) """
class MCFlavorEfficiencyProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., bdisc=0.8484, ak8PtMin=400., year=None, apv='', vfp='', RandomDebugMode=False):
        
        self.prng = prng
        self.htCut = htCut
        self.ak8PtMin = ak8PtMin
        self.year = year
        self.apv = apv
        self.vfp = vfp
        self.bdisc = bdisc
        self.RandomDebugMode = RandomDebugMode
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cats_axis = hist.Cat("anacat", "Analysis Category")

        subjetpt_axis = hist.Bin("subjetpt", r"SubJet $p_{T}$ [GeV]", 25, 0, 2000)
        subjetpt_laxis = hist.Bin("subjetpt", r"SubJet $p_{T}$ [GeV]", 8, 0, 2000) #Larger bins
        subjetpt_maxis = hist.Bin("subjetpt", r"SubJet $p_T$ [GeV]", manual_subjetpt_bins) #Manually defined bins for better statistics per bin
        
        subjeteta_axis = hist.Bin("subjeteta", r"SubJet $\eta$", 25, 0, 2.4)
        subjeteta_laxis = hist.Bin("subjeteta", r"SubJet $\eta$", 8, 0, 2.4) #Larger bins
        subjeteta_maxis = hist.Bin("subjeteta", r"SubJet $\eta$", manual_subjeteta_bins) #Manually defined bins for better statistics per bin

        
#    ====================================================================
#    EEEEEEE FFFFFFF FFFFFFF      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#    E       F       F            H     H    I     S         T     S          
#    E       F       F            H     H    I    S          T    S           
#    EEEEEEE FFFFFFF FFFFFFF      HHHHHHH    I     SSSSS     T     SSSSS      
#    E       F       F            H     H    I          S    T          S     
#    E       F       F            H     H    I         S     T         S      
#    EEEEEEE F       F       *    H     H IIIIIII SSSSS      T    SSSSS
#    ====================================================================

        self._accumulator = processor.dict_accumulator({

            # ---- 2D SubJet b-tag Efficiency ---- #
            'b_eff_numerator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_numerator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_numerator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_numerator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'b_eff_denominator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_denominator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_denominator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'b_eff_denominator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'b_eff_numerator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_numerator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_numerator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_numerator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'b_eff_denominator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_denominator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_denominator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'b_eff_denominator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'b_eff_numerator_s01_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'b_eff_numerator_s02_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'b_eff_numerator_s11_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'b_eff_numerator_s12_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            
            'b_eff_denominator_s01_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'b_eff_denominator_s02_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'b_eff_denominator_s11_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'b_eff_denominator_s12_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            
            # ---- 2D SubJet c-tag Efficiency ---- #
            'c_eff_numerator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_numerator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_numerator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_numerator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'c_eff_denominator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_denominator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_denominator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'c_eff_denominator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'c_eff_numerator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_numerator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_numerator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_numerator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'c_eff_denominator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_denominator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_denominator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'c_eff_denominator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'c_eff_numerator_s01_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'c_eff_numerator_s02_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'c_eff_numerator_s11_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'c_eff_numerator_s12_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            
            'c_eff_denominator_s01_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'c_eff_denominator_s02_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'c_eff_denominator_s11_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'c_eff_denominator_s12_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            
            # ---- 2D SubJet light quark-tag Efficiency ---- #
            'udsg_eff_numerator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_numerator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_numerator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_numerator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'udsg_eff_denominator_s01': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_denominator_s02': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_denominator_s11': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            'udsg_eff_denominator_s12': hist.Hist("Counts", dataset_axis, subjetpt_axis, subjeteta_axis),
            
            'udsg_eff_numerator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_numerator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_numerator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_numerator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'udsg_eff_denominator_s01_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_denominator_s02_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_denominator_s11_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            'udsg_eff_denominator_s12_largerbins': hist.Hist("Counts", dataset_axis, subjetpt_laxis, subjeteta_laxis),
            
            'udsg_eff_numerator_s01_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'udsg_eff_numerator_s02_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'udsg_eff_numerator_s11_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'udsg_eff_numerator_s12_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            
            'udsg_eff_denominator_s01_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'udsg_eff_denominator_s02_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'udsg_eff_denominator_s11_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            'udsg_eff_denominator_s12_manualbins': hist.Hist("Counts", dataset_axis, subjetpt_maxis, subjeteta_maxis),
            
            #********************************************************************************************************************#
            
            'cutflow': processor.defaultdict_accumulator(int),
            
        })
        
#   =======================================================================
#   FFFFFFF U     U N     N   CCCC  TTTTTTT IIIIIII   OOO   N     N   SSSSS     
#   F       U     U NN    N  C         T       I     O   O  NN    N  S          
#   F       U     U N N   N C          T       I    O     O N N   N S           
#   FFFFFFF U     U N  N  N C          T       I    O     O N  N  N  SSSSS      
#   F       U     U N   N N C          T       I    O     O N   N N       S     
#   F        U   U  N    NN  C         T       I     O   O  N    NN      S      
#   F         UUU   N     N   CCCC     T    IIIIIII   OOO   N     N SSSSS
#   =======================================================================

    
    def GetFlavorEfficiency(self, Subjet, Flavor): # Return "Flavor" efficiency numerator and denominator
        '''
        Subjet --> awkward array object after preselection i.e. SubJetXY
        Flavor --> integer i.e 5, 4, or 0 (b, c, or udsg)
        '''
        # --- Define pT and Eta for Both Candidates' Subjets (for simplicity) --- #
        pT = ak.flatten(Subjet.pt) # pT of subjet in ttbarcand 
        eta = np.abs(ak.flatten(Subjet.eta)) # eta of 1st subjet in ttbarcand 
        flav = np.abs(ak.flatten(Subjet.hadronFlavour)) # either 'normal' or 'anti' quark
        
        subjet_btagged = (Subjet.btagCSVV2 > self.bdisc)
        
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
            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        
        # ---- Define dataset ---- #
        dataset = events.metadata['dataset']

#    ===========================================================================================    
#    N     N    A    N     N   OOO      A      OOO   DDDD          OOO   BBBBBB  JJJJJJJ   SSSSS     
#    NN    N   A A   NN    N  O   O    A A    O   O  D   D        O   O  B     B    J     S          
#    N N   N  A   A  N N   N O     O  A   A  O     O D    D      O     O B     B    J    S           
#    N  N  N  AAAAA  N  N  N O     O  AAAAA  O     O D     D     O     O BBBBBB     J     SSSSS      
#    N   N N A     A N   N N O     O A     A O     O D    D      O     O B     B J  J          S     
#    N    NN A     A N    NN  O   O  A     A  O   O  D   D        O   O  B     B J  J         S      
#    N     N A     A N     N   OOO   A     A   OOO   DDDD          OOO   BBBBBB   JJ     SSSSS 
#    =========================================================================================== 

        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
        
        # ---- Define AK8 Jets as FatJets ---- #
        #FatJets = events.FatJet # Everything should already be defined in here.  example) df['FatJet_pt] -> events.FatJet.pt
        FatJets = ak.zip({
            "run": events.run,
            "nFatJet": events.nFatJet,
            "pt": events.FatJet_pt,
            "eta": events.FatJet_eta,
            "phi": events.FatJet_phi,
            "mass": events.FatJet_mass,
            "area": events.FatJet_area,
            "msoftdrop": events.FatJet_msoftdrop,
            "jetId": events.FatJet_jetId,
            "tau1": events.FatJet_tau1,
            "tau2": events.FatJet_tau2,
            "tau3": events.FatJet_tau3,
            "tau4": events.FatJet_tau4,
            "n3b1": events.FatJet_n3b1,
            "btagDeepB": events.FatJet_btagDeepB,
            "btagCSVV2": events.FatJet_btagCSVV2, # Use as a prior probability of containing a bjet?
            "deepTag_TvsQCD": events.FatJet_deepTag_TvsQCD,
            "deepTagMD_TvsQCD": events.FatJet_deepTagMD_TvsQCD,
            "subJetIdx1": events.FatJet_subJetIdx1,
            "subJetIdx2": events.FatJet_subJetIdx2,
            "p4": ak.zip({
                "pt": events.FatJet_pt,
                "eta": events.FatJet_eta,
                "phi": events.FatJet_phi,
                "mass": events.FatJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define AK4 jets as Jets ---- #
        Jets = ak.zip({
            "run": events.run,
            "pt": events.Jet_pt,
            "eta": events.Jet_eta,
            "phi": events.Jet_phi,
            "mass": events.Jet_mass,
            "area": events.Jet_area,
            "p4": ak.zip({
                "pt": events.Jet_pt,
                "eta": events.Jet_eta,
                "phi": events.Jet_phi,
                "mass": events.Jet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define SubJets ---- #
        SubJets = ak.zip({
            "run": events.run,
            "pt": events.SubJet_pt,
            "eta": events.SubJet_eta,
            "phi": events.SubJet_phi,
            "mass": events.SubJet_mass,
            "btagDeepB": events.SubJet_btagDeepB,
            "btagCSVV2": events.SubJet_btagCSVV2,
            "p4": ak.zip({
                "pt": events.SubJet_pt,
                "eta": events.SubJet_eta,
                "phi": events.SubJet_phi,
                "mass": events.SubJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })
        
        Jets['hadronFlavour'] = events.Jet_hadronFlavour
        SubJets['hadronFlavour'] = events.SubJet_hadronFlavour
            
#    ===================================================================================
#    PPPPPP  RRRRRR  EEEEEEE L       IIIIIII M     M       CCCC  U     U TTTTTTT   SSSSS     
#    P     P R     R E       L          I    MM   MM      C      U     U    T     S          
#    P     P R     R E       L          I    M M M M     C       U     U    T    S           
#    PPPPPP  RRRRRR  EEEEEEE L          I    M  M  M     C       U     U    T     SSSSS      
#    P       R   R   E       L          I    M     M     C       U     U    T          S     
#    P       R    R  E       L          I    M     M      C       U   U     T         S      
#    P       R     R EEEEEEE LLLLLLL IIIIIII M     M       CCCC    UUU      T    SSSSS
#    ===================================================================================
        
        # ---- Get event weights from dataset ---- #
        if isData: # If data is used...
            # print('if isData command works')
            evtweights = np.ones(ak.to_awkward0(FatJets).size) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = events.Generator_weight
        # ---- Show all events ---- #
        output['cutflow']['all events'] += ak.to_awkward0(FatJets).size

        # ---- Apply HT Cut ---- #
        # ---- This gives the analysis 99.8% efficiency (see 2016 AN) ---- #
        hT = ak.to_awkward0(Jets.pt).sum()
        passhT = (hT > self.htCut)
        FatJets = FatJets[passhT]
        Jets = Jets[passhT] # this used to not be here
        SubJets = SubJets[passhT]
        evtweights = evtweights[passhT]
        output['cutflow']['HT Cut'] += ak.to_awkward0(passhT).sum()
           
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['Loose Jet ID'] += ak.to_awkward0(jet_id).any().sum()
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['pT,y Cut'] += ak.to_awkward0(jetkincut_index).any().sum()
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) == 2)
        FatJets = FatJets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin] # this used to not be here
        evtweights = evtweights[twoFatJetsKin]
        output['cutflow']['two FatJets'] += ak.to_awkward0(twoFatJetsKin).sum()
        
        # ---- Randomly Assign AK8 Jets as TTbar Candidates 0 and 1 --- #
        Counts = np.ones(len(FatJets), dtype='i') # Number 1 for each FatJet
        
        if self.RandomDebugMode == True: # 'Sudo' randomizer to test for consistent results
            highPhi = FatJets.phi[:,0] > FatJets.phi[:,1]
            highRandIndex = np.where(highPhi, 0, 1) # 1D array of 0's and 1's
            index = ak.unflatten( highRandIndex, Counts ) # Subtly confusing logic of what this does is shown below
            """
                For example:
                
                    FatJets = [[FatJet0, FatJet1], [AnotherFatJet0, AnotherFatJet1], ..., [LastFatJet0, LastFatJet1]]
                    Counts = [1, 1, 1, 1, 1, 1]
                    highRandIndex = [1, 1, 0, 1, 0, 0]
                    
                unflattening highRandIndex with Counts will group "Counts" number of highRandIndex elements together 
                in a new higher order array:
                
                    index = [[1], [1], [0], [1], [0], [0]]
                    1 - index = [[0], [0], [1], [0], [1], [1]]
                    
               where the index is used to slice (select) either the first FatJet (FatJets[:,0]) or the second (FatJets[:,1]):
        
                    FatJets[index] = [FatJet1, AnotherFatJet1, ..., LastFatJet0]
                    FatJets[1-index] = [FatJet0, AnotherFatJet0, ..., LastFatJet1]
            """
        else: # Truly randomize
            index = ak.unflatten( self.prng.randint(2, size=len(FatJets)), Counts )
        
        jet0 = FatJets[index] #J0
        jet1 = FatJets[1 - index] #J1
        
        ttbarcands = ak.cartesian([jet0, jet1]) # Re-group the randomized pairs in a similar fashion to how they were
        
        """ NOTE that ak.cartesian gives a shape with one more layer than FatJets """
        # ---- Make sure we have at least 1 TTbar candidate pair and re-broadcast releveant arrays  ---- #
        oneTTbar = (ak.num(ttbarcands, axis=-1) >= 1)
        output['cutflow']['>= oneTTbar'] += ak.to_awkward0(oneTTbar).sum()
        ttbarcands = ttbarcands[oneTTbar]
        FatJets = FatJets[oneTTbar]
        Jets = Jets[oneTTbar] # this used to not be here
        SubJets = SubJets[oneTTbar]
        evtweights = evtweights[oneTTbar]
            
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        output['cutflow']['dPhi Cut'] += ak.to_awkward0(dPhiCut).sum()
        ttbarcands = ttbarcands[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        Jets = Jets[dPhiCut] # this used to not be here
        SubJets = SubJets[dPhiCut] 
        evtweights = evtweights[dPhiCut]
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
        output['cutflow']['Good Subjets'] += ak.to_awkward0(GoodSubjets).sum()
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        SubJets = SubJets[GoodSubjets]
        Jets = Jets[GoodSubjets] # this used to not be here
        evtweights = evtweights[GoodSubjets]
        
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # ttbarcandidate 0's first subjet 
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # ttbarcandidate 0's second subjet
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # ttbarcandidate 1's first subjet 
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # ttbarcandidate 1's second subjet
        
#       ===============================================================================
#       M     M   CCCC      FFFFFFF L          A    V     V     EEEEEEE FFFFFFF FFFFFFF     
#       MM   MM  C          F       L         A A   V     V     E       F       F           
#       M M M M C           F       L        A   A  V     V     E       F       F           
#       M  M  M C           FFFFFFF L        AAAAA  V     V     EEEEEEE FFFFFFF FFFFFFF     
#       M     M C           F       L       A     A  V   V      E       F       F           
#       M     M  C          F       L       A     A   V V       E       F       F           
#       M     M   CCCC      F       LLLLLLL A     A    V        EEEEEEE F       F  
#       ===============================================================================
        
        if not isData:

            bFlavEff01 = self.GetFlavorEfficiency(SubJet01, 5)
            bFlavEff02 = self.GetFlavorEfficiency(SubJet02, 5)
            bFlavEff11 = self.GetFlavorEfficiency(SubJet11, 5)
            bFlavEff12 = self.GetFlavorEfficiency(SubJet12, 5)

            cFlavEff01 = self.GetFlavorEfficiency(SubJet01, 4)
            cFlavEff02 = self.GetFlavorEfficiency(SubJet02, 4)
            cFlavEff11 = self.GetFlavorEfficiency(SubJet11, 4)
            cFlavEff12 = self.GetFlavorEfficiency(SubJet12, 4)

            lFlavEff01 = self.GetFlavorEfficiency(SubJet01, 0)
            lFlavEff02 = self.GetFlavorEfficiency(SubJet02, 0)
            lFlavEff11 = self.GetFlavorEfficiency(SubJet11, 0)
            lFlavEff12 = self.GetFlavorEfficiency(SubJet12, 0)

            # **************************************************************************************** #
            # ----------------------------- 2-D B-tagging Efficiencies ------------------------------- #
            # **************************************************************************************** #
            output['b_eff_numerator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_denominator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_numerator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_numerator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['b_eff_denominator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['b_eff_denominator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(bFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(bFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))

            # **************************************************************************************** #
            # ----------------------------- 2-D C-tagging Efficiencies ------------------------------- #
            # **************************************************************************************** #
            output['c_eff_numerator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_denominator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_numerator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_numerator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['c_eff_denominator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['c_eff_denominator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(cFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(cFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))

            # **************************************************************************************** #
            # ------------------------ 2-D Light Parton-tagging Efficiencies ------------------------- #
            # **************************************************************************************** #
            output['udsg_eff_numerator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_denominator_s01'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s02'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s11'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s12'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_numerator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_denominator_s01_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s02_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s11_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s12_largerbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_numerator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_numerator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Num_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Num_eta']),
                                              weight = ak.to_numpy(evtweights))
            #-----------------------------------------------------------------------------------------#
            output['udsg_eff_denominator_s01_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff01['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff01['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s02_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff02['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff02['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s11_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff11['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff11['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
            output['udsg_eff_denominator_s12_manualbins'].fill(dataset = dataset,
                                              subjetpt = ak.to_numpy(lFlavEff12['Denom_pT']),
                                              subjeteta = ak.to_numpy(lFlavEff12['Denom_eta']),
                                              weight = ak.to_numpy(evtweights))
 
        return output

    def postprocess(self, accumulator):
        return accumulator    
    
#===================================================================================================================================================    
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
       
"""Package to perform trigger analysis before applying triggers in main analysis (TTbarResProcessor)"""
class TriggerAnalysisProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., minMSD=105., maxMSD=210.,
                 tau32Cut=0.65, ak8PtMin=400., bdisc=0.8484, deepAK8Cut=0.435,
                 year=None, apv='', vfp='', RandomDebugMode=False):
        
        self.prng = prng
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.deepAK8Cut = deepAK8Cut
        self.year = year
        self.apv = apv
        self.vfp = vfp
        self.RandomDebugMode = RandomDebugMode

        
        # --- 0, >=1 ttags --- #
        self.ttagcats_forTriggerAnalysis = ["0t", ">=1t"]
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cats_axis = hist.Cat("anacat", "Analysis Category")
        jetht_axis = hist.Bin("Jet_HT", r'$AK4\ Jet\ HT$', manual_jetht_bins) # Used for Trigger Analysis
        
#    ===================================================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#       T    R     R    I    G       G       E       R     R     H     H    I     S         T     S          
#       T    R     R    I    G       G       E       R     R     H     H    I    S          T    S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR      HHHHHHH    I     SSSSS     T     SSSSS      
#       T    R   R      I    G     G G     G E       R   R       H     H    I          S    T          S     
#       T    R    R     I    G     G G     G E       R    R      H     H    I         S     T         S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R     H     H IIIIIII SSSSS      T    SSSSS
#    ===================================================================================================

        self._accumulator = processor.dict_accumulator({
           'condition1_numerator': hist.Hist("Counts", dataset_axis, cats_axis, jetht_axis),
           'condition2_numerator': hist.Hist("Counts", dataset_axis, cats_axis, jetht_axis),
           'condition3_numerator': hist.Hist("Counts", dataset_axis, cats_axis, jetht_axis),
           'condition4_numerator': hist.Hist("Counts", dataset_axis, cats_axis, jetht_axis),
           'condition_denominator': hist.Hist("Counts", dataset_axis, cats_axis, jetht_axis),
            
            'cutflow': processor.defaultdict_accumulator(int),
        })
        
            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()
        
        # ---- Define dataset ---- #
        dataset = events.metadata['dataset']

#    ===========================================================================================    
#    N     N    A    N     N   OOO      A      OOO   DDDD          OOO   BBBBBB  JJJJJJJ   SSSSS     
#    NN    N   A A   NN    N  O   O    A A    O   O  D   D        O   O  B     B    J     S          
#    N N   N  A   A  N N   N O     O  A   A  O     O D    D      O     O B     B    J    S           
#    N  N  N  AAAAA  N  N  N O     O  AAAAA  O     O D     D     O     O BBBBBB     J     SSSSS      
#    N   N N A     A N   N N O     O A     A O     O D    D      O     O B     B J  J          S     
#    N    NN A     A N    NN  O   O  A     A  O   O  D   D        O   O  B     B J  J         S      
#    N     N A     A N     N   OOO   A     A   OOO   DDDD          OOO   BBBBBB   JJ     SSSSS 
#    =========================================================================================== 

        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
        
        # ---- Define AK8 Jets as FatJets ---- #
        #FatJets = events.FatJet # Everything should already be defined in here.  example) df['FatJet_pt] -> events.FatJet.pt
        FatJets = ak.zip({
            "run": events.run,
            "nFatJet": events.nFatJet,
            "pt": events.FatJet_pt,
            "eta": events.FatJet_eta,
            "phi": events.FatJet_phi,
            "mass": events.FatJet_mass,
            "msoftdrop": events.FatJet_msoftdrop,
            "jetId": events.FatJet_jetId,
            "deepTag_TvsQCD": events.FatJet_deepTag_TvsQCD,
            "deepTagMD_TvsQCD": events.FatJet_deepTagMD_TvsQCD,
            "subJetIdx1": events.FatJet_subJetIdx1,
            "subJetIdx2": events.FatJet_subJetIdx2,
            "p4": ak.zip({
                "pt": events.FatJet_pt,
                "eta": events.FatJet_eta,
                "phi": events.FatJet_phi,
                "mass": events.FatJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define AK4 jets as Jets ---- #
        Jets = ak.zip({
            "run": events.run,
            "pt": events.Jet_pt,
            "eta": events.Jet_eta,
            "phi": events.Jet_phi,
            "mass": events.Jet_mass,
            "area": events.Jet_area,
            "p4": ak.zip({
                "pt": events.Jet_pt,
                "eta": events.Jet_eta,
                "phi": events.Jet_phi,
                "mass": events.Jet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })

        # ---- Define SubJets ---- #
        SubJets = ak.zip({
            "run": events.run,
            "pt": events.SubJet_pt,
            "eta": events.SubJet_eta,
            "phi": events.SubJet_phi,
            "mass": events.SubJet_mass,
            "btagDeepB": events.SubJet_btagDeepB,
            "btagCSVV2": events.SubJet_btagCSVV2,
            "p4": ak.zip({
                "pt": events.SubJet_pt,
                "eta": events.SubJet_eta,
                "phi": events.SubJet_phi,
                "mass": events.SubJet_mass,
                }, with_name="PtEtaPhiMLorentzVector"),
            })
            
            
#    ================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR    SSSSS     
#       T    R     R    I    G       G       E       R     R  S          
#       T    R     R    I    G       G       E       R     R S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR   SSSSS      
#       T    R   R      I    G     G G     G E       R   R         S     
#       T    R    R     I    G     G G     G E       R    R       S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R SSSSS  
#    ================================================================
        
#         Dataset_info = events.fields # All nanoaod events
#         listOfTriggers = np.array([name for name in Dataset_info if 'HLT' in name]) # Find event name info that have HLT to find all relevant triggers
        
#         isHLT_PF = np.array(['HLT_PF' in i for i in listOfTriggers])
#         isHLT_AK8 = np.array(['HLT_AK8' in i for i in listOfTriggers])
        
#         HLT_PF_triggers = listOfTriggers[isHLT_PF]
#         HLT_AK8_triggers = listOfTriggers[isHLT_AK8]
        
        
        trigDenom = events.HLT_Mu50 | events.HLT_IsoMu24 # WHY!!!!!????
        # print(events.HLT_Mu50)
        # print(events.HLT_IsoMu24)
        # print(trigDenom)
        # print('-----------------------------------------------')
        
        if self.year == 2016:
            triggers2016_1 = events.HLT_PFHT900
            triggers2016_2 = events.HLT_AK8PFHT700_TrimR0p1PT0p03Mass50
            triggers2016_3 = events.HLT_AK8PFJet450
            triggers2016_4 = events.HLT_AK8PFJet360_TrimMass30

#    ===================================================================================
#    PPPPPP  RRRRRR  EEEEEEE L       IIIIIII M     M       CCCC  U     U TTTTTTT   SSSSS     
#    P     P R     R E       L          I    MM   MM      C      U     U    T     S          
#    P     P R     R E       L          I    M M M M     C       U     U    T    S           
#    PPPPPP  RRRRRR  EEEEEEE L          I    M  M  M     C       U     U    T     SSSSS      
#    P       R   R   E       L          I    M     M     C       U     U    T          S     
#    P       R    R  E       L          I    M     M      C       U   U     T         S      
#    P       R     R EEEEEEE LLLLLLL IIIIIII M     M       CCCC    UUU      T    SSSSS
#    ===================================================================================
        
        # ---- Get event weights from dataset ---- #
        if isData: # If data is used...
            # print('if isData command works')
            evtweights = np.ones(ak.to_awkward0(FatJets).size) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = events.Generator_weight
        # ---- Show all events ---- #
        output['cutflow']['all events'] += ak.to_awkward0(FatJets).size
        
        # ---- Setup Trigger Analysis Conditions in higher scope ---- #
        condition1 = None
        condition2 = None
        condition3 = None
        condition4 = None
        firstCondition = None
        
        # ---- Apply Trigger(s) ---- # 
        if self.year == 2016: # Include other years later...
            firstCondition = (triggers2016_1) 
            
        # ---- Defining Trigger Analysis Conditions ---- #
        condition1 = firstCondition & trigDenom # WHY?! Figure out later...
        condition2 = (firstCondition | triggers2016_2) & trigDenom
        condition3 = ((firstCondition | triggers2016_2) | triggers2016_3) & trigDenom
        condition4 = ((firstCondition | triggers2016_2) | (triggers2016_3 | triggers2016_4)) & trigDenom
        
            
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['events with Loose Jet ID'] += ak.to_awkward0(jet_id).any().sum()
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['events with pT,y Cut'] += ak.to_awkward0(jetkincut_index).any().sum()
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) == 2)
        FatJets = FatJets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin] # this used to not be here

        condition1 = condition1[twoFatJetsKin]
        condition2 = condition2[twoFatJetsKin]
        condition3 = condition3[twoFatJetsKin]
        condition4 = condition4[twoFatJetsKin]
        trigDenom = trigDenom[twoFatJetsKin]
        evtweights = evtweights[twoFatJetsKin]
        output['cutflow']['events with two FatJets'] += ak.to_awkward0(twoFatJetsKin).sum()
        
        # ---- Randomly Assign AK8 Jets as TTbar Candidates 0 and 1 --- #
        Counts = np.ones(len(FatJets), dtype='i') # Number 1 for each FatJet
        
        if self.RandomDebugMode == True: # 'Sudo' randomizer to test for consistent results
            highPhi = FatJets.phi[:,0] > FatJets.phi[:,1]
            highRandIndex = np.where(highPhi, 0, 1) # 1D array of 0's and 1's
            index = ak.unflatten( highRandIndex, Counts ) # Subtly confusing logic of what this does is shown below
            """
                For example:
                
                    FatJets = [[FatJet0, FatJet1], [AnotherFatJet0, AnotherFatJet1], ..., [LastFatJet0, LastFatJet1]]
                    Counts = [1, 1, 1, 1, 1, 1]
                    highRandIndex = [1, 1, 0, 1, 0, 0]
                    
                unflattening highRandIndex with Counts will group "Counts" number of highRandIndex elements together 
                in a new higher order array:
                
                    index = [[1], [1], [0], [1], [0], [0]]
                    1 - index = [[0], [0], [1], [0], [1], [1]]
                    
               where the index is used to slice (select) either the first FatJet (FatJets[:,0]) or the second (FatJets[:,1]):
        
                    FatJets[index] = [FatJet1, AnotherFatJet1, ..., LastFatJet0]
                    FatJets[1-index] = [FatJet0, AnotherFatJet0, ..., LastFatJet1]
            """
        else: # Truly randomize
            index = ak.unflatten( self.prng.randint(2, size=len(FatJets)), Counts )
        
        jet0 = FatJets[index] #J0
        jet1 = FatJets[1 - index] #J1
        
        ttbarcands = ak.cartesian([jet0, jet1]) # Re-group the randomized pairs in a similar fashion to how they were
        
        """ NOTE that ak.cartesian gives a shape with one more layer than FatJets """
        # ---- Make sure we have at least 1 TTbar candidate pair and re-broadcast releveant arrays  ---- #
        oneTTbar = (ak.num(ttbarcands, axis=-1) >= 1)
        output['cutflow']['events with >= oneTTbar'] += ak.to_awkward0(oneTTbar).sum()
        ttbarcands = ttbarcands[oneTTbar]
        FatJets = FatJets[oneTTbar]
        Jets = Jets[oneTTbar] # this used to not be here
        
        condition1 = condition1[oneTTbar]
        condition2 = condition2[oneTTbar]
        condition3 = condition3[oneTTbar]
        condition4 = condition4[oneTTbar]
        trigDenom = trigDenom[oneTTbar]
        SubJets = SubJets[oneTTbar]
        evtweights = evtweights[oneTTbar]
            
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        output['cutflow']['events with dPhi Cut'] += ak.to_awkward0(dPhiCut).sum()
        ttbarcands = ttbarcands[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        Jets = Jets[dPhiCut] # this used to not be here
        
        condition1 = condition1[dPhiCut]
        condition2 = condition2[dPhiCut]
        condition3 = condition3[dPhiCut]
        condition4 = condition4[dPhiCut]
        trigDenom = trigDenom[dPhiCut]
        SubJets = SubJets[dPhiCut] 
        evtweights = evtweights[dPhiCut]
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets
        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
        output['cutflow']['events with Good Subjets'] += ak.to_awkward0(GoodSubjets).sum()
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        SubJets = SubJets[GoodSubjets]
        Jets = Jets[GoodSubjets] # this used to not be here
        
        condition1 = condition1[GoodSubjets]
        condition2 = condition2[GoodSubjets]
        condition3 = condition3[GoodSubjets]
        condition4 = condition4[GoodSubjets]
        trigDenom = trigDenom[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
        
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # ttbarcandidate 0's first subjet 
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # ttbarcandidate 0's second subjet
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # ttbarcandidate 1's first subjet 
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # ttbarcandidate 1's second subjet
        
        

#    ============================================================
#    TTTTTTT     TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR  
#       T           T      A A   G       G       E       R     R 
#       T           T     A   A  G       G       E       R     R 
#       T           T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR  
#       T           T    A     A G     G G     G E       R   R   
#       T           T    A     A G     G G     G E       R    R   
#       T           T    A     A  GGGGG   GGGGG  EEEEEEE R     R
#    ============================================================

        # ----------- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ----------- #
        # ---- NOTE: Must Change This to DeepAK8 Top Tag Discriminator Cut ----#
        # ---- Maybe we should ignore tau32 cut(s) when performing trigger analysis ---- #
        
        mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
        mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 
        
        ttag_s0 = mcut_s0
        ttag_s1 = mcut_s1
        
        # ---- Define "Top Tag" Regions ---- #
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)

                    

#    ===========================================================================================================================        
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR         A    N     N    A    L       Y     Y   SSSSS IIIIIII   SSSSS     
#       T    R     R    I    G       G       E       R     R       A A   NN    N   A A   L        Y   Y   S         I     S          
#       T    R     R    I    G       G       E       R     R      A   A  N N   N  A   A  L         Y Y   S          I    S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR       AAAAA  N  N  N  AAAAA  L          Y     SSSSS     I     SSSSS      
#       T    R   R      I    G     G G     G E       R   R       A     A N   N N A     A L          Y          S    I          S     
#       T    R    R     I    G     G G     G E       R    R      A     A N    NN A     A L          Y         S     I         S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R     A     A N     N A     A LLLLLLL    Y    SSSSS   IIIIIII SSSSS
#    ===========================================================================================================================  
        
        ConditionsDict = {
            '1': condition1,
            '2': condition2,
            '3': condition3,
            '4': condition4
        }

        # ---- Defining Jet Collections for Trigger Analysis Numerator and Denominator ---- #
        Jets_NumCondition1 = Jets[condition1] # contains jets to be used as numerator for trigger eff
        Jets_NumCondition2 = Jets[condition2]
        Jets_NumCondition3 = Jets[condition3]
        Jets_NumCondition4 = Jets[condition4]
        Jets_DenomCondition = Jets[trigDenom] # contains jets to be used as denominator for trigger eff
        output['cutflow']['events with jets cond1'] +=  ak.to_awkward0(condition1).sum()
        output['cutflow']['events with jets cond2'] +=  ak.to_awkward0(condition2).sum()
        output['cutflow']['events with jets cond3'] +=  ak.to_awkward0(condition3).sum()
        output['cutflow']['events with jets cond4'] +=  ak.to_awkward0(condition4).sum()
        output['cutflow']['events with jets Denom cond'] +=  ak.to_awkward0(trigDenom).sum()
        
        # ---- Must pass this cut before calculating HT variables for analysis ---- #
        passAK4_num1 = (Jets_NumCondition1.pt > 30.) & (np.abs(Jets_NumCondition1.eta) < 3.0) 
        passAK4_num2 = (Jets_NumCondition2.pt > 30.) & (np.abs(Jets_NumCondition2.eta) < 3.0)
        passAK4_num3 = (Jets_NumCondition3.pt > 30.) & (np.abs(Jets_NumCondition3.eta) < 3.0)
        passAK4_num4 = (Jets_NumCondition4.pt > 30.) & (np.abs(Jets_NumCondition4.eta) < 3.0)
        passAK4_denom = (Jets_DenomCondition.pt > 30.) & (np.abs(Jets_DenomCondition.eta) < 3.0) 
        
        # ---------------------------------------------------------------------------------------------#
        # ---- Remember to have weights array that is consistently the same size as num and denom ---- #
        # --------- This is only because input later in the code expects an array of weights --------- #
        # ------------- despite this array being simply an array of ones for the sake of ------------- #
        # ----------------------------------- not altering the data ---------------------------------- #
        # ---------------------------------------------------------------------------------------------#
        Num1Wgt = evtweights[condition1]
        Num2Wgt = evtweights[condition2]
        Num3Wgt = evtweights[condition3]
        Num4Wgt = evtweights[condition4]
        NumWgtDict = {
            '1': Num1Wgt,
            '2': Num2Wgt,
            '3': Num3Wgt,
            '4': Num4Wgt
        }
        DenomWgt = evtweights[trigDenom]
        
        # ---- Defining Trigger Analysis Numerator(s) and Denominator ---- #
        jet_HT_numerator1 = ak.sum(Jets_NumCondition1[passAK4_num1].pt, axis=-1) # Sum over each AK4 Jet per event
        jet_HT_numerator2 = ak.sum(Jets_NumCondition2[passAK4_num2].pt, axis=-1)
        jet_HT_numerator3 = ak.sum(Jets_NumCondition3[passAK4_num3].pt, axis=-1)
        jet_HT_numerator4 = ak.sum(Jets_NumCondition4[passAK4_num4].pt, axis=-1)
        jet_HT_numeratorDict = {
            '1': jet_HT_numerator1,
            '2': jet_HT_numerator2,
            '3': jet_HT_numerator3,
            '4': jet_HT_numerator4
        }
        jet_HT_denominator = ak.sum(Jets_DenomCondition[passAK4_denom].pt, axis=-1) # Sum over each AK4 Jet per event
        output['cutflow']['jets cond1 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num1)).sum()
        output['cutflow']['jets cond2 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num2)).sum()
        output['cutflow']['jets cond3 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num3)).sum()
        output['cutflow']['jets cond4 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num4)).sum()
        output['cutflow']['jets Denom with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_denom)).sum()
        
        # ----------------- Keep track of cutflow for individual bins ---------------- #
        # ---- [200, 800, 840, 880, 920, 960, 1000, 1200, 1400, 1600, 1800, 2000] ---- #
        num1_inBin1 = (200. < jet_HT_numerator1) & (jet_HT_numerator1 < 800.)
        num2_inBin1 = (200. < jet_HT_numerator2) & (jet_HT_numerator2 < 800.)
        num3_inBin1 = (200. < jet_HT_numerator3) & (jet_HT_numerator3 < 800.)
        num4_inBin1 = (200. < jet_HT_numerator4) & (jet_HT_numerator4 < 800.)
        denom_inBin1 = (200. < jet_HT_denominator) & (jet_HT_denominator < 800.)
        num1_inBin11 = (1800. < jet_HT_numerator1) & (jet_HT_numerator1 < 2000.)
        num2_inBin11 = (1800. < jet_HT_numerator2) & (jet_HT_numerator2 < 2000.)
        num3_inBin11 = (1800. < jet_HT_numerator3) & (jet_HT_numerator3 < 2000.)
        num4_inBin11 = (1800. < jet_HT_numerator4) & (jet_HT_numerator4 < 2000.)
        denom_inBin11 = (1800. < jet_HT_denominator) & (jet_HT_denominator < 2000.)
        output['cutflow']['numerator 1 in bin [200, 800]'] += ak.to_awkward0(num1_inBin1).sum()
        output['cutflow']['numerator 2 in bin [200, 800]'] += ak.to_awkward0(num2_inBin1).sum()
        output['cutflow']['numerator 3 in bin [200, 800]'] += ak.to_awkward0(num3_inBin1).sum()
        output['cutflow']['numerator 4 in bin [200, 800]'] += ak.to_awkward0(num4_inBin1).sum()
        output['cutflow']['denominator in bin [200, 800]'] += ak.to_awkward0(denom_inBin1).sum()
        output['cutflow']['numerator 1 in bin [1800, 2000]'] += ak.to_awkward0(num1_inBin11).sum()
        output['cutflow']['numerator 2 in bin [1800, 2000]'] += ak.to_awkward0(num2_inBin11).sum()
        output['cutflow']['numerator 3 in bin [1800, 2000]'] += ak.to_awkward0(num3_inBin11).sum()
        output['cutflow']['numerator 4 in bin [1800, 2000]'] += ak.to_awkward0(num4_inBin11).sum()
        output['cutflow']['denominator in bin [1800, 2000]'] += ak.to_awkward0(denom_inBin11).sum()
        
        # ---- Define Categories for Trigger Analysis Denominator and Fill Hists ---- #
        ttags = [ttag0[trigDenom],ttagI[trigDenom]]
        cats = [ ak.to_awkward0(ak.flatten(t)) for t in ttags ]
        labels_and_categories = dict(zip( self.ttagcats_forTriggerAnalysis, cats ))
        for ilabel,icat in labels_and_categories.items():
            output['condition_denominator'].fill(dataset = dataset, anacat = ilabel, 
                                                Jet_HT = ak.to_numpy(jet_HT_denominator[icat]),
                                                weight = ak.to_numpy(DenomWgt[icat]))
        # ---- Define Categories for Trigger Analysis Numerators and Fill Hists---- #
        for i in range(1, len(ConditionsDict)+1):
            c = ConditionsDict[str(i)]
            n = jet_HT_numeratorDict[str(i)]
            w = NumWgtDict[str(i)]
            ttags = [ttag0[c],ttagI[c]]
            cats = [ ak.to_awkward0(ak.flatten(t)) for t in ttags ]
            labels_and_categories = dict(zip( self.ttagcats_forTriggerAnalysis, cats))
            for ilabel,icat in labels_and_categories.items():
                output['condition' + str(i) + '_numerator'].fill(dataset = dataset, anacat = ilabel, 
                                                                Jet_HT = ak.to_numpy(n[icat]),
                                                                weight = ak.to_numpy(w[icat]))
 
        return output

    def postprocess(self, accumulator):
        return accumulator