import numpy as np
import awkward as ak
import correctionlib

def btagCorrections(btags, subjets, isData, bdisc, sysType='central'):
    
    
    btag0, btag1, btag2 = btags
    SubJet01, SubJet02, SubJet11, SubJet12 = subjets
    
    
    btag_s0 = ( np.maximum(SubJet01.btagDeepB , SubJet02.btagDeepB) > bdisc )
    btag_s1 = ( np.maximum(SubJet11.btagDeepB , SubJet12.btagDeepB) > bdisc )
    
    Btag_wgts = {} # To be filled with "btag_wgts" corrections below (Needs to be defined for higher scope)
    
    if not isData:


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
        ------------------------------------------------------------------------------------------------------
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
        SF_filename = 'data/corrections/subjet_btagging.json.gz'
        Fitting = "M"
        if bdisc < 0.5:
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
            BSF_s0_allHeavy = btag_sf['deepCSV_subjet'].evaluate(sysType, 'lt', Fitting, s0_allHeavy, abs(s0_eta), s0_pt)
        except RuntimeError as re:
            print('flavor (with light mask): \n', s0_allHeavy, flush=True)
            print('eta: \n', s0_eta, flush=True)
            print('pt: \n', s0_pt, flush=True)
            print('These subjets\' all heavy SFs evaluation failed', flush=True)
            print(re, flush=True)
        try:
            BSF_s1_allHeavy = btag_sf['deepCSV_subjet'].evaluate(sysType, 'lt', Fitting, s1_allHeavy, abs(s1_eta), s1_pt)
        except RuntimeError as RE:
            print('flavor (with light mask): \n', s1_allHeavy, flush=True)
            print('eta: \n', s1_eta, flush=True)
            print('pt: \n', s1_pt, flush=True)
            print('These subjets\' all heavy SFs evaluation failed', flush=True)
            print(RE, flush=True)

        BSF_s0_allLight = btag_sf['deepCSV_subjet'].evaluate(sysType, 'incl', Fitting, s0_allLight, abs(s0_eta), s0_pt)
        BSF_s1_allLight = btag_sf['deepCSV_subjet'].evaluate(sysType, 'incl', Fitting, s1_allLight, abs(s1_eta), s1_pt)

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
        
        
#         print("Btag_wgts['0b']", Btag_wgts['0b'])
#         print("len Btag_wgts['0b']", len(Btag_wgts['0b']))
#         print("count Btag_wgts['0b']", ak.count(Btag_wgts['0b']))
#         print("len events", len(btag0))
        
        
#         print("Btag_wgts['1b']", Btag_wgts['1b'])
#         print("len Btag_wgts['1b']", len(Btag_wgts['1b']))
#         print("count Btag_wgts['1b']", ak.count(Btag_wgts['1b']))
#         print("len events", len(btag0))
        
#         print("Btag_wgts['2b']", Btag_wgts['2b'])
#         print("len Btag_wgts['2b']", len(Btag_wgts['2b']))
#         print("count Btag_wgts['2b']", ak.count(Btag_wgts['2b']))
#         print("len events", len(btag0))
        


    # else: # Upgrade or Downgrade btag status based on btag efficiency of all four subjets

        # **************************************************************************************** #
        # --------------------------- Method 2a) Update B-tag Status ----------------------------- #
        # -------------- https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods -------------- #
        # **************************************************************************************** #

                    # ---- Import MC 'flavor' efficiencies ---- #

        # -- Scale Factor File -- #
#         SF_filename = self.ScaleFactorFile  
#         SF_filename = 'data/corrections/subjet_btagging.json.gz'
#         Fitting = "M"
#         if bdisc < 0.5:
#             Fitting = "L"

#         # -- Get Efficiency .csv Files -- #
#         FlavorTagsDict = {
#             5 : 'btag',
#             4 : 'ctag',
#             0 : 'udsgtag'
#         }

#         SubjetNumDict = {
#             'SubJet01' : [SubJet01, 's01'],
#             'SubJet02' : [SubJet02, 's02'],
#             'SubJet11' : [SubJet11, 's11'],
#             'SubJet12' : [SubJet12, 's12']
#         }

#         EffFileDict = {
#             'Eff_File_s01' : [], # List of eff files corresponding to 1st subjet's flavours
#             'Eff_File_s02' : [], # List of eff files corresponding to 2nd subjet's flavours
#             'Eff_File_s11' : [], # List of eff files corresponding to 3rd subjet's flavours
#             'Eff_File_s12' : []  # List of eff files corresponding to 4th subjet's flavours
#         }

#         for subjet,subjet_info in SubjetNumDict.items():
#             flav_tag_list = [FlavorTagsDict[num] for num in np.abs(ak.flatten(subjet_info[0].hadronFlavour))] # List of tags i.e.) ['btag', 'udsgtag', 'ctag',...]
#             for flav_tag in flav_tag_list:
#                 EffFileDict['Eff_File_'+subjet_info[1]].append(self.extraDaskDirectory+'srv/FlavorTagEfficiencies/' 
#                                                                + self.BDirect + flav_tag 
#                                                                + 'EfficiencyTables/' + dataset + '_' + subjet_info[1] 
#                                                                + '_' + flav_tag + 'eff.csv')

#         # -- Does Subjet pass the discriminator cut and is it updated -- #
#         SubJet01_isBtagged = BtagUpdater(SubJet01, EffFileDict['Eff_File_s01'], SF_filename, Fitting, sysType, bdisc)
#         SubJet02_isBtagged = BtagUpdater(SubJet02, EffFileDict['Eff_File_s02'], SF_filename, Fitting, sysType, bdisc)
#         SubJet11_isBtagged = BtagUpdater(SubJet11, EffFileDict['Eff_File_s11'], SF_filename, Fitting, sysType, bdisc)
#         SubJet12_isBtagged = BtagUpdater(SubJet12, EffFileDict['Eff_File_s12'], SF_filename, Fitting, sysType, bdisc)

#         # If either subjet 1 or 2 in FatJet 0 and 1 is btagged after update, then that FatJet is considered btagged #
#         btag_s0 = (SubJet01_isBtagged) | (SubJet02_isBtagged)  
#         btag_s1 = (SubJet11_isBtagged) | (SubJet12_isBtagged)

        # --- Re-Define b-Tag Regions with "Updated" Tags ---- #
#         btag0 = (~btag_s0) & (~btag_s1) #(0b)
#         btag1 = btag_s0 ^ btag_s1 #(1b)
#         btag2 = btag_s0 & btag_s1 #(2b)
        
        
    return Btag_wgts
