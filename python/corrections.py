# corrections.py

import numpy as np
from coffea.lumi_tools import LumiMask

    
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


def GetL1PreFiringWeight(self, events):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L50
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe
    ## var = "Nom", "Up", "Dn"
    L1PrefiringWeights = np.ones(len(events))
    if ("L1PreFiringWeight_Nom" in events.fields):
        L1PrefiringWeights = [events.L1PreFiringWeight_Nom, events.L1PreFiringWeight_Dn, events.L1PreFiringWeight_Up]

    return L1PrefiringWeights


def HEMCleaning(self, JetCollection):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L58

    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    isHEM = ak.ones_like(JetCollection.pt)
    if (self.year == 2018):
        detector_region1 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -1.3) & (JetCollection.eta > -2.5))
        detector_region2 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -2.5) & (JetCollection.eta > -3.0))
        jet_selection    = ((JetCollection.jetId > 1) & (JetCollection.pt > 15))

        isHEM            = ak.where(detector_region1 & jet_selection, 0.80, isHEM)
        isHEM            = ak.where(detector_region2 & jet_selection, 0.65, isHEM)

    return isHEM




def GetJECUncertainties(self, FatJets, events, isData=False):

    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py

    IOV = f'{self.year}{self.apv}'

    jer_tag=None
    if (IOV=='2018'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "RunA": "Summer19UL18_RunA_V5_DATA",
            "RunB": "Summer19UL18_RunB_V5_DATA",
            "RunC": "Summer19UL18_RunC_V5_DATA",
            "RunD": "Summer19UL18_RunD_V5_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        jec_tag="Summer19UL17_V5_MC"
        jec_tag_data={
            "RunB": "Summer19UL17_RunB_V5_DATA",
            "RunC": "Summer19UL17_RunC_V5_DATA",
            "RunD": "Summer19UL17_RunD_V5_DATA",
            "RunE": "Summer19UL17_RunE_V5_DATA",
            "RunF": "Summer19UL17_RunF_V5_DATA",
        }
        jer_tag = "Summer19UL17_JRV2_MC"
    elif (IOV=='2016noAPV'):
        jec_tag="Summer19UL16_V7_MC"
        jec_tag_data={
            "RunF": "Summer19UL16_RunFGH_V7_DATA",
            "RunG": "Summer19UL16_RunFGH_V7_DATA",
            "RunH": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V7_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "RunB_ver1": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunB_ver2": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunC": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunD": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunE": "Summer19UL16APV_RunEF_V7_DATA",
            "RunF": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    else:
        raise ValueError(f"Error: Unknown year \"{IOV}\".")



    ext = extractor()
    if not isData:
    #For MC
        ext.add_weight_sets([
            '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_L1FastJet_AK8PFchs.jec.txt'.format(jec_tag),
            '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_L2Relative_AK8PFchs.jec.txt'.format(jec_tag),
            '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_L3Absolute_AK8PFchs.jec.txt'.format(jec_tag),
            '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_UncertaintySources_AK8PFchs.junc.txt'.format(jec_tag),
            '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_Uncertainty_AK8PFchs.junc.txt'.format(jec_tag),
        ])

        if jer_tag:
            ext.add_weight_sets([
            '* * TTbarAllHadUproot/CorrectionFiles/JER/{0}/{0}_PtResolution_AK4PFchs.jr.txt'.format(jer_tag),
            '* * TTbarAllHadUproot/CorrectionFiles/JER/{0}/{0}_SF_AK4PFchs.jersf.txt'.format(jer_tag)])


    else:       
        #For data, make sure we don't duplicat
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                ext.add_weight_sets([
                '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_L1FastJet_AK8PFchs.jec.txt'.format(tag),
                '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_L2Relative_AK8PFchs.jec.txt'.format(tag),
                '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_L3Absolute_AK8PFchs.jec.txt'.format(tag),
                '* * TTbarAllHadUproot/CorrectionFiles/JEC/{0}/{0}_L2L3Residual_AK8PFchs.jec.txt'.format(tag),
                ])
                tags_done += [tag]

    ext.finalize()





    evaluator = ext.make_evaluator()



    if (not isData):
        jec_names = [
            '{0}_L1FastJet_AK8PFchs'.format(jec_tag),
            '{0}_L2Relative_AK8PFchs'.format(jec_tag),
            '{0}_L3Absolute_AK8PFchs'.format(jec_tag),
            '{0}_Uncertainty_AK8PFchs'.format(jec_tag)]

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_AK4PFchs'.format(jer_tag),
                              '{0}_SF_AK4PFchs'.format(jer_tag)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_AK8PFchs'.format(tag),
                '{0}_L3Absolute_AK8PFchs'.format(tag),
                '{0}_L2Relative_AK8PFchs'.format(tag),
                '{0}_L2L3Residual_AK8PFchs'.format(tag),]



    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_names_data = []
        for era in self.eras:
            jec_names_data += jec_names[f'Run{era}']

        jec_inputs = {name: evaluator[name] for name in jec_names_data}




    jec_stack = JECStack(jec_inputs)


    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]

    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'



    events_cache = events.caches[0]

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(FatJets, lazy_cache=events_cache)

    return corrected_jets


def GetPDFWeights(self, events):
    if "LHEPdfWeight" in events.fields:
            LHEPdfWeight = events.LHEPdfWeight
            pdf_up   = ak.flatten(LHEPdfWeight[2::2])
            pdf_down = ak.flatten(LHEPdfWeight[1::2])
            pdf_nom  = ak.flatten(LHEPdfWeight[0::2])

    else:

        pdf_up = np.ones(len(events))
        pdf_down = np.ones(len(events))
        pdf_nom = np.ones(len(events))            

    return [pdf_up, pdf_down, pdf_nom]



def GetPUSF(self, events):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
    if (self.year == 2016):
        fname = "TTbarAllHadUproot/CorrectionFiles/puWeights/{0}{1}_UL/puWeights.json.gz".format(self.year, self.vfp)
    else:
        fname = "TTbarAllHadUproot/CorrectionFiles/puWeights/{0}_UL/puWeights.json.gz".format(self.year)
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    puUp = evaluator[hname[str(self.year)]].evaluate(np.array(events.Pileup_nTrueInt), "up")
    puDown = evaluator[hname[str(self.year)]].evaluate(np.array(events.Pileup_nTrueInt), "down")
    puNom = evaluator[hname[str(self.year)]].evaluate(np.array(events.Pileup_nTrueInt), "nominal")

    return [puNom, puDown, puUp]


def getLumiMaskRun2(IOV):

    golden_json_path_2016 = "data/corrections/goldenJsons/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
    golden_json_path_2017 = "data/corrections/goldenJsons/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
    golden_json_path_2018 = "data/corrections/goldenJsons/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
    

    masks = {"2016APV":LumiMask(golden_json_path_2016),
             "2016":LumiMask(golden_json_path_2016),
             "2017":LumiMask(golden_json_path_2017),
             "2018":LumiMask(golden_json_path_2018)
            }

    return masks[IOV]


def getMETFilter(IOV, events):
 
    # Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL
    MET_filters = {'2016APV': [
                                "goodVertices",
                                "globalSuperTightHalo2016Filter",
                                "HBHENoiseFilter",
                                "HBHENoiseIsoFilter",
                                "EcalDeadCellTriggerPrimitiveFilter",
                                "BadPFMuonFilter",
                                "BadPFMuonDzFilter",
                                "eeBadScFilter",
                                "hfNoisyHitsFilter"
                               ],
                       '2016'   :["goodVertices",
                                  "globalSuperTightHalo2016Filter",
                                  "HBHENoiseFilter",
                                  "HBHENoiseIsoFilter",
                                  "EcalDeadCellTriggerPrimitiveFilter",
                                  "BadPFMuonFilter",
                                  "BadPFMuonDzFilter",
                                  "eeBadScFilter",
                                  "hfNoisyHitsFilter"],
                       '2017'   :["goodVertices",
                                  "globalSuperTightHalo2016Filter",
                                  "HBHENoiseFilter",
                                  "HBHENoiseIsoFilter",
                                  "EcalDeadCellTriggerPrimitiveFilter",
                                  "BadPFMuonFilter",
                                  "BadPFMuonDzFilter",
                                  "hfNoisyHitsFilter",
                                  "eeBadScFilter",
                                  "ecalBadCalibFilter"],
                       '2018'   :["goodVertices",
                                  "globalSuperTightHalo2016Filter",
                                  "HBHENoiseFilter",
                                  "HBHENoiseIsoFilter",
                                  "EcalDeadCellTriggerPrimitiveFilter",
                                  "BadPFMuonFilter",
                                  "BadPFMuonDzFilter",
                                  "hfNoisyHitsFilter",
                                  "eeBadScFilter",
                                  "ecalBadCalibFilter"]}
    
    metfilter = np.ones(len(events), dtype='bool')
    for flag in MET_filters[IOV]:
            metfilter &= np.array(events.Flag[flag])
            
    return metfilter


def pTReweighting(pt0, pt1):
        topcand0_wgt = np.exp(0.0615 - 0.0005*pt0)
        topcand1_wgt = np.exp(0.0615 - 0.0005*pt1)
        ttbar_wgt = np.sqrt(topcand0_wgt*topcand1_wgt) # used for re-weighting tttbar MC
        
        return ttbar_wgt
