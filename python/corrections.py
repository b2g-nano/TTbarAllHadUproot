from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
import awkward as ak
import os




def GetL1PreFiringWeight(events):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L50
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe
    ## var = "Nom", "Up", "Dn"
    L1PrefiringWeights = np.ones(len(events))
    if ("L1PreFiringWeight_Nom" in events.fields):
        L1PrefiringWeights = [events.L1PreFiringWeight_Nom, events.L1PreFiringWeight_Dn, events.L1PreFiringWeight_Up]

    return L1PrefiringWeights


def HEMCleaning(JetCollection, year):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L58

    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    isHEM = ak.ones_like(JetCollection.pt)
    if (year == 2018):
        detector_region1 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -1.3) & (JetCollection.eta > -2.5))
        detector_region2 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                           (JetCollection.eta < -2.5) & (JetCollection.eta > -3.0))
        jet_selection    = ((JetCollection.jetId > 1) & (JetCollection.pt > 15))

        isHEM            = ak.where(detector_region1 & jet_selection, 0.80, isHEM)
        isHEM            = ak.where(detector_region2 & jet_selection, 0.65, isHEM)

    return isHEM




def GetJECUncertainties(FatJets, events, IOV, isData=False):
    
    # uploadDir = 'srv/' for lpcjobqueue shell or TTbarAllHadUproot/ for coffea casa
    uploadDir = os.getcwd().replace('/','') + '/'
    if 'TTbarAllHadUproot' in uploadDir: 
        uploadDir = 'TTbarAllHadUproot/'
    elif 'jovyan' in uploadDir:
        uploadDir = 'TTbarAllHadUproot/'
    else:
        uploadDir = 'srv/'

    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py
    
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
    elif (IOV=='2016'):
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
            '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_L1FastJet_AK8PFchs.jec.txt'.format(jec_tag),
            '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_L2Relative_AK8PFchs.jec.txt'.format(jec_tag),
            '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_L3Absolute_AK8PFchs.jec.txt'.format(jec_tag),
            '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_UncertaintySources_AK8PFchs.junc.txt'.format(jec_tag),
            '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_Uncertainty_AK8PFchs.junc.txt'.format(jec_tag),
        ])

        if jer_tag:
            ext.add_weight_sets([
            '* * '+uploadDir+'/CorrectionFiles/JER/{0}/{0}_PtResolution_AK4PFchs.jr.txt'.format(jer_tag),
            '* * '+uploadDir+'/CorrectionFiles/JER/{0}/{0}_SF_AK4PFchs.jersf.txt'.format(jer_tag)])


    else:       
        #For data, make sure we don't duplicat
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                ext.add_weight_sets([
                '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_L1FastJet_AK8PFchs.jec.txt'.format(tag),
                '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_L2Relative_AK8PFchs.jec.txt'.format(tag),
                '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_L3Absolute_AK8PFchs.jec.txt'.format(tag),
                '* * '+uploadDir+'/CorrectionFiles/JEC/{0}/{0}_L2L3Residual_AK8PFchs.jec.txt'.format(tag),
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


def GetPDFWeights(events):
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



def GetPUSF(events, year):
    
    # uploadDir = 'srv/' for lpcjobqueue shell or TTbarAllHadUproot/ for coffea casa
    uploadDir = os.getcwd().replace('/','') + '/'
    if 'TTbarAllHadUproot' in uploadDir: 
        uploadDir = 'TTbarAllHadUproot/'
    elif 'jovyan' in uploadDir:
        uploadDir = 'TTbarAllHadUproot/'
    else:
        uploadDir = 'srv/'
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
    if (year == 2016):
        fname = "'+uploadDir+'/CorrectionFiles/puWeights/{0}{1}_UL/puWeights.json.gz".format(year, self.vfp)
    else:
        fname = "'+uploadDir+'/CorrectionFiles/puWeights/{0}_UL/puWeights.json.gz".format(year)
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    puUp = evaluator[hname[str(year)]].evaluate(np.array(events.Pileup_nTrueInt), "up")
    puDown = evaluator[hname[str(year)]].evaluate(np.array(events.Pileup_nTrueInt), "down")
    puNom = evaluator[hname[str(year)]].evaluate(np.array(events.Pileup_nTrueInt), "nominal")

    return [puNom, puDown, puUp]
