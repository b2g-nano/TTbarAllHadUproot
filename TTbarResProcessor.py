#!/usr/bin/env python 
# coding: utf-8

from coffea import processor, nanoevents
from coffea import util
from coffea.btag_tools import BTagScaleFactor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents.methods import vector
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
from collections import defaultdict
import sys
import os, psutil
import copy
import scipy.stats as ss
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState
import correctionlib
import hist
# from correctionlib.schemav2 import Correction
import re as regularexpressions

import awkward as ak

from cms_utils import getLumiMaskRun2


# for dask, `from python.corrections import` does not work
sys.path.append(os.getcwd()+'/python/')

from corrections import ( 
    GetL1PreFiringWeight,
    HEMCleaning,
    GetJECUncertainties,
    GetPDFWeights,
    GetPUSF,
)

from btag_flavor_efficiencies import (
    BtagUpdater,
    GetFlavorEfficiency,
)

from functions import (
    MemoryMb,
    ConvertLabelToInt,
    CartesianProduct,
)





ak.behavior.update(candidate.behavior)
ak.behavior.update(vector.behavior)

# --- Define 'Manual bins' to use for mistag plots for aesthetic purposes--- #
manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]

# --- Define 'Manual pT bins' to use for mc flavor efficiency plots for higher stats per bin--- # 
manual_subjetpt_bins = [0, 300, 600, 1200] # Used on 6/17/22 for ttbar (3 bins)
manual_subjeteta_bins = [0., 0.6, 1.2, 2.4] # Used on 6/17/22 for ttbar (3 bins)
manual_jetht_bins = [200, 800, 840, 880, 920, 960, 1000, 1200, 1400, 1600, 1800, 2000]
manual_sdMass_bins = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

"""Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. """
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., minMSD=105., maxMSD=210.,
                 tau32Cut=0.65, ak8PtMin=400., bdisc=0.5847, deepAK8Cut=0.632, BDirect='',
                 year=None, apv='', vfp='',eras=[], UseLookUpTables=False, lu=None,
                 ModMass=False, RandomDebugMode=False, UseEfficiencies=False, xsSystematicWeight=1., lumSystematicWeight=1.,
                 ApplybtagSF=False, ScaleFactorFile='', ApplyttagSF=False, ApplyTopReweight=False, 
                 ApplyJes=False, ApplyJer=False, var="nominal", ApplyPdf=False, ApplyPrefiring=False, ApplyPUweights=False,
                 ApplyHEMCleaning=False, trigs_to_run=[''], csvv2=False, 
                 sysType=None):

        self.lumimasks = getLumiMaskRun2()
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
        self.eras = eras
        self.trigs_to_run = trigs_to_run
        self.UseLookUpTables = UseLookUpTables
        self.ModMass = ModMass
        self.RandomDebugMode = RandomDebugMode
        self.ScaleFactorFile = ScaleFactorFile
        self.ApplybtagSF = ApplybtagSF # Only apply scale factors when MC efficiencies are being imported in second run of processor
        self.ApplyttagSF = ApplyttagSF
        self.ApplyTopReweight = ApplyTopReweight
        self.ApplyJes = ApplyJes
        self.ApplyJer = ApplyJer
        self.var = var
        self.ApplyPdf = ApplyPdf
        self.ApplyPrefiring = ApplyPrefiring
        self.ApplyPUweights = ApplyPUweights
        self.ApplyHEMCleaning = ApplyHEMCleaning
        self.sysType = sysType # string for btag SF evaluator --> "central", "up", or "down"
        self.UseEfficiencies = UseEfficiencies
        self.xsSystematicWeight = xsSystematicWeight
        self.lumSystematicWeight = lumSystematicWeight
        self.lu = lu # Look Up Tables
        self.means_stddevs = defaultdict() # To remove anomalous MC weights
        self.csvv2 = csvv2
        
        # --- anti-tag+probe, anti-tag, pre-tag, 0, 1, >=1, 2 ttags, any t-tag (>=0t) --- #
        self.ttagcats = ["AT&Pt", "at", "pret", "0t", "1t", ">=1t", "2t", ">=0t"] 
        
        # --- 0, 1, or 2 b-tags --- #
        self.btagcats = ["0b", "1b", "2b"]
        
        # --- Central and forward --- #
        self.ycats = ['cen', 'fwd']
        
        # --- Combine categories like "0bcen", "0bfwd", etc: --- #
        self.anacats = [ t+b+y for t,b,y in itertools.product( self.ttagcats, self.btagcats, self.ycats) ]
        self.label_dict = {i: label for i, label in enumerate(self.anacats)}
        self.label_to_int_dict = {label: i for i, label in enumerate(self.anacats)}

        
        # rewriting axes for scikit-hep/hist   
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary Dataset")

        # --- map analysis categories to array --- #
        cats_axis = hist.axis.IntCategory(range(48), name="anacat", label="Analysis Category")

        # --- axes for jets and ttbar candidates --- #
        ttbarmass_axis = hist.axis.Regular(50, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        jetmass_axis   = hist.axis.Regular(50, 0, 500, name="jetmass", label=r"Jet $m$ [GeV]")
        SDjetmass_axis = hist.axis.Regular(50, 0, 500, name="SDjetmass", label=r"Jet $m_{SD}$ [GeV]")
        jetpt_axis     = hist.axis.Regular(50, 400, 2000, name="jetpt", label=r"Jet $p_{T}$ [GeV]")
        jeteta_axis    = hist.axis.Regular(50, -2.4, 2.4, name="jeteta", label=r"Jet $\eta$")
        jetphi_axis    = hist.axis.Regular(50, -np.pi, np.pi, name="jetphi", label=r"Jet $\phi$")
        jety_axis      = hist.axis.Regular(50, -3, 3, name="jety", label=r"Jet $y$")
        jetdy_axis     = hist.axis.Regular(50, 0, 5, name="jetdy", label=r"Jet $\Delta y$")
        m_pT_axis      = hist.axis.Regular(50, 0, 1.5, name="m_pT", label=r"Jet $m/p_T$")

        # --- axes for top tagger --- #
        manual_axis = hist.axis.Variable(manual_bins, name="jetp", label=r"Jet Momentum [GeV]")
        tagger_axis = hist.axis.Regular(50, 0, 1, name="tagger", label=r"MD deepAK8")
        subjettagger_axis = hist.axis.Regular(50, 0, 1, name="subjettagger", label=r"DeepCSV")
        # tau32_axis  = hist.axis.Regular(50, 0, 2, name="tau32", label=r"$\tau_3/\tau_2$")

        # --- axes for subjets --- #
        subjetmass_axis = hist.axis.Regular(50, 0, 500, name="subjetmass", label=r"SubJet $m$ [GeV]")
        subjetpt_axis   = hist.axis.Regular(50, 400, 2000, name="subjetpt", label=r"SubJet $p_{T}$ [GeV]")
        subjeteta_axis  = hist.axis.Regular(50, -2.4, 2.4, name="subjeteta", label=r"SubJet $\eta$")
        subjetphi_axis  = hist.axis.Regular(50, -np.pi, np.pi, name="subjetphi", label=r"SubJet $\phi$")
        subjetm_pT_axis = hist.axis.Regular(50, 0, 1.5, name="m_pT", label=r"SubJet $m/p_T$")

        # --- axes for weights --- #
        # jethem_axis    = hist.axis.Regular(30, 0, 1.5, name=   "JetWeights", label=r"2018 HEM Weights")
        # fatjethem_axis = hist.axis.Regular(30, 0, 1.5, name="FatJetWeights", label=r"2018 HEM Weights")
        prefiring_axis = hist.axis.Regular(30, 0, 1.5, name=      "Weights", label=r"L1 Prefiring Weight")
            
        self.histo_dict = {
            
        #    ===================================================================================================================
        #    K     K IIIIIII N     N EEEEEEE M     M    A    TTTTTTT IIIIIII   CCCC      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
        #    K   K      I    NN    N E       MM   MM   A A      T       I     C          H     H    I     S         T     S          
        #    K K        I    N N   N E       M M M M  A   A     T       I    C           H     H    I    S          T    S           
        #    KKk        I    N  N  N EEEEEEE M  M  M  AAAAA     T       I    C           HHHHHHH    I     SSSSS     T     SSSSS      
        #    K  K       I    N   N N E       M     M A     A    T       I    C           H     H    I          S    T          S     
        #    K   K      I    N    NN E       M     M A     A    T       I     C          H     H    I         S     T         S      
        #    K   K   IIIIIII N     N EEEEEEE M     M A     A    T    IIIIIII   CCCC      H     H IIIIIII SSSSS      T    SSSSS 
        #    ===================================================================================================================    

            'ttbarmass' : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_bare' : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            
            'ttbarmass_pdfUp'   : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_pdfDown' : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_pdfNom'  : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),

            'ttbarmass_puUp'   : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_puDown' : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_puNom'  : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            
            'ttbarmass_hemUp'   : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_hemDown' : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_hemNom'  : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            
            'ttbarmass_prefiringUp'   : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_prefiringDown' : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            

            #################################################################################################################

            'jetmass' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'jetmass_jesUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_jesDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_jesNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'jetmass_pdfUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_pdfDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_pdfNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'jetmass_puUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_puDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_puNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'jetmass_hemUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_hemDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_hemNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'jetmass_prefiringUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_prefiringDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'jetmass_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            #################################################################################################################
            
            'SDmass'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'SDmass_jesUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_jesDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_jesNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'SDmass_pdfUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_pdfDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_pdfNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'SDmass_puUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_puDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_puNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'SDmass_hemUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_hemDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_hemNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            # 'SDmass_prefiringUp'   : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_prefiringDown' : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            # 'SDmass_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, jetmass_axis, storage="weight", name="Counts"),
            
            #################################################################################################################

            'jetpt'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            
            # 'jetpt_jesUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_jesDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_jesNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            
            # 'jetpt_pdfUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_pdfDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_pdfNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            
            # 'jetpt_puUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_puDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_puNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            
            # 'jetpt_hemUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_hemDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_hemNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            
            # 'jetpt_prefiringUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_prefiringDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'jetpt_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            
            #################################################################################################################
            
            'jeteta' : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            
            # 'jeteta_jesUp'   : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_jesDown' : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_jesNom'  : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            
            # 'jeteta_pdfUp'   : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_pdfDown' : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_pdfNom'  : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            
            # 'jeteta_puUp'   : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_puDown' : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_puNom'  : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            
            # 'jeteta_hemUp'   : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_hemDown' : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_hemNom'  : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            
            # 'jeteta_prefiringUp'   : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_prefiringDown' : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            # 'jeteta_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, jeteta_axis, storage="weight", name="Counts"),
            
            #################################################################################################################
            
            'jetphi' : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            
            # 'jetphi_jesUp'   : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_jesDown' : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_jesNom'  : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            
            # 'jetphi_pdfUp'   : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_pdfDown' : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_pdfNom'  : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            
            # 'jetphi_puUp'   : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_puDown' : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_puNom'  : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            
            # 'jetphi_hemUp'   : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_hemDown' : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_hemNom'  : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            
            # 'jetphi_prefiringUp'   : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_prefiringDown' : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            # 'jetphi_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, jetphi_axis, storage="weight", name="Counts"),
            
            #################################################################################################################

            'probept' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            'probep'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            
            # 'probept_jesUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_jesDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_jesNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probep_jesUp'    : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_jesDown'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_jesNom'   : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            
            # 'probept_pdfUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_pdfDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_pdfNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probep_pdfUp'    : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_pdfDown'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_pdfNom'   : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            
            # 'probept_puUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_puDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_puNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probep_puUp'    : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_puDown'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_puNom'   : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            
            # 'probept_hemUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_hemDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_hemNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probep_hemUp'    : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_hemDown'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_hemNom'   : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            
            # 'probept_prefiringUp'   : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_prefiringDown' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probept_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, jetpt_axis, storage="weight", name="Counts"),
            # 'probep_prefiringUp'    : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_prefiringDown'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            # 'probep_prefiringNom'   : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            
            #################################################################################################################

            'jety'  : hist.Hist(dataset_axis, cats_axis, jety_axis, storage="weight", name="Counts"),
            'jetdy' : hist.Hist(dataset_axis, cats_axis, jetdy_axis, storage="weight", name="Counts"),
        
        #    ===========================================================================================     
        #    TTTTTTT    A    GGGGGGG GGGGGGG EEEEEEE RRRRRR      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
        #       T      A A   G       G       E       R     R     H     H    I     S         T     S          
        #       T     A   A  G       G       E       R     R     H     H    I    S          T    S           
        #       T     AAAAA  G  GGGG G  GGGG EEEEEEE RRRRRR      HHHHHHH    I     SSSSS     T     SSSSS      
        #       T    A     A G     G G     G E       R   R       H     H    I          S    T          S     
        #       T    A     A G     G G     G E       R    R      H     H    I         S     T         S      
        #       T    A     A  GGGGG   GGGGG  EEEEEEE R     R     H     H IIIIIII SSSSS      T    SSSSS 
        #    =========================================================================================== 

            'deepTagMD_TvsQCD' : hist.Hist(dataset_axis, jetpt_axis, SDjetmass_axis, m_pT_axis, tagger_axis, storage="weight", name="Counts"),
            'deepB_subjet'     : hist.Hist(dataset_axis, subjetpt_axis, subjetmass_axis, subjetm_pT_axis, subjettagger_axis, storage="weight", name="Counts"),
            'deepB_fatjet'     : hist.Hist(dataset_axis, jetpt_axis, SDjetmass_axis, m_pT_axis, subjettagger_axis, storage="weight", name="Counts"),
            # 'tau32'            : hist.Hist(dataset_axis, cats_axis, tau32_axis, storage="weight", name="Counts"),

        #    ===========================================================================================    
        #    M     M IIIIIII   SSSSS TTTTTTT    A    GGGGGGG     H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
        #    MM   MM    I     S         T      A A   G           H     H    I     S         T     S          
        #    M M M M    I    S          T     A   A  G           H     H    I    S          T    S           
        #    M  M  M    I     SSSSS     T     AAAAA  G  GGGG     HHHHHHH    I     SSSSS     T     SSSSS      
        #    M     M    I          S    T    A     A G     G     H     H    I          S    T          S     
        #    M     M    I         S     T    A     A G     G     H     H    I         S     T         S      
        #    M     M IIIIIII SSSSS      T    A     A  GGGGG      H     H IIIIIII SSSSS      T    SSSSS 
        #    ===========================================================================================    
            
            'numerator'  : hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
            'denominator': hist.Hist(dataset_axis, cats_axis, manual_axis, storage="weight", name="Counts"),
        
        #    ===========================================================================================    
        #    W     W EEEEEEE IIIIIII GGGGGGG H     H TTTTTTT     H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
        #    W     W E          I    G       H     H    T        H     H    I     S         T     S          
        #    W     W E          I    G       H     H    T        H     H    I    S          T    S           
        #    W  W  W EEEEEEE    I    G  GGGG HHHHHHH    T        HHHHHHH    I     SSSSS     T     SSSSS      
        #    W W W W E          I    G     G H     H    T        H     H    I          S    T          S     
        #    WW   WW E          I    G     G H     H    T        H     H    I         S     T         S      
        #    W     W EEEEEEE IIIIIII  GGGGG  H     H    T        H     H IIIIIII SSSSS      T    SSSSS
        #    ===========================================================================================    
            
            # 'weights_HEM' : hist.Hist(dataset_axis, jethem_axis, fatjethem_axis, storage="weight", name="Counts"),
            
            'weights_prefiringUp'  : hist.Hist(dataset_axis, cats_axis, prefiring_axis, storage="weight", name="Counts"),
            'weights_prefiringDown'  : hist.Hist(dataset_axis, cats_axis, prefiring_axis, storage="weight", name="Counts"),
            'weights_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, prefiring_axis, storage="weight", name="Counts"),
            
            #********************************************************************************************************************#

            'cutflow': processor.defaultdict_accumulator(int),
        }             
            
    
    

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        
        output = self.histo_dict
        
        # ---- Define dataset ---- #
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']

#    ===========================================================================================    
#    N     N    A    N     N   OOO      A      OOO   DDDD          OOO   BBBBBB  JJJJJJJ   SSSSS     
#    NN    N   A A   NN    N  O   O    A A    O   O  D   D        O   O  B     B    J     S          
#    N N   N  A   A  N N   N O     O  A   A  O     O D    D      O     O B     B    J    S           
#    N  N  N  AAAAA  N  N  N O     O  AAAAA  O     O D     D     O     O BBBBBB     J     SSSSS      
#    N   N N A     A N   N N O     O A     A O     O D    D      O     O B     B J  J          S     
#    N    NN A     A N    NN  O   O  A     A  O   O  D   D        O   O  B     B J  J         S      
#    N     N A     A N     N   OOO   A     A   OOO   DDDD          OOO   BBBBBB   JJ     SSSSS 
#    =========================================================================================== 

        isData = ('JetHT' in filename) or ('SingleMu' in filename)
        isSignal = ('RSGluon' in filename) #or ('DM' in filename)
        
        IOV = ('2016APV' if any(regularexpressions.findall(r'preVFP', dataset))
               else '2018' if any(regularexpressions.findall(r'UL18', dataset))
               else '2017' if any(regularexpressions.findall(r'UL17', dataset))
               else '2016')
        
        #blinding
        if isData and (('2017' in IOV) or ('2018' in IOV)):
            events = events[::10]
       
        if "QCD_Pt-15to7000" in filename: 
                events = events[ events.Generator_binvar > 400 ] # Remove events with large weights
        
        # ---- Get event weights from dataset ---- #
        if isData: # If data is used...
            # print('if isData command works')
            evtweights = np.ones( len(events) ) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            if "LHEWeight_originalXWGTUP" not in events.fields: 
                evtweights = events.Generator_weight
            else: 
                evtweights = events.LHEWeight_originalXWGTUP
                
        # ---- Define lumimasks ---- #
        # print(f'\nbefore lumimask:\n {events.nFatJet}')
        if isData: 
            lumi_mask = np.array(self.lumimasks[IOV](events.run, events.luminosityBlock), dtype=bool)
            events = events[lumi_mask]
            evtweights = evtweights[lumi_mask]
        elif isSignal:
            pass # Do nothing to the number of events here...
        else: 
            if dataset not in self.means_stddevs : 
                average = np.average( events.Generator_weight )
                stddev = np.std( events.Generator_weight )
                self.means_stddevs[dataset] = (average, stddev)            
            average,stddev = self.means_stddevs[dataset]
            vals = (events.Generator_weight - average ) / stddev
            events = events[ np.abs(vals) < 2 ]


        
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
            "btagCSVV2": events.FatJet_btagCSVV2,
            "deepTag_TvsQCD": events.FatJet_deepTag_TvsQCD,
            "deepTagMD_TvsQCD": events.FatJet_deepTagMD_TvsQCD,
            "subJetIdx1": events.FatJet_subJetIdx1,
            "subJetIdx2": events.FatJet_subJetIdx2,
            "rawFactor": events.FatJet_rawFactor,
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
            "rawFactor": events.Jet_rawFactor,
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
        
        if not isData:
            # ---- Define GenJets ---- #
            GenJets = ak.zip({
                "run": events.run,
                "pt": events.GenJetAK8_pt,
                "eta": events.GenJetAK8_eta,
                "phi": events.GenJetAK8_phi,
                "mass": events.GenJetAK8_mass,
                "p4": ak.zip({
                    "pt": events.GenJetAK8_pt,
                    "eta": events.GenJetAK8_eta,
                    "phi": events.GenJetAK8_phi,
                    "mass": events.GenJetAK8_mass,
                    }, with_name="PtEtaPhiMLorentzVector"),
                })
            
            Jets['hadronFlavour'] = events.Jet_hadronFlavour
            Jets["genJetIdx"] = events.Jet_genJetIdx
            SubJets['hadronFlavour'] = events.SubJet_hadronFlavour
            FatJets["genJetIdx"] = events.FatJet_genJetAK8Idx

        # ---- Show all events ---- #
        output['cutflow']['all events'] += len(FatJets) #ak.to_awkward0(FatJets).size
        # print("all events", len(FatJets))
        
        # ---- Define the SumW2 for MC Datasets (Probably unnecessary now) ---- #
        output['cutflow']['sumw'] += np.sum(evtweights)
        output['cutflow']['sumw2'] += np.sum(evtweights**2)        
        
        
        # ---- Jet Corrections ---- #
        
        # match gen jets to AK8 jets
        if not isData:
            
            
            FatJets["matched_gen_0p2"] = FatJets.p4.nearest(GenJets.p4, threshold=0.2)
            FatJets["pt_gen"] = ak.values_astype(ak.fill_none(FatJets.matched_gen_0p2.pt, 0), np.float32)
            
            

#    =======================        
#    JJJJJJJ EEEEEEE   SSSSS     
#       J    E        S          
#       J    E       S           
#       J    EEEEEEE  SSSSS      
#    J  J    E             S     
#    J  J    E            S      
#     JJ     EEEEEEE SSSSS
#    =======================
        
        if(self.ApplyJes):
            
            CorrectedJets = GetJECUncertainties(FatJets, events, IOV, isData)
                
            if (self.var == "central"):

                FatJets['pt'] = CorrectedJets['pt']
                FatJets['eta'] = CorrectedJets['eta']
                FatJets['phi'] = CorrectedJets['phi']
                FatJets['mass'] = CorrectedJets['mass']

                del CorrectedJets
                
            else:
                if not isData:

                    if (self.var == "up"):
                        FatJets['pt'] = CorrectedJets.JES_jes.up.pt
                        FatJets['eta'] = CorrectedJets.JES_jes.up.eta
                        FatJets['phi'] = CorrectedJets.JES_jes.up.phi
                        FatJets['mass'] = CorrectedJets.JES_jes.up.mass
                        
                    elif (self.var == "down"):
                        FatJets['pt'] = CorrectedJets.JES_jes.down.pt
                        FatJets['eta'] = CorrectedJets.JES_jes.down.eta
                        FatJets['phi'] = CorrectedJets.JES_jes.down.phi
                        FatJets['mass'] = CorrectedJets.JES_jes.down.mass
                    
                    del CorrectedJets
        
                    
                    
        if(self.ApplyJer):
            
            if not isData:
                
                CorrectedJets = GetJECUncertainties(FatJets, events, IOV, isData)
                
                if (self.var == "central"):

                    FatJets['pt'] = CorrectedJets['pt']
                    FatJets['eta'] = CorrectedJets['eta']
                    FatJets['phi'] = CorrectedJets['phi']
                    FatJets['mass'] = CorrectedJets['mass']

                    del CorrectedJets

                else:
                    if not isData:

                        if (self.var == "up"):
                            FatJets['pt'] = CorrectedJets.JER.up.pt
                            FatJets['eta'] = CorrectedJets.JER.up.eta
                            FatJets['phi'] = CorrectedJets.JER.up.phi
                            FatJets['mass'] = CorrectedJets.JER.up.mass

                        elif (self.var == "down"):
                            FatJets['pt'] = CorrectedJets.JER.down.pt
                            FatJets['eta'] = CorrectedJets.JER.down.eta
                            FatJets['phi'] = CorrectedJets.JER.down.phi
                            FatJets['mass'] = CorrectedJets.JER.down.mass

                        del CorrectedJets
                    
                    
            
            
            
            
            
                    
                    
                    
                    

            
#    ===========================================================================================            
#    H     H EEEEEEE M     M       CCCC  L       EEEEEEE    A    N     N IIIIIII N     N GGGGGGG     
#    H     H E       MM   MM      C      L       E         A A   NN    N    I    NN    N G           
#    H     H E       M M M M     C       L       E        A   A  N N   N    I    N N   N G           
#    HHHHHHH EEEEEEE M  M  M     C       L       EEEEEEE  AAAAA  N  N  N    I    N  N  N G  GGGG     
#    H     H E       M     M     C       L       E       A     A N   N N    I    N   N N G     G     
#    H     H E       M     M      C      L       E       A     A N    NN    I    N    NN G     G     
#    H     H EEEEEEE M     M       CCCC  LLLLLLL EEEEEEE A     A N     N IIIIIII N     N  GGGGG 
#    ===========================================================================================

        if self.ApplyHEMCleaning and not isData:
                    
            JetWeights_HEM = HEMCleaning(Jets, self.year)
            FatJetWeights_HEM = HEMCleaning(FatJets, self.year)
            
            # output['weights_HEM'].fill(dataset = dataset,
            #                  JetWeights = JetWeights_HEM,
            #                  FatJetWeights = FatJetWeights_HEM)

            Jets = ak.with_field(Jets, JetWeights_HEM*Jets.pt, 'pt')
            FatJets = ak.with_field(FatJets, FatJetWeights_HEM*FatJets.pt, 'pt')

            del JetWeights_HEM, FatJetWeights_HEM

#    ===================================================================================            
#    M     M EEEEEEE TTTTTTT     FFFFFFF IIIIIII L       TTTTTTT EEEEEEE RRRRRR    SSSSS     
#    MM   MM E          T        F          I    L          T    E       R     R  S          
#    M M M M E          T        F          I    L          T    E       R     R S           
#    M  M  M EEEEEEE    T        FFFFFFF    I    L          T    EEEEEEE RRRRRR   SSSSS      
#    M     M E          T        F          I    L          T    E       R   R         S     
#    M     M E          T        F          I    L          T    E       R    R       S      
#    M     M EEEEEEE    T        F       IIIIIII LLLLLLL    T    EEEEEEE R     R SSSSS 
#    ===================================================================================            
        
    # ---- Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#2018_2017_data_and_MC_UL ---- #
    
        MET_filters = {'2016APV':["goodVertices",
                                  "globalSuperTightHalo2016Filter",
                                  "HBHENoiseFilter",
                                  "HBHENoiseIsoFilter",
                                  "EcalDeadCellTriggerPrimitiveFilter",
                                  "BadPFMuonFilter",
                                  "BadPFMuonDzFilter",
                                  "eeBadScFilter",
                                  "hfNoisyHitsFilter"],
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
        if isData:
            filteredEvents = np.array([getattr(events, f'Flag_{MET_filters[IOV][i]}') for i in range(len(MET_filters[IOV]))])
            filteredEvents = np.logical_or.reduce(filteredEvents, axis=0)
        
            if ak.sum(filteredEvents) < 1 :
                print("\nNo events passed the MET filters.\n", flush=True)
                return output
            else:
                FatJets = FatJets[filteredEvents]
                Jets = Jets[filteredEvents]
                SubJets = SubJets[filteredEvents]
                evtweights = evtweights[filteredEvents]
                events = events[filteredEvents]

                output['cutflow']['Passed MET Filters'] += ak.sum(filteredEvents)
                # print("Passed MET Filters", ak.sum(filteredEvents))
                # print(len(FatJets))

            del filteredEvents


#    ================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR    SSSSS     
#       T    R     R    I    G       G       E       R     R  S          
#       T    R     R    I    G       G       E       R     R S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR   SSSSS      
#       T    R   R      I    G     G G     G E       R   R         S     
#       T    R    R     I    G     G G     G E       R    R       S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R SSSSS  
#    ================================================================

        condition = None
        Triggers = []
        
        if isData: 
            
            ### 2016 triggers : "HLT_PFHT800", "HLT_PFHT900", "HLT_AK8PFJet450", "HLT_AK8PFJet360_TrimMass30"
            
            for itrig in self.trigs_to_run: 
                thetrig = getattr( events, itrig )
                Triggers.append(thetrig)
                
            condition = ak.flatten(ak.any(Triggers, axis=0, keepdims=True))
            # print(condition)
                        
        #if isData:
            FatJets = FatJets[condition]
            Jets = Jets[condition]
            SubJets = SubJets[condition]
            evtweights = evtweights[condition]
            events = events[condition]
            
            output['cutflow']['Passed Trigger(s)'] += ak.sum(condition)
            # print("Passed Trigger(s)", ak.sum(condition))
            # print(len(FatJets))

        del condition, Triggers
        
            
#    ===================================================================================
#    PPPPPP  RRRRRR  EEEEEEE L       IIIIIII M     M       CCCC  U     U TTTTTTT   SSSSS     
#    P     P R     R E       L          I    MM   MM      C      U     U    T     S          
#    P     P R     R E       L          I    M M M M     C       U     U    T    S           
#    PPPPPP  RRRRRR  EEEEEEE L          I    M  M  M     C       U     U    T     SSSSS      
#    P       R   R   E       L          I    M     M     C       U     U    T          S     
#    P       R    R  E       L          I    M     M      C       U   U     T         S      
#    P       R     R EEEEEEE LLLLLLL IIIIIII M     M       CCCC    UUU      T    SSSSS
#    ===================================================================================

        # ---- Apply HT Cut ---- #
        # ---- This gives the analysis 99.8% efficiency (see 2016 AN) ---- #
        hT = ak.sum(Jets.pt, axis=1)
        passhT = (hT > self.htCut)
        FatJets = FatJets[passhT]
        Jets = Jets[passhT]
        SubJets = SubJets[passhT]
        evtweights = evtweights[passhT]
        events = events[passhT]
        if not isData:
            GenJets = GenJets[passhT]
        
        output['cutflow']['Passed HT Cut'] += ak.sum(passhT)
        # print("Passed HT Cut", ak.sum(passhT))
        # print(len(FatJets))

        del hT, passhT
          
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['Passed Loose Jet ID'] += len(FatJets)

        del jet_id
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['Passed pT,y Cut'] += len(FatJets)

        del FatJets_rapidity, jetkincut_index
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) >= 2)
        FatJets = FatJets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin]
        events = events[twoFatJetsKin]
        evtweights = evtweights[twoFatJetsKin]
        if not isData:
            GenJets = GenJets[twoFatJetsKin]

        del twoFatJetsKin
        
        # print("before randomization")
        # print(len(FatJets))

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

        # print("after randomization")
        # print("FatJets", len(FatJets))
        # print("jet0", len(jet0))
        # print("jet1", len(jet1))
        
        ttbarcands = ak.cartesian([jet0, jet1]) # Re-group the randomized pairs in a similar fashion to how they were

        # print("ttbarcands ", len(ttbarcands))
        # print("ttbar mass ", ttbarcands.slot0.p4.add( ttbarcands.slot1.p4 ).mass)

        del Counts, index
        # ttbarcand_size = len(ttbarcands)
        # FatJets_size = len(FatJets)
        
#         print(f'Size of ttbarcands = {ttbarcand_size}', flush=True)
#         print(f'Size of fatjets = {FatJets_size}', flush=True)
#         print(f'ttbarcands slot 0 pt = {ttbarcands.slot0.pt}', flush=True)
#         print(f'ttbarcands slot 1 pt = {ttbarcands.slot1.pt}', flush=True)
#         print(f'fatjets slot 0 pt = {FatJets.pt[:,0]}', flush=True)
#         print(f'fatjets slot 1 pt = {FatJets.pt[:,1]}', flush=True)
#         print('==============================================\n', flush=True)
        
        """ NOTE that ak.cartesian gives a shape with one more layer than FatJets """
        # ---- Make sure we have at least 1 TTbar candidate pair and re-broadcast releveant arrays  ---- #
        oneTTbar = (ak.num(ttbarcands, axis=-1) >= 1)
        ttbarcands = ttbarcands[oneTTbar]
        FatJets = FatJets[oneTTbar]
        Jets = Jets[oneTTbar]
        SubJets = SubJets[oneTTbar]
        events = events[oneTTbar]
        evtweights = evtweights[oneTTbar]
        if not isData:
            GenJets = GenJets[oneTTbar]
        output['cutflow']['>= oneTTbar'] += len(FatJets)
            
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        # print(np.abs(ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4)))
        dPhiCut = np.abs(ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4)) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        ttbarcands = ttbarcands[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        Jets = Jets[dPhiCut]
        SubJets = SubJets[dPhiCut] 
        events = events[dPhiCut]
        evtweights = evtweights[dPhiCut]
        if not isData:
            GenJets = GenJets[dPhiCut]
        output['cutflow']['Passed dPhi Cut'] += len(FatJets)
                
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets

        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        FatJets = FatJets[GoodSubjets]
        SubJets = SubJets[GoodSubjets]
        events = events[GoodSubjets]
        Jets = Jets[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
        if not isData:
            GenJets = GenJets[GoodSubjets]
        output['cutflow']['Good Subjets'] += len(FatJets)
        
        SubJet01 = SubJets[ttbarcands.slot0.subJetIdx1] # ttbarcandidate 0's first subjet 
        SubJet02 = SubJets[ttbarcands.slot0.subJetIdx2] # ttbarcandidate 0's second subjet
        SubJet11 = SubJets[ttbarcands.slot1.subJetIdx1] # ttbarcandidate 1's first subjet 
        SubJet12 = SubJets[ttbarcands.slot1.subJetIdx2] # ttbarcandidate 1's second subjet
        
        # print(f'Dataset = {dataset}\n***************************************************************\n', flush=True)
        # print(f'Jet 0\'s first subjet\'s ID = {ttbarcands.slot0.subJetIdx1}', flush=True)
        # print(f'Jet 0\'s second subjet\'s ID = {ttbarcands.slot0.subJetIdx2}', flush=True)
        # print(f'Jet 1\'s first subjet\'s ID = {ttbarcands.slot1.subJetIdx1}', flush=True)
        # print(f'Jet 1\'s second subjet\'s ID = {ttbarcands.slot1.subJetIdx2}\n-----------------------------------------\n', flush=True)
        
        # 'deepB' : hist.Hist(dataset_axis, subjetmass_axis, subjetpt_axis, subjeteta_axis, subjetphi_axis, subjettagger_axis, storage="weight", name="Counts"),
        
        # output["deepb"].fill(dataset = dataset,
        #                     subjetmass = ak.to_numpy(ak.flatten(SubJet01.mass)),
        #                     subjetpt = ak.to_numpy(ak.flatten(SubJet01.pt)),
        #                     subjettagger = ak.to_numpy(ak.flatten(SubJet01.btagDeepB)),
        #                     weight = ak.to_numpy(evtweights))
        
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

        # del s0_energy, s1_energy, s0_pz, s1_pz
        

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
#         tau32_s0 = np.where(ttbarcands.slot0.tau2>0,ttbarcands.slot0.tau3/ttbarcands.slot0.tau2, 0 )
#         tau32_s1 = np.where(ttbarcands.slot1.tau2>0,ttbarcands.slot1.tau3/ttbarcands.slot1.tau2, 0 )
        
#         taucut_s0 = tau32_s0 < self.tau32Cut
#         taucut_s1 = tau32_s1 < self.tau32Cut
        
#         mcut_s0 = (self.minMSD < ttbarcands.slot0.msoftdrop) & (ttbarcands.slot0.msoftdrop < self.maxMSD) 
#         mcut_s1 = (self.minMSD < ttbarcands.slot1.msoftdrop) & (ttbarcands.slot1.msoftdrop < self.maxMSD) 

#         ttag_s0 = (taucut_s0) & (mcut_s0)
#         ttag_s1 = (taucut_s1) & (mcut_s1)
#         antitag = (~taucut_s0) & (mcut_s0) # The Probe jet will always be ttbarcands.slot1 (at)

        # ----------- DeepAK8 Tagger (Discriminator Cut) ----------- #
        ttag_s0 = ttbarcands.slot0.deepTagMD_TvsQCD > self.deepAK8Cut
        ttag_s1 = ttbarcands.slot1.deepTagMD_TvsQCD > self.deepAK8Cut
        antitag = ttbarcands.slot0.deepTagMD_TvsQCD < self.deepAK8Cut # The Probe jet will always be ttbarcands.slot1 (at)
        
        # ---- Define "Top Tag" Regions ---- #
        antitag_probe = np.logical_and(antitag, ttag_s1) # Found an antitag and ttagged probe pair for mistag rate (AT&Pt)
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
        if not self.csvv2:
            btag_s0 = ( np.maximum(SubJet01.btagDeepB , SubJet02.btagDeepB) > self.bdisc )
            btag_s1 = ( np.maximum(SubJet11.btagDeepB , SubJet12.btagDeepB) > self.bdisc )
        else: # Only when running -med2016 option
            btag_s0 = ( np.maximum(SubJet01.btagCSVV2 , SubJet02.btagCSVV2) > self.bdisc )
            btag_s1 = ( np.maximum(SubJet11.btagCSVV2 , SubJet12.btagCSVV2) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_s0) & (~btag_s1) #(0b)
        btag1 = btag_s0 ^ btag_s1 #(1b)
        btag2 = btag_s0 & btag_s1 #(2b)

#    ===========================================================================================================================        
#    BBBBBB  TTTTTTT    A    GGGGGGG       CCCC    OOO   RRRRRR  RRRRRR  EEEEEEE   CCCC  TTTTTTT IIIIIII   OOO   N     N   SSSSS     
#    B     B    T      A A   G            C       O   O  R     R R     R E        C         T       I     O   O  NN    N  S          
#    B     B    T     A   A  G           C       O     O R     R R     R E       C          T       I    O     O N N   N S           
#    BBBBBB     T     AAAAA  G  GGGG     C       O     O RRRRRR  RRRRRR  EEEEEEE C          T       I    O     O N  N  N  SSSSS      
#    B     B    T    A     A G     G     C       O     O R   R   R   R   E       C          T       I    O     O N   N N       S     
#    B     B    T    A     A G     G      C       O   O  R    R  R    R  E        C         T       I     O   O  N    NN      S      
#    BBBBBB     T    A     A  GGGGG        CCCC    OOO   R     R R     R EEEEEEE   CCCC     T    IIIIIII   OOO   N     N SSSSS
#    ===========================================================================================================================
        
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
                    if not self.csvv2:
                        LeadingSubjet_s0 = np.where(SubJet01.btagCSVV2>SubJet02.btagCSVV2, SubJet01, SubJet02)
                        LeadingSubjet_s1 = np.where(SubJet11.btagCSVV2>SubJet12.btagCSVV2, SubJet11, SubJet12)
                    else:
                        LeadingSubjet_s0 = np.where(SubJet01.btagDeepB>SubJet02.btagDeepB, SubJet01, SubJet02)
                        LeadingSubjet_s1 = np.where(SubJet11.btagDeepB>SubJet12.btagDeepB, SubJet11, SubJet12)

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
                        print('flavor (with light mask): \n', s0_allHeavy, flush=True)
                        print('eta: \n', s0_eta, flush=True)
                        print('pt: \n', s0_pt, flush=True)
                        print('These subjets\' all heavy SFs evaluation failed', flush=True)
                        print(re, flush=True)
                    try:
                        BSF_s1_allHeavy = btag_sf['deepCSV_subjet'].evaluate(self.sysType, 'lt', Fitting, s1_allHeavy, abs(s1_eta), s1_pt)
                    except RuntimeError as RE:
                        print('flavor (with light mask): \n', s1_allHeavy, flush=True)
                        print('eta: \n', s1_eta, flush=True)
                        print('pt: \n', s1_pt, flush=True)
                        print('These subjets\' all heavy SFs evaluation failed', flush=True)
                        print(RE, flush=True)
                    
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
                            EffFileDict['Eff_File_'+subjet_info[1]].append('FlavorTagEfficiencies/' 
                                                                           + self.BDirect + flav_tag 
                                                                           + 'EfficiencyTables/' + dataset + '_' + subjet_info[1] 
                                                                           + '_' + flav_tag + 'eff.csv')
                            
                    # -- Does Subjet pass the discriminator cut and is it updated -- #
                    SubJet01_isBtagged = BtagUpdater(SubJet01, EffFileDict['Eff_File_s01'], SF_filename, Fitting, self.sysType, self.bdisc)
                    SubJet02_isBtagged = BtagUpdater(SubJet02, EffFileDict['Eff_File_s02'], SF_filename, Fitting, self.sysType, self.bdisc)
                    SubJet11_isBtagged = BtagUpdater(SubJet11, EffFileDict['Eff_File_s11'], SF_filename, Fitting, self.sysType, self.bdisc)
                    SubJet12_isBtagged = BtagUpdater(SubJet12, EffFileDict['Eff_File_s12'], SF_filename, Fitting, self.sysType, self.bdisc)

                    # If either subjet 1 or 2 in FatJet 0 and 1 is btagged after update, then that FatJet is considered btagged #
                    btag_s0 = (SubJet01_isBtagged) | (SubJet02_isBtagged)  
                    btag_s1 = (SubJet11_isBtagged) | (SubJet12_isBtagged)

                    # --- Re-Define b-Tag Regions with "Updated" Tags ---- #
                    btag0 = (~btag_s0) & (~btag_s1) #(0b)
                    btag1 = btag_s0 ^ btag_s1 #(1b)
                    btag2 = btag_s0 & btag_s1 #(2b)
                    
#    ===================================================================================================
#       A    N     N    A    L       Y     Y   SSSSS IIIIIII   SSSSS       CCCC     A    TTTTTTT   SSSSS     
#      A A   NN    N   A A   L        Y   Y   S         I     S           C        A A      T     S          
#     A   A  N N   N  A   A  L         Y Y   S          I    S           C        A   A     T    S           
#     AAAAA  N  N  N  AAAAA  L          Y     SSSSS     I     SSSSS      C        AAAAA     T     SSSSS      
#    A     A N   N N A     A L          Y          S    I          S     C       A     A    T          S     
#    A     A N    NN A     A L          Y         S     I         S       C      A     A    T         S      
#    A     A N     N A     A LLLLLLL    Y    SSSSS   IIIIIII SSSSS         CCCC  A     A    T    SSSSS 
#    ===================================================================================================

        # ---- Get Analysis Categories ---- # 
        # ---- They are (central, forward) cross (0b,1b,2b) cross (Probet,at,pret,0t,1t,>=1t,2t,>=0t) ---- #
        regs = [cen,fwd]
        btags = [btag0,btag1,btag2]
        ttags = [antitag_probe,antitag,pretag,ttag0,ttag1,ttagI,ttag2,Alltags]
        cats = [ ak.to_awkward0(ak.flatten(t&b&y)) for t,b,y in itertools.product(ttags, btags, regs) ]
        labels_and_categories = dict(zip( self.anacats, cats ))
        # labels_and_categories = dict(zip(self.label_dict.keys(), cats))
        #print(labels_and_categories)

        del regs, btags, ttags
        
        # ---- Variables for Kinematic Histograms ---- #
        # ---- "slot0" is the control jet, "slot1" is the probe jet ---- #
        jetpt = ak.flatten(ttbarcands.slot1.pt)
        jeteta = ak.flatten(ttbarcands.slot1.eta)
        jetphi = ak.flatten(ttbarcands.slot1.phi)
        jetmass = ak.flatten(ttbarcands.slot1.mass)
        
        SDmass = ak.flatten(ttbarcands.slot1.msoftdrop)
        m_over_pT = SDmass / jetpt
        # Tau32 = ak.flatten((ttbarcands.slot1.tau3/ttbarcands.slot1.tau2))
        ak8tagger = ak.flatten(ttbarcands.slot1.deepTagMD_TvsQCD)

        """ Add 4-vectors and get its total mass """
        ttbarp4sum = ttbarcands.slot0.p4.add(ttbarcands.slot1.p4)
        ttbarmass = ak.flatten(ttbarp4sum.mass)
        
        """ Use previously defined definitions for rapidity (until/unless better method is found) """
        jety = ak.flatten(ttbarcands_s0_rapidity)
        jetdy = np.abs(ak.flatten(ttbarcands_s0_rapidity) - ak.flatten(ttbarcands_s1_rapidity))
        
        # ---- Weights ---- #
        weights = evtweights*self.xsSystematicWeight*self.lumSystematicWeight
        
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
        
        # print(f'antitag and t-tagged probe:\n{antitag_probe}')
        # print(f'antitag and all probes:\n{antitag}')
        
        numerator = ak.flatten(numerator)
        denominator = ak.flatten(denominator)
        
        df = pd.DataFrame({"momentum":p}) # DataFrame used for finding values in LookUp Tables
        file_df = None # Initial Declaration
        
        # print(self.lu[])
        
        letter = ''
        for i in range(len(self.eras)):
            letter = self.eras[i]
            if 'JetHT'+letter in dataset:
                # print(f'letter {letter} found', flush=True)
                continue

        # ---- Fill Outputs for Taggers ---- #
            
        output['deepTagMD_TvsQCD'].fill(dataset = dataset,
                                     jetpt = ak.to_numpy(jetpt),
                                     SDjetmass = ak.to_numpy(SDmass),
                                     m_pT = ak.to_numpy(m_over_pT),
                                     tagger = ak.to_numpy(ak8tagger),       
                                     weight = ak.to_numpy(weights),
                                    )
        output['deepB_subjet'].fill(dataset = dataset,
                                     subjetpt = ak.to_numpy(ak.flatten(SubJet11.p4.pt)),
                                     subjetmass = ak.to_numpy(ak.flatten(SubJet11.p4.mass)),
                                     m_pT = ak.to_numpy(ak.flatten(SubJet11.p4.mass) / ak.flatten(SubJet11.p4.mass)),
                                     subjettagger = ak.to_numpy(ak.flatten(SubJet11.btagDeepB)),       
                                     weight = ak.to_numpy(weights),
                                    )
        output['deepB_fatjet'].fill(dataset = dataset,
                                     jetpt = ak.to_numpy(jetpt),
                                     SDjetmass = ak.to_numpy(SDmass),
                                     m_pT = ak.to_numpy(m_over_pT),
                                     subjettagger = ak.to_numpy(ak.flatten(SubJet11.btagDeepB)),       
                                     weight = ak.to_numpy(weights),
                                    )
        
        #print("labels_and_categories len ", len(labels_and_categories))
        for i, [ilabel,icat] in enumerate(labels_and_categories.items()):

            #print("Running ", ilabel)
            #print("n icat ", len(icat))
#    ===============================================================================================================================            
#       A    PPPPPP  PPPPPP  L       Y     Y     M     M IIIIIII   SSSSS TTTTTTT    A    GGGGGGG     W     W GGGGGGG TTTTTTT   SSSSS     
#      A A   P     P P     P L        Y   Y      MM   MM    I     S         T      A A   G           W     W G          T     S          
#     A   A  P     P P     P L         Y Y       M M M M    I    S          T     A   A  G           W     W G          T    S           
#     AAAAA  PPPPPP  PPPPPP  L          Y        M  M  M    I     SSSSS     T     AAAAA  G  GGGG     W  W  W G  GGGG    T     SSSSS      
#    A     A P       P       L          Y        M     M    I          S    T    A     A G     G     W W W W G     G    T          S     
#    A     A P       P       L          Y        M     M    I         S     T    A     A G     G     WW   WW G     G    T         S      
#    A     A P       P       LLLLLLL    Y        M     M IIIIIII SSSSS      T    A     A  GGGGG      W     W  GGGGG     T    SSSSS 
#    ===============================================================================================================================
            
            ###------------------------------------------------------------------------------------------###
            ### ------------------------------------ Mistag Scaling ------------------------------------ ###
            ###------------------------------------------------------------------------------------------###
            
            if self.UseLookUpTables == True and (isData or ('TTbar' in dataset)): #No need to apply to signals
                # ---- Weight dataset by mistag from data (corresponding to its year) ---- #
                # ---- Pick out proper JetHT year mistag for TTbar sim. ---- #
                
                if self.year > 0: # this UL string should only appear in MC dataset name when year is either 2016, 2017 or 2018
                    file_df = self.lu['JetHT'+str(self.year)+letter+'_Data']['at' + str(ilabel[-5:])] # Only the corresponding JetHT year mistag rate
                elif self.year == 0: # all years; not just 2016, 17 or 18 alone
                    file_df = self.lu['JetHT_Data']['at' + str(ilabel[-5:])] # All JetHT years mistag rate
                else:
                    print('Something is wrong...\n\nNecessary JetHT LUT(s) not found', flush=True)
                    quit()
               
                # with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
                #     print(file_df)
                
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
                        print(df, flush=True)
                    print(VE, flush=True)
            
                WeightMatching = wgts[BinNumber] # Match 'wgts' with corresponding p bin using the bin number
                Weights = weights*WeightMatching # Include 'wgts' with the previously defined 'weights'
            else:
                Weights = weights # No mistag rate changes to weights

#    =======================================================================================================================================            
#    M     M    A      SSSSS   SSSSS     M     M   OOO   DDDD        PPPPPP  RRRRRR    OOO     CCCC  EEEEEEE DDDD    U     U RRRRRR  EEEEEEE     
#    MM   MM   A A    S       S          MM   MM  O   O  D   D       P     P R     R  O   O   C      E       D   D   U     U R     R E           
#    M M M M  A   A  S       S           M M M M O     O D    D      P     P R     R O     O C       E       D    D  U     U R     R E           
#    M  M  M  AAAAA   SSSSS   SSSSS      M  M  M O     O D     D     PPPPPP  RRRRRR  O     O C       EEEEEEE D     D U     U RRRRRR  EEEEEEE     
#    M     M A     A       S       S     M     M O     O D    D      P       R   R   O     O C       E       D    D  U     U R   R   E           
#    M     M A     A      S       S      M     M  O   O  D   D       P       R    R   O   O   C      E       D   D    U   U  R    R  E           
#    M     M A     A SSSSS   SSSSS       M     M   OOO   DDDD        P       R     R   OOO     CCCC  EEEEEEE DDDD      UUU   R     R EEEEEEE
#    =======================================================================================================================================
            
    
    
            ###---------------------------------------------------------------------------------------------###
            ### ----------------------------------- Mod-mass Procedure ------------------------------------ ###
            ###---------------------------------------------------------------------------------------------###
            if (self.ModMass == True and (isData or ('TTbar' in dataset))):
                QCD_hist = None # Higher scope declaration
                if self.BDirect == '':
                    od = '_oldANdisc'
                else:
                    od = ''
                
                QCD_unweighted = util.load('CoffeaOutputsForCombine/Coffea_FirstRun/QCD/'
                +self.BDirect+str(self.year)+'/'+self.apv+'/TTbarRes_0l_UL'+str(self.year-2000)+self.vfp+'_QCD'+od+'.coffea') 
                    
                # ---- Define Histogram ---- #
                loaded_dataset = 'UL'+str(self.year-2000)+self.vfp+'_QCD'
                    
                self.label_to_int_dict
                
                QCD_hist = QCD_unweighted['jetmass'][loaded_dataset, self.label_to_int_dict['2t' + str(ilabel[-5:])], :]
                # QCD_hist = QCD_unweighted['jetmass'][loaded_dataset, ConvertLabelToInt(self.label_dict, '2t' + str(ilabel[-5:])), :]
                    
                # ---- Extract event counts from QCD MC hist in signal region ---- #
                # data = QCD_hist.values() # Dictionary of values
                QCD_data = QCD_hist.view().value

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
                
#    ===================================================================                
#    BBBBBB  TTTTTTT    A    GGGGGGG     W     W GGGGGGG TTTTTTT   SSSSS     
#    B     B    T      A A   G           W     W G          T     S          
#    B     B    T     A   A  G           W     W G          T    S           
#    BBBBBB     T     AAAAA  G  GGGG     W  W  W G  GGGG    T     SSSSS      
#    B     B    T    A     A G     G     W W W W G     G    T          S     
#    B     B    T    A     A G     G     WW   WW G     G    T         S      
#    BBBBBB     T    A     A  GGGGG      W     W  GGGGG     T    SSSSS 
#    ===================================================================
                
            ###---------------------------------------------------------------------------------------------###
            ### ------------------------------ B-Tag Weighting (S.F. Only) -------------------------------- ###
            ###---------------------------------------------------------------------------------------------###
            if not isData:
                if (self.ApplybtagSF == True) and (self.UseEfficiencies == False):
                    Weights = Weights*Btag_wgts[str(ilabel[-5:-3])]


                if self.ApplyPrefiring:
                    
                    prefiringNom, prefiringDown, prefiringUp = GetL1PreFiringWeight(events)

                    Weights_prefiringUp = Weights * prefiringUp
                    Weights_prefiringDown = Weights * prefiringDown
                    Weights_prefiringNom = Weights * prefiringNom

                    output['ttbarmass_prefiringNom'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_prefiringNom[icat]),
                                    )
                    output['ttbarmass_prefiringUp'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_prefiringUp[icat]),
                                    )
                    output['ttbarmass_prefiringDown'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_prefiringDown[icat]),
                                    )

                    output['weights_prefiringNom'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     Weights = prefiringNom,
                                    )
                    output['weights_prefiringUp'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     Weights = prefiringUp,
                                    )
                    output['weights_prefiringDown'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     Weights = prefiringDown,
                                    )
                    
                
                    
                if self.ApplyPdf:
                    
                    pdfUp, pdfDown, pdfNom = self.GetPDFWeights(events)
                     
                    if len(pdfUp > 0):
                        Weights_pdfUp   = Weights * pdfUp
                        Weights_pdfDown = Weights * pdfDown
                        Weights_pdfNom  = Weights * pdfNom
                        
                    else:
                        Weights_pdfUp   = Weights
                        Weights_pdfDown = Weights
                        Weights_pdfNom  = Weights
                    
                    
                        
                    output['ttbarmass_pdfNom'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_pdfNom[icat]),
                                    )
                    output['ttbarmass_pdfUp'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_pdfUp[icat]),
                                    )
                    output['ttbarmass_pdfDown'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_pdfDown[icat]),
                                    )
                    
                if self.ApplyPUweights:
                    
                    puNom, puDown, puUp = self.GetPUSF(events, self.year)

                    Weights_puUp = Weights * puUp
                    Weights_puDown = Weights * puDown
                    Weights_puNom = Weights * puNom

                    output['ttbarmass_puNom'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_puNom[icat]),
                                    )
                    output['ttbarmass_puUp'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_puUp[icat]),
                                    )
                    output['ttbarmass_puDown'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_puDown[icat]),
                                    )
                    
                    
                        
                    
                                    
            ###---------------------------------------------------------------------------------------------###
            ### ----------------------- Top pT Reweighting (S.F. as function of pT) ----------------------- ###
            ###---------------------------------------------------------------------------------------------###
            if ('TTbar' in dataset) and (self.ApplyTopReweight):
                Weights = Weights*ttbar_wgt

# ************************************************************************************************************ #    

            output['cutflow'][ilabel] += np.sum(icat)
                
            output['ttbarmass'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )


            #badweights = Weights[ (self.label_to_int_dict[ilabel] == 36) & (ttbarmass > 1100) & (ttbarmass < 1800)]
            #badttbar = ttbarmass[ (self.label_to_int_dict[ilabel] == 36) & (ttbarmass > 1100) & (ttbarmass < 1800)]

#            if icat == "2t0bcen": 
#                print("------")
#                print("badweights ", ak.max(Weights))

            output['ttbarmass_bare'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     ttbarmass = ak.to_numpy(ttbarmass[icat])
                                    )

            # probe ttbar candidate histograms
            output['probept'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetpt = ak.to_numpy(pT[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['probep'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetp = ak.to_numpy(p[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )

            # jet histograms 
            output['jetpt'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetpt = ak.to_numpy(jetpt[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jeteta'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jeteta = ak.to_numpy(jeteta[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jetphi'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetphi = ak.to_numpy(jetphi[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jety'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jety = ak.to_numpy(jety[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jetdy'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetdy = ak.to_numpy(jetdy[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            
            output['jetmass'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetmass = ak.to_numpy(jetmass[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['SDmass'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetmass = ak.to_numpy(SDmass[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )


            # mistag rate histograms
            output['numerator'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetp = ak.to_numpy(numerator[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['denominator'].fill(dataset = dataset,
                                     anacat = self.label_to_int_dict[ilabel],
                                     jetp = ak.to_numpy(denominator[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )


            # top tagger histograms
            # output['tau32'].fill(dataset = dataset,
            #                          anacat = self.label_to_int_dict[ilabel],
            #                          tau32 = ak.to_numpy(Tau32[icat]),
            #                          weight = ak.to_numpy(Weights[icat]),
            #                         )
            
        del df
        # MemoryMb()
        return output
    
    def postprocess(self, accumulator):
        return accumulator
    
#===================================================================================================================================================    
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#=================================================================================================================================================== 
#===================================================================================================================================================     
