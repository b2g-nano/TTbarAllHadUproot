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
import sys
import os 
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
                 tau32Cut=0.65, ak8PtMin=400., bdisc=0.5847, deepAK8Cut=0.435, BDirect='',
                 year=None, apv='', vfp='',eras=[], UseLookUpTables=False, lu=None, extraDaskDirectory='',
                 ModMass=False, RandomDebugMode=False, UseEfficiencies=False, xsSystematicWeight=1., lumSystematicWeight=1.,
                 ApplybtagSF=False, ScaleFactorFile='', ApplyttagSF=False, ApplyTopReweight=False, 
                 ApplyJes=False, var="nominal", ApplyPdf=False, ApplyPrefiring=False, ApplyPUweights=False,
                 ApplyHEMCleaning=False, trigs_to_run=[''],
                 sysType=None):
        
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
        self.extraDaskDirectory = extraDaskDirectory
        self.UseLookUpTables = UseLookUpTables
        self.ModMass = ModMass
        self.RandomDebugMode = RandomDebugMode
        self.ScaleFactorFile = ScaleFactorFile
        self.ApplybtagSF = ApplybtagSF # Only apply scale factors when MC efficiencies are being imported in second run of processor
        self.ApplyttagSF = ApplyttagSF
        self.ApplyTopReweight = ApplyTopReweight
        self.ApplyJes = ApplyJes
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
        
        
        # --- anti-tag+probe, anti-tag, pre-tag, 0, 1, >=1, 2 ttags, any t-tag (>=0t) --- #
        self.ttagcats = ["AT&Pt", "at", "pret", "0t", "1t", ">=1t", "2t", ">=0t"] 
        
        # --- 0, 1, or 2 b-tags --- #
        self.btagcats = ["0b", "1b", "2b"]
        
        # --- Central and forward --- #
        self.ycats = ['cen', 'fwd']
        
        # --- Combine categories like "0bcen", "0bfwd", etc: --- #
        self.anacats = [ t+b+y for t,b,y in itertools.product( self.ttagcats, self.btagcats, self.ycats) ]
        self.label_dict = {i: label for i, label in enumerate(self.anacats)}
        
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

        # --- axes for top tagger --- #
        manual_axis = hist.axis.Variable(manual_bins, name="jetp", label=r"Jet Momentum [GeV]")
        tagger_axis = hist.axis.Regular(50, 0, 1, name="tagger", label=r"deepTag")
        subjettagger_axis = hist.axis.Regular(50, -2, 1, name="subjettagger", label=r"Deep B")
        tau32_axis  = hist.axis.Regular(50, 0, 2, name="tau32", label=r"$\tau_3/\tau_2$")

        # --- axes for subjets --- #
        subjetmass_axis = hist.axis.Regular(50, 0, 500, name="subjetmass", label=r"SubJet $m$ [GeV]")
        subjetpt_axis   = hist.axis.Regular(50, 400, 2000, name="subjetpt", label=r"SubJet $p_{T}$ [GeV]")
        subjeteta_axis  = hist.axis.Regular(50, -2.4, 2.4, name="subjeteta", label=r"SubJet $\eta$")
        subjetphi_axis  = hist.axis.Regular(50, -np.pi, np.pi, name="subjetphi", label=r"SubJet $\phi$")
        
        # --- axes for weights --- #
        jethem_axis    = hist.axis.Regular(30, 0, 1.5, name=   "JetWeights", label=r"2018 HEM Weights")
        fatjethem_axis = hist.axis.Regular(30, 0, 1.5, name="FatJetWeights", label=r"2018 HEM Weights")
        subjethem_axis = hist.axis.Regular(30, 0, 1.5, name="SubJetWeights", label=r"2018 HEM Weights")
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

            'ttbarmass_jesUp'   : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_jesDown' : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),
            'ttbarmass_jesNom'  : hist.Hist(dataset_axis, cats_axis, ttbarmass_axis, storage="weight", name="Counts"),

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

            'deepTagMD_TvsQCD' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, SDjetmass_axis, tagger_axis, storage="weight", name="Counts"),
            'deepb' : hist.Hist(dataset_axis, subjetmass_axis, subjetpt_axis, subjettagger_axis, storage="weight", name="Counts"),
            'tau32'        : hist.Hist(dataset_axis, cats_axis, tau32_axis, storage="weight", name="Counts"),

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
            
            # 'weights_HEM' : hist.Hist(dataset_axis, jethem_axis, fatjethem_axis, subjethem_axis, storage="weight", name="Counts"),
            
            'weights_prefiringUp'  : hist.Hist(dataset_axis, cats_axis, prefiring_axis, storage="weight", name="Counts"),
            'weights_prefiringDown'  : hist.Hist(dataset_axis, cats_axis, prefiring_axis, storage="weight", name="Counts"),
            'weights_prefiringNom'  : hist.Hist(dataset_axis, cats_axis, prefiring_axis, storage="weight", name="Counts"),
            
            #********************************************************************************************************************#

            'cutflow': processor.defaultdict_accumulator(int),
        }             
            
        
        
#   =======================================================================
#   FFFFFFF U     U N     N   CCCC  TTTTTTT IIIIIII   OOO   N     N   SSSSS     
#   F       U     U NN    N  C         T       I     O   O  NN    N  S          
#   F       U     U N N   N C          T       I    O     O N N   N S           
#   FFFFFFF U     U N  N  N C          T       I    O     O N  N  N  SSSSS      
#   F       U     U N   N N C          T       I    O     O N   N N       S     
#   F        U   U  N    NN  C         T       I     O   O  N    NN      S      
#   F         UUU   N     N   CCCC     T    IIIIIII   OOO   N     N SSSSS
#   =======================================================================

    def ConvertLabelToInt(self, mapping, str_label):
        for intkey, string in mapping.items():
            if str_label == string:
                return intkey

        return "The label has not been found :("

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
    
    def GetJESUncertainties(self, FatJets, GenJets, events):
        
        ext = extractor()
        ext.add_weight_sets([
            "* * TTbarAllHadUproot/CorrectionFiles/JEC/Summer19UL17_V5_MC/Summer19UL17_V5_MC_L1FastJet_AK8PFchs.jec.txt",
            "* * TTbarAllHadUproot/CorrectionFiles/JEC/Summer19UL17_V5_MC/Summer19UL17_V5_MC_L2Relative_AK8PFchs.jec.txt",
            "* * TTbarAllHadUproot/CorrectionFiles/JEC/Summer19UL17_V5_MC/Summer19UL17_V5_MC_L3Absolute_AK8PFchs.jec.txt",
            "* * TTbarAllHadUproot/CorrectionFiles/JEC/Summer19UL17_V5_MC/Summer19UL17_V5_MC_UncertaintySources_AK8PFchs.junc.txt",
            "* * TTbarAllHadUproot/CorrectionFiles/JEC/Summer19UL17_V5_MC/Summer19UL17_V5_MC_Uncertainty_AK8PFchs.junc.txt",
        ])
        ext.finalize()

        jec_stack_names = [
            "Summer19UL17_V5_MC_L1FastJet_AK8PFchs",
            "Summer19UL17_V5_MC_L2Relative_AK8PFchs",
            "Summer19UL17_V5_MC_L3Absolute_AK8PFchs",
            "Summer19UL17_V5_MC_Uncertainty_AK8PFchs",
        ]

        evaluator = ext.make_evaluator()

        jec_inputs = {name: evaluator[name] for name in jec_stack_names}
        jec_stack = JECStack(jec_inputs)

        name_map = jec_stack.blank_name_map
        name_map['JetPt'] = 'pt'
        name_map['JetMass'] = 'mass'
        name_map['JetEta'] = 'eta'
        name_map['JetA'] = 'area'

        
        # match gen jets to AK8 jets
        matched_genjet_index = ak.mask(FatJets.genJetIdx, (FatJets.genJetIdx != -1) & (FatJets.genJetIdx < ak.count(GenJets.pt, axis=1)))
        matched_GenJet_pt = GenJets.pt[matched_genjet_index]        

        FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
        FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
        FatJets['pt_gen'] = matched_GenJet_pt
        FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]
        
        name_map['ptGenJet'] = 'pt_gen'
        name_map['ptRaw'] = 'pt_raw'
        name_map['massRaw'] = 'mass_raw'
        name_map['Rho'] = 'rho'


        events_cache = events.caches[0]
        corrector = FactorizedJetCorrector(
            Fall17_17Nov2017_V32_MC_L2Relative_AK8PFPuppi=evaluator['Summer19UL17_V5_MC_L2Relative_AK8PFchs'],
        )
        uncertainties = JetCorrectionUncertainty(
            Fall17_17Nov2017_V32_MC_Uncertainty_AK8PFPuppi=evaluator['Summer19UL17_V5_MC_Uncertainty_AK8PFchs']
        )

        jet_factory = CorrectedJetsFactory(name_map, jec_stack)
        corrected_jets = jet_factory.build(FatJets, lazy_cache=events_cache)
        
        # nominal jes
        jes_correction = corrected_jets.pt/corrected_jets.pt_raw
        CorrectedJets = corrected_jets

        if (self.var == "up"):
            jes_correction = corrected_jets.JES_jes.up.pt/corrected_jets.pt_raw
            CorrectedJets = corrected_jets.JES_jes.up
            
        elif (self.var == "down"):
            jes_correction = corrected_jets.JES_jes.down.pt/corrected_jets.pt_raw
            CorrectedJets = corrected_jets.JES_jes.up

        
        
        return CorrectedJets
    
    
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

        isData = ('JetHT' in dataset) or ('SingleMu' in dataset)
        
        IOV = ('2016APV' if any(regularexpressions.findall(r'preVFP', dataset))
               else '2018' if any(regularexpressions.findall(r'UL18', dataset))
               else '2017' if any(regularexpressions.findall(r'UL17', dataset))
               else '2016')
                
        # ---- Define lumimasks ---- #
        
        # if isData: 
        #     lumi_mask = np.array(self.lumimasks[IOV](events.run, events.luminosityBlock), dtype=bool)
        #     events = events[lumi_mask]
        
        
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
        
        # ---- Get event weights from dataset ---- #
        if isData: # If data is used...
            # print('if isData command works')
            evtweights = np.ones(ak.to_awkward0(FatJets).size) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = events.Generator_weight
        # ---- Show all events ---- #
        output['cutflow']['all events'] += ak.to_awkward0(FatJets).size
        
        # ---- Define the SumW2 for MC Datasets (Probably unnecessary now) ---- #
        output['cutflow']['sumw'] += np.sum(evtweights)
        output['cutflow']['sumw2'] += np.sum(evtweights**2)
        
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
                                
            CorrectedJets = self.GetJESUncertainties(FatJets, GenJets, events)
            
            FatJets['pt'] = CorrectedJets['pt']
            FatJets['eta'] = CorrectedJets['eta']
            FatJets['phi'] = CorrectedJets['phi']
            FatJets['mass'] = CorrectedJets['mass']
            
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
                    
            JetWeights_HEM = self.HEMCleaning(Jets)
            FatJetWeights_HEM = self.HEMCleaning(FatJets)
            SubJetWeights_HEM = self.HEMCleaning(SubJets)

            output['weights_HEM'].fill(dataset = dataset,
                             JetWeights = JetWeights_HEM,
                             FatJetWeights = FatJetWeights_HEM,
                             SubJetWeights = SubJetWeights_HEM
                            )

            Jets = ak.with_field(Jets, JetWeights_HEM*Jets.pt, 'pt')
            FatJets = ak.with_field(FatJets, FatJetWeights_HEM*FatJets.pt, 'pt')
            SubJets = ak.with_field(SubJets, SubJetWeights_HEM*SubJets.pt, 'pt')

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
        
        filteredEvents = np.array([getattr(events, f'Flag_{MET_filters[IOV][i]}') for i in range(len(MET_filters[IOV]))])
        filteredEvents = np.logical_or.reduce(filteredEvents, axis=0)
        
        if ak.sum(filteredEvents) < 1 :
            print("\nNo events passed the MET filters.\n")
            return output
        else:
            FatJets = FatJets[filteredEvents]
            Jets = Jets[filteredEvents]
            SubJets = SubJets[filteredEvents]
            evtweights = evtweights[filteredEvents]
            events = events[filteredEvents]
            
            output['cutflow']['Passed MET Filters'] += ak.sum(filteredEvents)


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
                
        if isData:
            FatJets = FatJets[condition]
            Jets = Jets[condition]
            SubJets = SubJets[condition]
            evtweights = evtweights[condition]
            events = events[condition]
            
            output['cutflow']['Passed Trigger(s)'] += ak.sum(condition)
            
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
        hT = ak.to_awkward0(Jets.pt).sum()
        passhT = (hT > self.htCut)
        FatJets = FatJets[passhT]
        Jets = Jets[passhT]
        SubJets = SubJets[passhT]
        evtweights = evtweights[passhT]
        events = events[passhT]
        if not isData:
            GenJets = GenJets[passhT]
        
        output['cutflow']['Passed HT Cut'] += ak.to_awkward0(passhT).sum()
          
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['Passed Loose Jet ID'] += ak.to_awkward0(jet_id).any().sum()
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        FatJets_rapidity = .5*np.log( (FatJets.p4.energy + FatJets.p4.pz)/(FatJets.p4.energy - FatJets.p4.pz) )
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets_rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['Passed pT,y Cut'] += ak.to_awkward0(jetkincut_index).any().sum()
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (ak.num(FatJets, axis=-1) == 2)
        FatJets = FatJets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin]
        events = events[twoFatJetsKin]
        evtweights = evtweights[twoFatJetsKin]
        if not isData:
            GenJets = GenJets[twoFatJetsKin]
        
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
        Jets = Jets[oneTTbar]
        SubJets = SubJets[oneTTbar]
        events = events[oneTTbar]
        evtweights = evtweights[oneTTbar]
        if not isData:
            GenJets = GenJets[oneTTbar]
            
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        """ NOTE: Should find function for this; avoids 2pi problem """
        dPhiCut = ttbarcands.slot0.p4.delta_phi(ttbarcands.slot1.p4) > 2.1
        dPhiCut = ak.flatten(dPhiCut)
        output['cutflow']['Passed dPhi Cut'] += ak.to_awkward0(dPhiCut).sum()
        ttbarcands = ttbarcands[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        Jets = Jets[dPhiCut]
        SubJets = SubJets[dPhiCut] 
        events = events[dPhiCut]
        evtweights = evtweights[dPhiCut]
        if not isData:
            GenJets = GenJets[dPhiCut]
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.slot0.subJetIdx1 > -1) & (ttbarcands.slot0.subJetIdx2 > -1)) # 1st candidate has two subjets
        hasSubjets1 = ((ttbarcands.slot1.subJetIdx1 > -1) & (ttbarcands.slot1.subJetIdx2 > -1)) # 2nd candidate has two subjets

        GoodSubjets = ak.flatten(((hasSubjets0) & (hasSubjets1))) # Selection of 4 (leading) subjects
        output['cutflow']['Good Subjets'] += ak.to_awkward0(GoodSubjets).sum()
        ttbarcands = ttbarcands[GoodSubjets] # Choose only ttbar candidates with this selection of subjets
        FatJets = FatJets[GoodSubjets]
        SubJets = SubJets[GoodSubjets]
        events = events[GoodSubjets]
        Jets = Jets[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
        if not isData:
            GenJets = GenJets[GoodSubjets]
        
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
        
        output["deepb"].fill(dataset = dataset,
                            subjetmass = ak.to_numpy(ak.flatten(SubJet01.mass)),
                            subjetpt = ak.to_numpy(ak.flatten(SubJet01.pt)),
                            subjettagger = ak.to_numpy(ak.flatten(SubJet01.btagDeepB)),
                            weight = ak.to_numpy(evtweights))
        
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
        
        btag_s0 = ( np.maximum(SubJet01.btagCSVV2 , SubJet02.btagCSVV2) > self.bdisc )
        btag_s1 = ( np.maximum(SubJet11.btagCSVV2 , SubJet12.btagCSVV2) > self.bdisc )
        # print(f'Jet 0\'s first subjet\'s CSVV2 = {SubJet01.btagCSVV2}', flush=True)
        # print(f'Jet 0\'s second subjet\'s CSVV2 = {SubJet02.btagCSVV2}', flush=True)
        # print(f'Jet 1\'s first subjet\'s CSVV2 = {SubJet11.btagCSVV2}', flush=True)
        # print(f'Jet 1\'s second subjet\'s CSVV2 = {SubJet12.btagCSVV2}\n-----------------------------------------\n', flush=True)
        # print(f'Jet 0\'s first subjet\'s DeepB = {SubJet01.btagDeepB}', flush=True)
        # print(f'Jet 0\'s second subjet\'s DeepB = {SubJet02.btagDeepB}', flush=True)
        # print(f'Jet 1\'s first subjet\'s DeepB = {SubJet11.btagDeepB}', flush=True)
        # print(f'Jet 1\'s second subjet\'s DeepB = {SubJet12.btagDeepB}\n-----------------------------------------\n', flush=True)
        # print(f'Jet 0\'s largest DeepB? = {np.maximum(SubJet01.btagDeepB , SubJet02.btagDeepB)}', flush=True)
        # print(f'Jet 1\'s largest DeepB? = {np.maximum(SubJet11.btagDeepB , SubJet12.btagDeepB)}', flush=True)
        # print(f'is Jet 0 btagged? = {btag_s0}', flush=True)
        # print(f'is Jet 1 btagged? = {btag_s1}', flush=True)
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
                            EffFileDict['Eff_File_'+subjet_info[1]].append(self.extraDaskDirectory+'TTbarAllHadUproot/FlavorTagEfficiencies/' 
                                                                           + self.BDirect + flav_tag 
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
        # print(labels_and_categories)
        
        # ---- Variables for Kinematic Histograms ---- #
        # ---- "slot0" is the control jet, "slot1" is the probe jet ---- #
        jetpt = ak.flatten(ttbarcands.slot1.pt)
        jeteta = ak.flatten(ttbarcands.slot1.eta)
        jetphi = ak.flatten(ttbarcands.slot1.phi)
        jetmass = ak.flatten(ttbarcands.slot1.mass)
        
        SDmass = ak.flatten(ttbarcands.slot1.msoftdrop)
        Tau32 = ak.flatten((ttbarcands.slot1.tau3/ttbarcands.slot1.tau2))
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
        
        for ilabel,icat in labels_and_categories.items():

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
                    print('Something is wrong...\n\nNecessary JetHT LUT(s) not found')
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
                        print(df)
                    print(VE)
            
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
                if self.year > 0:
                    QCD_unweighted = util.load(self.extraDaskDirectory+'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/QCD/'
                                               +self.BDirect+str(self.year)+'/'+self.apv+'/TTbarRes_0l_UL'+str(self.year-2000)+self.vfp+'_QCD.coffea') 
                    
                    # ---- Define Histogram ---- #
                    loaded_dataset = 'UL'+str(self.year-2000)+self.vfp+'_QCD'
                    QCD_hist = QCD_unweighted['jetmass'][loaded_dataset, self.ConvertLabelToInt(self.label_dict, '2t' + str(ilabel[-5:])), :]
                    
                else: # All years !NOTE: Needs to be fixed for all years later!
                    QCD_unwgt_2016 = util.load(self.extraDaskDirectory+'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/QCD/'
                                               +self.BDirect+'2016/'+self.apv+'/TTbarRes_0l_UL16'+self.vfp+'_QCD.coffea') 
                    # QCD_unwgt_2017 = util.load(self.extraDaskDirectory+'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/QCD/'
                    #                            +self.BDirect+'2017/'+self.apv+'/TTbarRes_0l_UL17'+self.vfp+'_QCD.coffea') 
                    # QCD_unwgt_2018 = util.load(self.extraDaskDirectory+'TTbarAllHadUproot/CoffeaOutputsForCombine/Coffea_FirstRun/QCD/'
                    #                            +self.BDirect+'2018/'+self.apv+'/TTbarRes_0l_UL18'+self.vfp+'_QCD.coffea') 
                    
                    # ---- Define Histogram ---- #
                    QCD_hist_2016 = QCD_unwgt_2016['jetmass']['UL16'+self.vfp+'_QCD', self.ConvertLabelToInt(self.label_dict, '2t' + str(ilabel[-5:])), :]
                    # QCD_hist_2017 = QCD_unwgt_2017['jetmass']['UL17'+self.vfp+'_QCD', self.ConvertLabelToInt(self.label_dict, '2t' + str(ilabel[-5:])), :]
                    # QCD_hist_2018 = QCD_unwgt_2018['jetmass']['UL18'+self.vfp+'_QCD', self.ConvertLabelToInt(self.label_dict, '2t' + str(ilabel[-5:])), :]
                    
                    QCD_hist = QCD_hist_2016.copy()
                    # QCD_hist += (QCD_hist_2017)
                    # QCD_hist += (QCD_hist_2018)
                    
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
                    
                    prefiringNom, prefiringDown, prefiringUp = self.GetL1PreFiringWeight(events)

                    Weights_prefiringUp = Weights * prefiringUp
                    Weights_prefiringDown = Weights * prefiringDown
                    Weights_prefiringNom = Weights * prefiringNom

                    output['ttbarmass_prefiringNom'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_prefiringNom[icat]),
                                    )
                    output['ttbarmass_prefiringUp'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_prefiringUp[icat]),
                                    )
                    output['ttbarmass_prefiringDown'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_prefiringDown[icat]),
                                    )

                    output['weights_prefiringNom'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     Weights = prefiringNom,
                                    )
                    output['weights_prefiringUp'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     Weights = prefiringUp,
                                    )
                    output['weights_prefiringDown'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
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
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_pdfNom[icat]),
                                    )
                    output['ttbarmass_pdfUp'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_pdfUp[icat]),
                                    )
                    output['ttbarmass_pdfDown'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_pdfDown[icat]),
                                    )
                    
                if self.ApplyPUweights:
                    
                    puNom, puDown, puUp = self.GetPUSF(events)

                    Weights_puUp = Weights * puUp
                    Weights_puDown = Weights * puDown
                    Weights_puNom = Weights * puNom

                    output['ttbarmass_puNom'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_puNom[icat]),
                                    )
                    output['ttbarmass_puUp'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights_puUp[icat]),
                                    )
                    output['ttbarmass_puDown'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
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
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     ttbarmass = ak.to_numpy(ttbarmass[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )

            # probe ttbar candidate histograms
            output['probept'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetpt = ak.to_numpy(pT[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['probep'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetp = ak.to_numpy(p[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )

            # jet histograms 
            output['jetpt'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetpt = ak.to_numpy(jetpt[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jeteta'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jeteta = ak.to_numpy(jeteta[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jetphi'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetphi = ak.to_numpy(jetphi[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jety'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jety = ak.to_numpy(jety[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jetdy'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetdy = ak.to_numpy(jetdy[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            # 'deepTagMD_TvsQCD' : hist.Hist(dataset_axis, cats_axis, jetpt_axis, jetmass_axis, tagger_axis, storage="weight", name="Counts"),
            
            output['deepTagMD_TvsQCD'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetpt = ak.to_numpy(jetpt[icat]),
                                     SDjetmass = ak.to_numpy(SDmass[icat]),
                                     tagger = ak.to_numpy(ak8tagger[icat]),       
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['jetmass'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetmass = ak.to_numpy(jetmass[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['SDmass'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetmass = ak.to_numpy(SDmass[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )


            # mistag rate histograms
            output['numerator'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetp = ak.to_numpy(numerator[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )
            output['denominator'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     jetp = ak.to_numpy(denominator[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )


            # top tagger histograms
            output['tau32'].fill(dataset = dataset,
                                     anacat = self.ConvertLabelToInt(self.label_dict, ilabel),
                                     tau32 = ak.to_numpy(Tau32[icat]),
                                     weight = ak.to_numpy(Weights[icat]),
                                    )

 
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
        
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary Dataset")

        ttbarmass_axis = hist.axis.Regular(50, 800, 8000, name="ttbarmass", label=r"$m_{t\bar{t}}$ [GeV]")
        
        subjetpt_axis = hist.axis.Regular(25, 0, 2000, name = "subjetpt", label = "SubJet $p_{T}$ [GeV]")
        subjetpt_laxis = hist.axis.Regular( 8, 0, 2000, name = "subjetpt", label = r"SubJet $p_{T}$ [GeV]") #Larger bins
        subjetpt_maxis = hist.axis.Variable(manual_subjetpt_bins, name = "subjetpt", label = r"SubJet $p_T$ [GeV]") #Manually defined bins for better statistics per bin
        
        subjeteta_axis = hist.axis.Regular(25, 0, 2.4, name = "subjeteta", label = r"SubJet $\eta$")
        subjeteta_laxis = hist.axis.Regular(8, 0, 2.4, name = "subjeteta", label = r"SubJet $\eta$") #Larger bins
        subjeteta_maxis = hist.axis.Variable(manual_subjeteta_bins, name = "subjeteta", label = r"SubJet $\eta$") #Manually defined bins for better statistics per bin

        
#    ====================================================================
#    EEEEEEE FFFFFFF FFFFFFF      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#    E       F       F            H     H    I     S         T     S          
#    E       F       F            H     H    I    S          T    S           
#    EEEEEEE FFFFFFF FFFFFFF      HHHHHHH    I     SSSSS     T     SSSSS      
#    E       F       F            H     H    I          S    T          S     
#    E       F       F            H     H    I         S     T         S      
#    EEEEEEE F       F       *    H     H IIIIIII SSSSS      T    SSSSS
#    ====================================================================

        self.histo_dict = {
            # ---- 2D SubJet b-tag Efficiency ---- #
            'b_eff_numerator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'b_eff_denominator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'b_eff_numerator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'b_eff_denominator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'b_eff_numerator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_numerator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'b_eff_denominator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'b_eff_denominator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            # ---- 2D SubJet c-tag Efficiency ---- #
            'c_eff_numerator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'c_eff_denominator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'c_eff_numerator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'c_eff_denominator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'c_eff_numerator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_numerator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'c_eff_denominator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'c_eff_denominator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            # ---- 2D SubJet light quark-tag Efficiency ---- #
            'udsg_eff_numerator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'udsg_eff_denominator_s01': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s02': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s11': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s12': hist.Hist(dataset_axis, subjetpt_axis, subjeteta_axis, storage = "weight", name = "Counts"),
            
            'udsg_eff_numerator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'udsg_eff_denominator_s01_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s02_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s11_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s12_largerbins': hist.Hist(dataset_axis, subjetpt_laxis, subjeteta_laxis, storage = "weight", name = "Counts"),
            
            'udsg_eff_numerator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_numerator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'udsg_eff_denominator_s01_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s02_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s11_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            'udsg_eff_denominator_s12_manualbins': hist.Hist(dataset_axis, subjetpt_maxis, subjeteta_maxis, storage = "weight", name = "Counts"),
            
            'cutflow': processor.defaultdict_accumulator(int),
        }
                                                          
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
        output = self.histo_dict
        
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
        twoFatJetsKin = (ak.num(FatJets, axis=-1) > 1)
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
        oneTTbar = (ak.num(ttbarcands, axis=-1) > 0)
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
        self.ttagcats_forTriggerAnalysis = ["NoCut", ">=1t"]                                     
        self.label_dict = {i: label for i, label in enumerate(self.ttagcats_forTriggerAnalysis)}
                                                          
        dataset_axis = hist.axis.StrCategory([], growth=True, name = "dataset", label = "Primary Dataset")
        cats_axis = hist.axis.Regular(2, 0, 1, name = "anacat", label = "Analysis Category")
        jetht_axis = hist.axis.Variable(manual_jetht_bins, name = "Jet_HT", label = r'$AK4\ Jet\ HT$') # Used for Trigger Analysis
        sdMass_axis = hist.axis.Variable(manual_sdMass_bins, name = "Jet_sdMass", label = r'$AK4\ M_{SD}$')
        
#    ===================================================================================================
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR      H     H IIIIIII   SSSSS TTTTTTT   SSSSS     
#       T    R     R    I    G       G       E       R     R     H     H    I     S         T     S          
#       T    R     R    I    G       G       E       R     R     H     H    I    S          T    S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR      HHHHHHH    I     SSSSS     T     SSSSS      
#       T    R   R      I    G     G G     G E       R   R       H     H    I          S    T          S     
#       T    R    R     I    G     G G     G E       R    R      H     H    I         S     T         S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R     H     H IIIIIII SSSSS      T    SSSSS
#    ===================================================================================================
        
        self.histo_dict = {
           'condition1_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition2_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition3_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition4_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition5_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'condition_denominator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
            
           'trigger1_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger2_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger3_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger4_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
           'trigger5_numerator': hist.Hist(dataset_axis, cats_axis, jetht_axis, sdMass_axis, storage = "weight", name = "Counts"),
            
           'cutflow': processor.defaultdict_accumulator(int),
        }
        
            
    @property
    def accumulator(self):
        return self._accumulator
    
    def ConvertLabelToInt(self, mapping, str_label):
        for intkey, string in mapping.items():
            if str_label == string:
                return intkey

        return "The label has not been found :("

    def process(self, events):
        output = self.histo_dict
        
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
            # "PU": events.Pileup_nPU,
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
        
        # print(Jets.PU, '\n')
        trigDenom = events.HLT_Mu50 | events.HLT_IsoMu24
        # print(HLT_PF_triggers)
        # print(HLT_AK8_triggers)
        # print(events.HLT_IsoMu24)
        # print(trigDenom)
        # print('-----------------------------------------------')
        
        # is2017Bun = False # Switch to isolate 2017 B chunks
        
        trigger1 = None
        trigger2 = None
        trigger3 = None
        trigger4 = None
        
        if self.year == 2016: #HLT PFHT890
            trigger1 = events.HLT_PFHT900
            trigger2 = events.HLT_AK8PFHT700_TrimR0p1PT0p03Mass50
            trigger3 = events.HLT_AK8PFJet450
            trigger4 = events.HLT_AK8PFJet360_TrimMass30
        elif self.year == 2017:
            trigger1 = events.HLT_PFHT1050
            trigger2 = events.HLT_AK8PFJet550 
        else:
            trigger1 = events.HLT_PFHT1050
            trigger2 = events.HLT_AK8PFHT800_TrimMass50
            trigger3 = events.HLT_AK8PFJet550 
            trigger4 = events.HLT_AK8PFJet400_TrimMass30
            
        Trigger1 = trigger1 & trigDenom
        Trigger2 = trigger2 & trigDenom
        if self.year != 2017:
            Trigger3 = trigger3 & trigDenom
            Trigger4 = trigger4 & trigDenom
            

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
        
        # ---- Apply Trigger(s) ---- # 
        if self.year != 2017: # Include other years later...
            # ---- Defining Trigger Analysis Conditions ---- #
            condition1 = trigger1 & trigDenom 
            condition2 = (trigger1 | trigger2) & trigDenom
            condition3 = ((trigger1 | trigger2) | trigger3) & trigDenom
            condition4 = ((trigger1 | trigger2) | (trigger3 | trigger4)) & trigDenom
            
        else: # Special for 2017 only
            # ---- Defining Trigger Analysis Conditions ---- #
            condition1 = trigger1 & trigDenom 
            condition2 = (trigger1 | trigger2) & trigDenom
        
            
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
        
        Trigger1 = Trigger1[twoFatJetsKin]
        Trigger2 = Trigger2[twoFatJetsKin]
        condition1 = condition1[twoFatJetsKin]
        condition2 = condition2[twoFatJetsKin]

        if self.year != 2017:
            Trigger3 = Trigger3[twoFatJetsKin]
            Trigger4 = Trigger4[twoFatJetsKin]
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
        
        Trigger1 = Trigger1[oneTTbar]
        Trigger2 = Trigger2[oneTTbar]
        condition1 = condition1[oneTTbar]
        condition2 = condition2[oneTTbar]
        
        if self.year != 2017:
            Trigger3 = Trigger3[oneTTbar]
            Trigger4 = Trigger4[oneTTbar]
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
        
        Trigger1 = Trigger1[dPhiCut]
        Trigger2 = Trigger2[dPhiCut]
        condition1 = condition1[dPhiCut]
        condition2 = condition2[dPhiCut]
        
        if self.year != 2017:
            Trigger3 = Trigger3[dPhiCut]
            Trigger4 = Trigger4[dPhiCut]
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
        
        Trigger1 = Trigger1[GoodSubjets]
        Trigger2 = Trigger2[GoodSubjets]
        condition1 = condition1[GoodSubjets]
        condition2 = condition2[GoodSubjets]
        
        if self.year != 2017:
            Trigger3 = Trigger3[GoodSubjets]
            Trigger4 = Trigger4[GoodSubjets]
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
        ttag0 =   (~ttag_s0) & (~ttag_s1) # No tops tagged (0t) (will not store 2D hist values of SD within softdrop window)
        ttagI =   ttag_s0 | ttag_s1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttagAny = ttag0 | ttagI # Any tag region (0, 1, or 2)
        
#    ===========================================================================================================================        
#    TTTTTTT RRRRRR  IIIIIII GGGGGGG GGGGGGG EEEEEEE RRRRRR         A    N     N    A    L       Y     Y   SSSSS IIIIIII   SSSSS     
#       T    R     R    I    G       G       E       R     R       A A   NN    N   A A   L        Y   Y   S         I     S          
#       T    R     R    I    G       G       E       R     R      A   A  N N   N  A   A  L         Y Y   S          I    S           
#       T    RRRRRR     I    G  GGGG G  GGGG EEEEEEE RRRRRR       AAAAA  N  N  N  AAAAA  L          Y     SSSSS     I     SSSSS      
#       T    R   R      I    G     G G     G E       R   R       A     A N   N N A     A L          Y          S    I          S     
#       T    R    R     I    G     G G     G E       R    R      A     A N    NN A     A L          Y         S     I         S      
#       T    R     R IIIIIII  GGGGG   GGGGG  EEEEEEE R     R     A     A N     N A     A LLLLLLL    Y    SSSSS   IIIIIII SSSSS
#    ===========================================================================================================================  
        
        TriggersDict = {
            '1': Trigger1,
            '2': Trigger2
        }   
    
        ConditionsDict = {
            '1': condition1,
            '2': condition2
        } 
        if self.year != 2017:
            TriggersDict['3'] = Trigger3
            TriggersDict['4'] = Trigger4
            ConditionsDict['3'] = condition3
            ConditionsDict['4'] = condition4
            

        # ---- Defining Jet Collections for Trigger Analysis Numerator and Denominator ---- #
        Jets_NumTrigger1 = Jets[Trigger1] # contains jets to be used as numerator for trigger eff
        Jets_NumTrigger2 = Jets[Trigger2]
        Jets_NumCondition1 = Jets[condition1] # contains jets to be used as numerator for trigger eff
        Jets_NumCondition2 = Jets[condition2]
        
        if self.year != 2017:
            Jets_NumTrigger3 = Jets[Trigger3]
            Jets_NumTrigger4 = Jets[Trigger4]
            Jets_NumCondition3 = Jets[condition3]
            Jets_NumCondition4 = Jets[condition4]
            
        Jets_DenomCondition = Jets[trigDenom] # contains jets to be used as denominator for trigger eff
        
        output['cutflow']['events with jets cond1'] +=  ak.to_awkward0(condition1).sum()
        output['cutflow']['events with jets cond2'] +=  ak.to_awkward0(condition2).sum()
        
        if self.year != 2017:
            output['cutflow']['events with jets cond3'] +=  ak.to_awkward0(condition3).sum()
            output['cutflow']['events with jets cond4'] +=  ak.to_awkward0(condition4).sum()
        
        output['cutflow']['events with jets Denom cond'] +=  ak.to_awkward0(trigDenom).sum()
        
        # ---- Must pass this cut before calculating HT variables for analysis ---- #
        passAK4_num1_trig = (Jets_NumTrigger1.pt > 30.) & (np.abs(Jets_NumTrigger1.eta) < 3.0) 
        passAK4_num2_trig = (Jets_NumTrigger2.pt > 30.) & (np.abs(Jets_NumTrigger2.eta) < 3.0)
        passAK4_num1 = (Jets_NumCondition1.pt > 30.) & (np.abs(Jets_NumCondition1.eta) < 3.0) 
        passAK4_num2 = (Jets_NumCondition2.pt > 30.) & (np.abs(Jets_NumCondition2.eta) < 3.0)
        
        if self.year != 2017:
            passAK4_num3_trig = (Jets_NumTrigger3.pt > 30.) & (np.abs(Jets_NumTrigger3.eta) < 3.0) 
            passAK4_num4_trig = (Jets_NumTrigger4.pt > 30.) & (np.abs(Jets_NumTrigger4.eta) < 3.0)
            passAK4_num3 = (Jets_NumCondition3.pt > 30.) & (np.abs(Jets_NumCondition3.eta) < 3.0)
            passAK4_num4 = (Jets_NumCondition4.pt > 30.) & (np.abs(Jets_NumCondition4.eta) < 3.0)
        
        passAK4_denom = (Jets_DenomCondition.pt > 30.) & (np.abs(Jets_DenomCondition.eta) < 3.0) 
        
        # ---------------------------------------------------------------------------------------------#
        # ---- Remember to have weights array that is consistently the same size as num and denom ---- #
        # --------- This is only because input later in the code expects an array of weights --------- #
        # ------------- despite this array being simply an array of ones for the sake of ------------- #
        # ----------------------------------- not altering the data ---------------------------------- #
        # ---------------------------------------------------------------------------------------------#
        
        Num1Wgt_trig = evtweights[Trigger1]
        Num2Wgt_trig = evtweights[Trigger2]
        Num1Wgt = evtweights[condition1]
        Num2Wgt = evtweights[condition2]
        
        if self.year != 2017:
            Num3Wgt_trig = evtweights[Trigger3]
            Num4Wgt_trig = evtweights[Trigger4]
            Num3Wgt = evtweights[condition3]
            Num4Wgt = evtweights[condition4]
        
        NumWgtTrigDict = {
            '1': Num1Wgt_trig,
            '2': Num2Wgt_trig
        }
        NumWgtDict = {
            '1': Num1Wgt,
            '2': Num2Wgt
        }
        
        if self.year != 2017:
            NumWgtTrigDict['3'] = Num3Wgt_trig
            NumWgtTrigDict['4'] = Num4Wgt_trig
            NumWgtDict['3'] = Num3Wgt
            NumWgtDict['4'] = Num4Wgt
            
        DenomWgt = evtweights[trigDenom]
        
        # ---- Defining Trigger Analysis Numerator(s) and Denominator as function of HT ---- #
        jet_HT_numerator1_trig = ak.sum(Jets_NumTrigger1[passAK4_num1_trig].pt, axis=-1) # Sum over each AK4 Jet per event
        jet_HT_numerator2_trig = ak.sum(Jets_NumTrigger2[passAK4_num2_trig].pt, axis=-1)
        jet_HT_numerator1 = ak.sum(Jets_NumCondition1[passAK4_num1].pt, axis=-1) # Sum over each AK4 Jet per event
        jet_HT_numerator2 = ak.sum(Jets_NumCondition2[passAK4_num2].pt, axis=-1)
        
        if self.year != 2017:
            jet_HT_numerator3_trig = ak.sum(Jets_NumTrigger3[passAK4_num3_trig].pt, axis=-1)
            jet_HT_numerator4_trig = ak.sum(Jets_NumTrigger4[passAK4_num4_trig].pt, axis=-1)
            jet_HT_numerator3 = ak.sum(Jets_NumCondition3[passAK4_num3].pt, axis=-1)
            jet_HT_numerator4 = ak.sum(Jets_NumCondition4[passAK4_num4].pt, axis=-1)
        
        jet_HT_numeratorTrigDict = {
            '1': jet_HT_numerator1_trig,
            '2': jet_HT_numerator2_trig
        }
        
        jet_HT_numeratorDict = {
            '1': jet_HT_numerator1,
            '2': jet_HT_numerator2
        }
        
        if self.year != 2017:
            jet_HT_numeratorTrigDict['3'] = jet_HT_numerator3_trig
            jet_HT_numeratorTrigDict['4'] = jet_HT_numerator4_trig
            jet_HT_numeratorDict['3'] = jet_HT_numerator3
            jet_HT_numeratorDict['4'] = jet_HT_numerator4
            
        jet_HT_denominator = ak.sum(Jets_DenomCondition[passAK4_denom].pt, axis=-1) # Sum over each AK4 Jet per event
        
        # ---- Defining Trigger Analysis Numerator(s) and Denominator as function of SD ---- #
        sdMass = ak.flatten(ttbarcands.slot0.msoftdrop)
        
        jet_SD_numerator1_trig = sdMass[Trigger1]
        jet_SD_numerator2_trig = sdMass[Trigger2]
        jet_SD_numerator1 = sdMass[condition1]
        jet_SD_numerator2 = sdMass[condition2]

        if self.year != 2017:
            jet_SD_numerator3_trig = sdMass[Trigger3]
            jet_SD_numerator4_trig = sdMass[Trigger4]
            jet_SD_numerator3 = sdMass[condition3]
            jet_SD_numerator4 = sdMass[condition4]
            
        jet_SD_denominator = sdMass[trigDenom]
        
        jet_SD_numeratorTrigDict = {
            '1': jet_SD_numerator1_trig,
            '2': jet_SD_numerator2_trig
        }
        
        jet_SD_numeratorDict = {
            '1': jet_SD_numerator1,
            '2': jet_SD_numerator2
        }
        
        if self.year != 2017:
            jet_SD_numeratorTrigDict['3'] = jet_SD_numerator3_trig
            jet_SD_numeratorTrigDict['4'] = jet_SD_numerator4_trig
            jet_SD_numeratorDict['3'] = jet_SD_numerator3
            jet_SD_numeratorDict['4'] = jet_SD_numerator4
        
        output['cutflow']['jets cond1 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num1)).sum()
        output['cutflow']['jets cond2 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num2)).sum()
        
        if self.year != 2017:
            output['cutflow']['jets cond3 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num3)).sum()
            output['cutflow']['jets cond4 with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_num4)).sum()
            
        output['cutflow']['jets Denom with ak4cut'] += ak.to_awkward0(ak.flatten(passAK4_denom)).sum()
        
        # ----------------- Keep track of cutflow for individual bins ---------------- #
        # ---- [200, 800, 840, 880, 920, 960, 1000, 1200, 1400, 1600, 1800, 2000] ---- #
        # num1_inBin1 = (200. < jet_HT_numerator1) & (jet_HT_numerator1 < 800.)
        # num2_inBin1 = (200. < jet_HT_numerator2) & (jet_HT_numerator2 < 800.)
        # num3_inBin1 = (200. < jet_HT_numerator3) & (jet_HT_numerator3 < 800.)
        # num4_inBin1 = (200. < jet_HT_numerator4) & (jet_HT_numerator4 < 800.)
        # denom_inBin1 = (200. < jet_HT_denominator) & (jet_HT_denominator < 800.)
        # num1_inBin11 = (1800. < jet_HT_numerator1) & (jet_HT_numerator1 < 2000.)
        # num2_inBin11 = (1800. < jet_HT_numerator2) & (jet_HT_numerator2 < 2000.)
        # num3_inBin11 = (1800. < jet_HT_numerator3) & (jet_HT_numerator3 < 2000.)
        # num4_inBin11 = (1800. < jet_HT_numerator4) & (jet_HT_numerator4 < 2000.)
        # denom_inBin11 = (1800. < jet_HT_denominator) & (jet_HT_denominator < 2000.)
        # output['cutflow']['numerator 1 in bin [200, 800]'] += ak.to_awkward0(num1_inBin1).sum()
        # output['cutflow']['numerator 2 in bin [200, 800]'] += ak.to_awkward0(num2_inBin1).sum()
        # output['cutflow']['numerator 3 in bin [200, 800]'] += ak.to_awkward0(num3_inBin1).sum()
        # output['cutflow']['numerator 4 in bin [200, 800]'] += ak.to_awkward0(num4_inBin1).sum()
        # output['cutflow']['denominator in bin [200, 800]'] += ak.to_awkward0(denom_inBin1).sum()
        # output['cutflow']['numerator 1 in bin [1800, 2000]'] += ak.to_awkward0(num1_inBin11).sum()
        # output['cutflow']['numerator 2 in bin [1800, 2000]'] += ak.to_awkward0(num2_inBin11).sum()
        # output['cutflow']['numerator 3 in bin [1800, 2000]'] += ak.to_awkward0(num3_inBin11).sum()
        # output['cutflow']['numerator 4 in bin [1800, 2000]'] += ak.to_awkward0(num4_inBin11).sum()
        # output['cutflow']['denominator in bin [1800, 2000]'] += ak.to_awkward0(denom_inBin11).sum()
        
        # ---- Define Categories for Trigger Analysis Denominator and Fill Hists ---- #
        ttags = [ttagAny[trigDenom],ttagI[trigDenom]]
        cats = [ ak.to_awkward0(ak.flatten(t)) for t in ttags ]
        labels_and_categories = dict(zip(self.ttagcats_forTriggerAnalysis, cats))
        for ilabel,icat in labels_and_categories.items():
            output['condition_denominator'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                Jet_HT = ak.to_numpy(jet_HT_denominator[icat]),
                                                Jet_sdMass = ak.to_numpy(jet_SD_denominator[icat]),
                                                weight = ak.to_numpy(DenomWgt[icat]))
        # ---- Define Categories for Trigger Analysis Numerators and Fill Hists---- #
        for i in range(1, len(ConditionsDict)+1):
            c = ConditionsDict[str(i)]
            ct = TriggersDict[str(i)]
            n_HT = jet_HT_numeratorDict[str(i)]
            n_SD = jet_SD_numeratorDict[str(i)]
            nt_HT = jet_HT_numeratorTrigDict[str(i)]
            nt_SD = jet_SD_numeratorTrigDict[str(i)]
            w = NumWgtDict[str(i)]
            wt = NumWgtTrigDict[str(i)]
            ttags_cond = [ttagAny[c],ttagI[c]]
            ttags_trig = [ttagAny[ct],ttagI[ct]]
            cats_cond = [ ak.to_awkward0(ak.flatten(t)) for t in ttags_cond]
            cats_trig = [ ak.to_awkward0(ak.flatten(t)) for t in ttags_trig]
            labels_and_categories_cond = dict(zip(self.ttagcats_forTriggerAnalysis, cats_cond))
            labels_and_categories_trig = dict(zip(self.ttagcats_forTriggerAnalysis, cats_trig))
            # print(labels_and_categories_cond)
            for ilabel,icat in labels_and_categories_cond.items():
                output['condition' + str(i) + '_numerator'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                                Jet_HT = ak.to_numpy(n_HT[icat]),
                                                                Jet_sdMass = ak.to_numpy(n_SD[icat]),
                                                                weight = ak.to_numpy(w[icat]))
            for ilabel,icat in labels_and_categories_trig.items():
                output['trigger' + str(i) + '_numerator'].fill(dataset = dataset, anacat = self.ConvertLabelToInt(self.label_dict, ilabel), 
                                                                Jet_HT = ak.to_numpy(nt_HT[icat]),
                                                                Jet_sdMass = ak.to_numpy(nt_SD[icat]),
                                                                weight = ak.to_numpy(wt[icat]))
 
        return output

    def postprocess(self, accumulator):
        return accumulator
