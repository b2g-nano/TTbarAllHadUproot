#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import scipy.stats as ss
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea import util
from awkward import JaggedArray
import numpy as np
import itertools
import pandas as pd
from numpy.random import RandomState


# In[ ]:


manual_bins = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 7000, 10000]


# In[ ]:


"""@TTbarResAnaHadronic Package to perform the data-driven mistag-rate-based ttbar hadronic analysis. 
"""
class TTbarResProcessor(processor.ProcessorABC):
    def __init__(self, prng=RandomState(1234567890), htCut=950., minMSD=105., maxMSD=210., tau32Cut=0.65, ak8PtMin=400., bdisc=0.8484,
                writePredDist=True,isData=True,year=2019, UseLookUpTables=False, lu=None, 
                ModMass=False, RandomDebugMode=False):
        
        self.prng = prng
        self.htCut = htCut
        self.minMSD = minMSD
        self.maxMSD = maxMSD
        self.tau32Cut = tau32Cut
        self.ak8PtMin = ak8PtMin
        self.bdisc = bdisc
        self.writePredDist = writePredDist
        self.writeHistFile = True
        self.isData = isData
        self.year=year
        self.UseLookUpTables = UseLookUpTables
        self.ModMass = ModMass
        self.RandomDebugMode = RandomDebugMode
        self.lu = lu # Look Up Tables
        
        self.ttagcats = ["Pt", "at", "pret", "0t", "1t", "1t+2t", "2t", "0t+1t+2t"] #anti-tag+probe, anti-tag, pre-tag, 0, 1, >=1, 2 ttags, any t-tag
        self.btagcats = ["0b", "1b", "2b"]   # 0, 1, >=2 btags
        self.ycats = ['cen', 'fwd']          # Central and forward
        # Combine categories like "0bcen", "0bfwd", etc:
        self.anacats = [ t+b+y for t,b,y in itertools.product( self.ttagcats, self.btagcats, self.ycats) ]
        print(self.anacats)
        
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        cats_axis = hist.Cat("anacat", "Analysis Category")
        
        jetmass_axis = hist.Bin("jetmass", r"Jet $m$ [GeV]", 50, 0, 500)
        jetpt_axis = hist.Bin("jetpt", r"Jet $p_{T}$ [GeV]", 50, 0, 5000)
        ttbarmass_axis = hist.Bin("ttbarmass", r"$m_{t\bar{t}}$ [GeV]", 50, 0, 5000)
        jeteta_axis = hist.Bin("jeteta", r"Jet $\eta$", 50, -5, 5)
        jetphi_axis = hist.Bin("jetphi", r"Jet $\phi$", 50, -np.pi, np.pi)
        jety_axis = hist.Bin("jety", r"Jet $y$", 50, -3, 3)
        jetdy_axis = hist.Bin("jetdy", r"Jet $\Delta y$", 50, 0, 5)
        manual_axis = hist.Bin("jetp", r"Jet Momentum [GeV]", manual_bins)
        tagger_axis = hist.Bin("tagger", r"deepTag", 50, 0, 1)
        tau32_axis = hist.Bin("tau32", r"$\tau_3/\tau_2$", 50, 0, 2)
        
        subjetmass_axis = hist.Bin("subjetmass", r"SubJet $m$ [GeV]", 50, 0, 500)
        subjetpt_axis = hist.Bin("subjetpt", r"SubJet $p_{T}$ [GeV]", 50, 0, 2000)
        subjeteta_axis = hist.Bin("subjeteta", r"SubJet $\eta$", 50, -4, 4)
        subjetphi_axis = hist.Bin("subjetphi", r"SubJet $\phi$", 50, -np.pi, np.pi)

        self._accumulator = processor.dict_accumulator({
            'ttbarmass': hist.Hist("Counts", dataset_axis, cats_axis, ttbarmass_axis),
            
            'jetmass':         hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass':          hist.Hist("Counts", dataset_axis, cats_axis, jetmass_axis),
            'SDmass_precat':   hist.Hist("Counts", dataset_axis, jetpt_axis, jetmass_axis),
            
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
            
            'subjetmass':   hist.Hist("Counts", dataset_axis, cats_axis, subjetmass_axis),
            'subjetpt':     hist.Hist("Counts", dataset_axis, cats_axis, subjetpt_axis),
            'subjeteta':    hist.Hist("Counts", dataset_axis, cats_axis, subjeteta_axis),
            'subjetphi':    hist.Hist("Counts", dataset_axis, cats_axis, subjetphi_axis),
            
            'numerator':   hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            'denominator': hist.Hist("Counts", dataset_axis, cats_axis, manual_axis),
            
            'cutflow': processor.defaultdict_accumulator(int),
            
        })

            
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        
        output = self.accumulator.identity()
        
        # ---- Define dataset ---- #
        dataset = df['dataset'] #coffea.processor.LazyDataFrame
        Dataset_info = df.available #list of available columns in LazyDataFrame object (Similar to 'Events->Show()' command in ROOT)
        
        # ---- Get triggers from Dataset_info ---- #
        #triggers = [itrig for itrig in Dataset_info if 'HLT_PFHT' in itrig]
        #AK8triggers = [itrig for itrig in Dataset_info if 'HLT_AK8PFHT' in itrig]

        # ---- Find numeric values in trigger strings ---- #
        #triggers_cut1 = [sub.split('PFHT')[1] for sub in triggers] # Remove string characters from left of number
        #triggers_cut2 = [sub.split('_')[0] for sub in triggers_cut1] # Remove string characters from right of number
        #isTriggerValue = [val.isnumeric() for val in triggers_cut2] # Boolean -> if string is only a number
        #triggers_cut2 = np.where(isTriggerValue, triggers_cut2, 0) # If string is not a number, replace with 0
        #triggers_vals = [int(val) for val in triggers_cut2] # Convert string numbers to integers
        
        #AK8triggers_cut1 = [sub.split('HT')[1] for sub in AK8triggers]
        #AK8triggers_cut2 = [sub.split('_')[0] for sub in AK8triggers_cut1]
        #isAK8TriggerValue = [val.isnumeric() for val in AK8triggers_cut2]
        #AK8triggers_cut2 = np.where(isAK8TriggerValue, AK8triggers_cut2, 0)
        #AK8triggers_vals = [int(val) for val in AK8triggers_cut2]
        
        # ---- Find Largest and Second Largest Value ---- #
        #triggers_vals.sort(reverse = True)
        #AK8triggers_vals.sort(reverse = True)
        
        #triggers_vals1 = str(triggers_vals[0])
        #triggers_vals2 = str(triggers_vals[1])
        #AK8triggers_vals1 = str(AK8triggers_vals[0])
        #AK8triggers_vals2 = str(AK8triggers_vals[1])
        
        # ---- Define strings for the selected triggers ---- #
        #HLT_trig1_str = [itrig for itrig in triggers if (triggers_vals1) in itrig][0]
        #HLT_trig2_str = [itrig for itrig in triggers if (triggers_vals2) in itrig][0]
        #HLT_AK8_trig1_str = [itrig for itrig in AK8triggers if (AK8triggers_vals1) in itrig][0]
        #HLT_AK8_trig2_str = [itrig for itrig in AK8triggers if (AK8triggers_vals2) in itrig][0]
        
        # ---- Define HLT triggers to be used ---- #
        #HLT_trig1 = df[HLT_trig1_str]
        #HLT_trig2 = df[HLT_trig2_str]
        #HLT_AK8_trig1 = df[HLT_AK8_trig1_str]
        #HLT_AK8_trig2 = df[HLT_AK8_trig2_str]
       
        
        # ---- Define AK8 Jets as FatJets ---- #
        FatJets = JaggedCandidateArray.candidatesfromcounts(
            df['nFatJet'],
            pt=df['FatJet_pt'],
            eta=df['FatJet_eta'],
            phi=df['FatJet_phi'],
            mass=df['FatJet_mass'],
            area=df['FatJet_area'],
            msoftdrop=df['FatJet_msoftdrop'],
            jetId=df['FatJet_jetId'],
            tau1=df['FatJet_tau1'],
            tau2=df['FatJet_tau2'],
            tau3=df['FatJet_tau3'],
            tau4=df['FatJet_tau4'],
            n3b1=df['FatJet_n3b1'],
            btagDeepB=df['FatJet_btagDeepB'],
            btagCSVV2=df['FatJet_btagCSVV2'],
            deepTag_TvsQCD=df['FatJet_deepTag_TvsQCD'],
            deepTagMD_TvsQCD=df['FatJet_deepTagMD_TvsQCD'],
            subJetIdx1=df['FatJet_subJetIdx1'],
            subJetIdx2=df['FatJet_subJetIdx2']
            )
        
        # ---- Define AK4 jets as Jets ---- #
        Jets = JaggedCandidateArray.candidatesfromcounts(
            df['nJet'],
            pt=df['Jet_pt'],
            eta=df['Jet_eta'],
            phi=df['Jet_phi'],
            mass=df['Jet_mass'],
            area=df['Jet_area']
            )
        # ---- Define SubJets ---- #
        SubJets = JaggedCandidateArray.candidatesfromcounts(
            df['nSubJet'],
            pt=df['SubJet_pt'],
            eta=df['SubJet_eta'],
            phi=df['SubJet_phi'],
            mass=df['SubJet_mass'],
            btagDeepB=df['SubJet_btagDeepB'],
            btagCSVV2=df['SubJet_btagCSVV2']
            )
        
        
        # ---- Get event weights from dataset ---- #
        if 'JetHT' in dataset: # If data is used...
            evtweights = np.ones(FatJets.size) # set all "data weights" to one
        else: # if Monte Carlo dataset is used...
            evtweights = df["Generator_weight"].reshape(-1, 1).flatten()
        
        # ---- Show all events ---- #
        output['cutflow']['all events'] += FatJets.size
        
        # ---- Apply Trigger(s) ---- #
        #FatJets = FatJets[HLT_AK8_trig1]
        #evtweights = evtweights[HLT_AK8_trig1]
        #Jets = Jets[HLT_AK8_trig1]
        #SubJets = SubJets[HLT_AK8_trig1]
        
        # ---- Jets that satisfy Jet ID ---- #
        jet_id = (FatJets.jetId > 0) # Loose jet ID
        FatJets = FatJets[jet_id]
        output['cutflow']['jet id'] += jet_id.any().sum()
        
        # ---- Apply pT Cut and Rapidity Window ---- #
        jetkincut_index = (FatJets.pt > self.ak8PtMin) & (np.abs(FatJets.p4.rapidity) < 2.4)
        FatJets = FatJets[ jetkincut_index ]
        output['cutflow']['jet kin'] += jetkincut_index.any().sum()
        
        # ---- Find two AK8 Jets ---- #
        twoFatJetsKin = (FatJets.counts == 2)
        FatJets = FatJets[twoFatJetsKin]
        evtweights = evtweights[twoFatJetsKin]
        Jets = Jets[twoFatJetsKin]
        SubJets = SubJets[twoFatJetsKin]
        output['cutflow']['two FatJets and jet kin'] += twoFatJetsKin.sum()
        
        # ---- Apply HT Cut ---- #
        hT = Jets.pt.sum()
        passhT = (hT > self.htCut)
        evtweights = evtweights[passhT]
        FatJets = FatJets[passhT]
        SubJets = SubJets[passhT]
        
        # ---- Randomly Assign AK8 Jets as TTbar Candidates 0 and 1 --- #
        if self.RandomDebugMode == True: # 'Sudo' randomizer for consistent results
            highPhi = FatJets.phi[:,0] > FatJets.phi[:,1]
            highRandIndex = np.where(highPhi, 0, 1)
            index = JaggedArray.fromcounts(np.ones(len(FatJets), dtype='i'), highRandIndex )
        else: # Truly randomize
            index = JaggedArray.fromcounts(np.ones(len(FatJets), dtype='i'), self.prng.randint(2, size=len(FatJets)))
        jet0 = FatJets[index] #J0
        jet1 = FatJets[1 - index] #J1
        
        ttbarcands = jet0.cross(jet1) #FatJets[:,0:2].distincts()
    
        # ---- Make sure we have at least 1 TTbar candidate pair and re-broadcast releveant arrays  ---- #
        oneTTbar = (ttbarcands.counts >= 1)
        output['cutflow']['>= one oneTTbar'] += oneTTbar.sum()
        ttbarcands = ttbarcands[oneTTbar]
        evtweights = evtweights[oneTTbar]
        FatJets = FatJets[oneTTbar]
        SubJets = SubJets[oneTTbar]
         
        # ---- Apply Delta Phi Cut for Back to Back Topology ---- #
        dPhiCut = (ttbarcands.i0.p4.delta_phi(ttbarcands.i1.p4) > 2.1).flatten()
        output['cutflow']['dPhi > 2.1'] += dPhiCut.sum()
        ttbarcands = ttbarcands[dPhiCut]
        evtweights = evtweights[dPhiCut]
        FatJets = FatJets[dPhiCut] 
        SubJets = SubJets[dPhiCut] 
        
        # ---- Identify subjets according to subjet ID ---- #
        hasSubjets0 = ((ttbarcands.i0.subJetIdx1 > -1) & (ttbarcands.i0.subJetIdx2 > -1))
        hasSubjets1 = ((ttbarcands.i1.subJetIdx1 > -1) & (ttbarcands.i1.subJetIdx2 > -1))
        GoodSubjets = ((hasSubjets0) & (hasSubjets1)).flatten()
   
        ttbarcands = ttbarcands[GoodSubjets]
        
        SubJets = SubJets[GoodSubjets]
        evtweights = evtweights[GoodSubjets]
       
        SubJet01 = SubJets[ttbarcands.i0.subJetIdx1] # FatJet i0 with subjet 1
        SubJet02 = SubJets[ttbarcands.i0.subJetIdx2] # FatJet i0 with subjet 2
        SubJet11 = SubJets[ttbarcands.i1.subJetIdx1] # FatJet i1 with subjet 1
        SubJet12 = SubJets[ttbarcands.i1.subJetIdx2] # FatJet i1 with subjet 2
        
        # ---- Define Rapidity Regions ---- #
        cen = np.abs(ttbarcands.i0.p4.rapidity - ttbarcands.i1.p4.rapidity) < 1.0
        fwd = (~cen)
        
        # ---- CMS Top Tagger Version 2 (SD and Tau32 Cuts) ---- #
        tau32_i0 = np.where(ttbarcands.i0.tau2>0,ttbarcands.i0.tau3/ttbarcands.i0.tau2, 0 )
        tau32_i1 = np.where(ttbarcands.i1.tau2>0,ttbarcands.i1.tau3/ttbarcands.i1.tau2, 0 )
        taucut_i0 = tau32_i0 < self.tau32Cut
        taucut_i1 = tau32_i1 < self.tau32Cut
        mcut_i0 = (self.minMSD < ttbarcands.i0.msoftdrop) & (ttbarcands.i0.msoftdrop < self.maxMSD) 
        mcut_i1 = (self.minMSD < ttbarcands.i1.msoftdrop) & (ttbarcands.i1.msoftdrop < self.maxMSD) 

        ttag_i0 = (taucut_i0) & (mcut_i0)
        ttag_i1 = (taucut_i1) & (mcut_i1)
        
        # ---- Define "Top Tag" Regions ---- #
        antitag = (~taucut_i0) & (mcut_i0) #Probe will always be ttbarcands.i1 (at)
        antitag_probe = np.logical_and(antitag, ttag_i1) #Found an antitag and ttagged probe pair for mistag rate (Pt)
        pretag =  ttag_i0 # Only jet0 (pret)
        ttag0 =   (~ttag_i0) & (~ttag_i1) # No tops tagged (0t)
        ttag1 =   ttag_i0 ^ ttag_i1 # Exclusively one top tagged (1t)
        ttagI =   ttag_i0 | ttag_i1 # At least one top tagged ('I' for 'inclusive' tagger; >=1t; 1t+2t)
        ttag2 =   ttag_i0 & ttag_i1 # Both jets top tagged (2t)
        Alltags = ttag0 | ttagI #Either no tag or at least one tag (0t+1t+2t)
        
        # ---- Pick FatJet that passes btag cut based on its subjet with the highest btag value ---- # 
        btag_i0 = ( np.maximum(SubJet01.btagCSVV2 , SubJet02.btagCSVV2) > self.bdisc )
        btag_i1 = ( np.maximum(SubJet11.btagCSVV2 , SubJet12.btagCSVV2) > self.bdisc )
        
        # --- Define "B Tag" Regions ---- #
        btag0 = (~btag_i0) & (~btag_i1) #(0b)
        btag1 = btag_i0 ^ btag_i1 #(1b)
        btag2 = btag_i0 & btag_i1 #(2b)
        
        # ---- Get Analysis Categories ---- # 
        # ---- They are (central, forward) cross (0b,1b,2b) cross (At,at,0t,1t,>=1t,2t) ---- #
        regs = [cen,fwd]
        btags = [btag0,btag1,btag2]
        ttags = [antitag_probe,antitag,pretag,ttag0,ttag1,ttagI,ttag2,Alltags]
        cats = [ (t&b&y).flatten() for t,b,y in itertools.product( ttags, btags, regs) ]
        labels_and_categories = dict(zip( self.anacats, cats ))
        
        # ---- Variables for Kinematic Histograms ---- #
        # ---- "i0" is the control jet, "i1" is the probe jet ---- #
        ttbarmass = ttbarcands.p4.sum().mass.flatten()
        jetpt = ttbarcands.i1.pt.flatten()
        jeteta = ttbarcands.i1.eta.flatten()
        jetphi = ttbarcands.i1.phi.flatten()
        jety = ttbarcands.i1.p4.rapidity.flatten()
        jetmass = ttbarcands.i1.p4.mass.flatten()
        SDmass = ttbarcands.i1.msoftdrop.flatten()
        jetdy = np.abs(ttbarcands.i0.p4.rapidity.flatten() - ttbarcands.i1.p4.rapidity.flatten())
        Tau32 = (ttbarcands.i1.tau3/ttbarcands.i1.tau2).flatten()
        # ---- Variables for Deep Tagger Analysis ---- #
        deepTag = ttbarcands.i1.deepTag_TvsQCD.flatten()
        deepTagMD = ttbarcands.i1.deepTagMD_TvsQCD.flatten()
        
        weights = evtweights.flatten()
        
        # ---- Define the SumW2 for MC Datasets ---- #
        output['cutflow']['sumw'] += np.sum(weights)
        output['cutflow']['sumw2'] += np.sum(weights**2)
        
        # ---- Define Momentum p of probe jet as the Mistag Rate variable; M(p) ---- #
        # ---- Transverse Momentum pT can also be used instead; M(pT) ---- #
        pT = ttbarcands.i1.pt.flatten()
        eta = ttbarcands.i1.eta.flatten()
        pz = np.sinh(eta)*pT
        p = np.absolute(np.sqrt(pT**2 + pz**2))
        
        # ---- Define the Numerator and Denominator for Mistag Rate ---- #
        numerator = np.where(antitag_probe, p, -1) # If no antitag and tagged probe, move event to useless bin
        denominator = np.where(antitag, p, -1) # If no antitag, move event to useless bin
        
        df = pd.DataFrame({"momentum":p}) # Used for finding values in LookUp Tables
        
        for ilabel,icat in labels_and_categories.items():
            ### ------------------------------------ Mistag Scaling ------------------------------------ ###
            if self.UseLookUpTables == True:
                # ---- Weight ttbar M.C. and data by mistag from data (corresponding to its year) ---- #
                if 'TTbar_' in dataset:
                    file_df = self.lu['JetHT' + dataset[-4:] + '_Data']['at' + str(ilabel[-5:])] #Pick out proper JetHT year mistag for TTbar sim.
                elif dataset == 'TTbar':
                    file_df = self.lu['JetHT']['at' + str(ilabel[-5:])] # All JetHT years mistag for TTbar sim.
                else:
                    file_df = self.lu[dataset]['at' + str(ilabel[-5:])] # get mistag (lookup) filename for 'at'
                
                bin_widths = file_df['p'].values # collect bins as written in .csv file
                mtr = file_df['M(p)'].values # collect mistag rate as function of p as written in file
                wgts = mtr # Define weights based on mistag rates
                
                BinKeys = np.arange(bin_widths.size) # Use as label for BinNumber column in the new dataframe
                
                #Bins = pd.interval_range(start=0, periods=100, freq=100, closed='left') # Recreate the momentum bins from file_df as something readable for pd.cut()
                Bins = np.array(manual_bins)
                
                df['BinWidth'] = pd.cut(p, bins=Bins) # new dataframe column
                df['BinNumber'] = pd.cut(p, bins=Bins, labels=BinKeys)
                
                BinNumber = df['BinNumber'].values # Collect the Bin Numbers into a numpy array
                BinNumber = BinNumber.astype('int64') # Insures the bin numbers are integers
            
                WeightMatching = wgts[BinNumber] # Match 'wgts' with corresponding p bin using the bin number
                Weights = weights*WeightMatching # Include 'wgts' with the previously defined 'weights'
            else:
                Weights = weights # No mistag rates, no change to weights
            ###---------------------------------------------------------------------------------------------###
            ### ----------------------------------- Mod-mass Procedure ------------------------------------ ###
            if self.ModMass == True:
                QCD_unweighted = util.load('TTbarResCoffea_QCD_unweighted_output.coffea')
    
                # ---- Extract event counts from QCD MC hist in signal region ---- #
                QCD_hist = QCD_unweighted['jetmass'].integrate('anacat', '2t' + str(ilabel[-5:])).integrate('dataset', 'QCD')
                data = QCD_hist.values() # Dictionary of values
                QCD_data = [i for i in data.values()][0] # place every element of the dictionary into a numpy array

                # ---- Re-create Bins from QCD_hist as Numpy Array ---- #
                bins = np.arange(510) #Re-make bins from the jetmass_axis starting with the appropriate range
                QCD_bins = bins[::10] #Finish re-making bins by insuring exactly 50 bins like the jetmass_axis

                # ---- Define Mod Mass Distribution ---- #
                ModMass_hist_dist = ss.rv_histogram([QCD_data,QCD_bins])
                jet1_modp4 = copy.copy(jet1.p4) #J1's Lorentz four vector that can be safely modified
                jet1_modp4["fMass"] = ModMass_hist_dist.rvs(size=jet1_modp4.size) #Replace J1's mass with random value of mass from mm hist
                ttbarcands_modmass = jet0.p4.cross(jet1_modp4) #J0's four vector x modified J1's four vector

                # ---- Apply Necessary Selections to new modmass version ---- #
                ttbarcands_modmass = ttbarcands_modmass[oneTTbar]
                ttbarcands_modmass = ttbarcands_modmass[dPhiCut]
                ttbarcands_modmass = ttbarcands_modmass[GoodSubjets]
                
                # ---- Manually sum the modmass p4 candidates (Coffea technicality) ---- #
                ttbarcands_modmass_p4_sum = (ttbarcands_modmass.i0 + ttbarcands_modmass.i1)
                
                # ---- Re-define Mass Variables for ModMass Procedure (pt, eta, phi are redundant to change) ---- #
                ttbarmass = ttbarcands_modmass_p4_sum.flatten().mass
                jetmass = ttbarcands_modmass.i1.mass.flatten()
            ###---------------------------------------------------------------------------------------------###
            output['cutflow'][ilabel] += np.sum(icat)
          
            output['ttbarmass'].fill(dataset=dataset, anacat=ilabel, 
                                ttbarmass=ttbarmass[icat],
                                weight=Weights[icat])
            output['jetpt'].fill(dataset=dataset, anacat=ilabel, 
                                jetpt=jetpt[icat],
                                weight=Weights[icat])
            output['probept'].fill(dataset=dataset, anacat=ilabel, 
                                jetpt=pT[icat],
                                weight=Weights[icat])
            output['probep'].fill(dataset=dataset, anacat=ilabel, 
                                jetp=p[icat],
                                weight=Weights[icat])
            output['jeteta'].fill(dataset=dataset, anacat=ilabel, 
                                jeteta=jeteta[icat],
                                weight=Weights[icat])
            output['jetphi'].fill(dataset=dataset, anacat=ilabel, 
                                jetphi=jetphi[icat],
                                weight=Weights[icat])
            output['jety'].fill(dataset=dataset, anacat=ilabel, 
                                jety=jety[icat],
                                weight=Weights[icat])
            output['jetdy'].fill(dataset=dataset, anacat=ilabel, 
                                jetdy=jetdy[icat],
                                weight=Weights[icat])
            output['numerator'].fill(dataset=dataset, anacat=ilabel, 
                                jetp=numerator[icat],
                                weight=Weights[icat])
            output['denominator'].fill(dataset=dataset, anacat=ilabel, 
                                jetp=denominator[icat],
                                weight=Weights[icat])
            output['jetmass'].fill(dataset=dataset, anacat=ilabel, 
                                   jetmass=jetmass[icat],
                                   weight=Weights[icat])
            output['SDmass'].fill(dataset=dataset, anacat=ilabel, 
                                   jetmass=SDmass[icat],
                                   weight=Weights[icat])
            output['tau32'].fill(dataset=dataset, anacat=ilabel,
                                          tau32=Tau32[icat],
                                          weight=Weights[icat])
            output['tau32_2D'].fill(dataset=dataset, anacat=ilabel,
                                          jetpt=pT[icat],
                                          tau32=Tau32[icat],
                                          weight=Weights[icat])
            output['deepTag_TvsQCD'].fill(dataset=dataset, anacat=ilabel,
                                          jetpt=pT[icat],
                                          tagger=deepTag[icat],
                                          weight=Weights[icat])
            output['deepTagMD_TvsQCD'].fill(dataset=dataset, anacat=ilabel,
                                            jetpt=pT[icat],
                                            tagger=deepTagMD[icat],
                                            weight=Weights[icat])
        
        return output

    def postprocess(self, accumulator):
        return accumulator


# In[ ]:


#get_ipython().system('jupyter nbconvert --to script TTbarResProcessor.ipynb')

