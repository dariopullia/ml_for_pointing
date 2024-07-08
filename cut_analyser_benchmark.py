import re
import numpy as npy
import matplotlib.pyplot as plt
import numpy as np
#import ROOT
from scipy.stats import linregress
from scipy.optimize import curve_fit
import time
import os
import argparse
import warnings
import gc
from sklearn.metrics import r2_score
from scipy.special import gamma
from scipy.stats import chisquare
import random
import pickle
import os
from tensorflow.keras.models import load_model
import pandas as pd
import run_mt_id


import sys
sys.path.append('../python/') 
sys.path.append("/afs/cern.ch/work/h/hakins/private/online-pointing-utils/python")
from cluster import read_root_file_to_clusters

# from utils import save_tps_array, create_channel_map_array
# from hdf5_converter import convert_tpstream_to_numpy 
# from image_creator import save_image, show_image
# from cluster_maker import make_clusters


clusters, n_events = read_root_file_to_clusters('/afs/cern.ch/work/h/hakins/private/root_cluster_files/benchmark/X/clusters_tick_limits_3_channel_limits_1_min_tps_to_cluster_1_cut_0.root') 
cuts = [50000,55000, 60000, 65000, 70000, 80000, 90000,100000,110000, 120000,130000, 140000, 150000, 160000,170000, 180000, 225000, 250000, 275000, 300000, 325000, 
        350000, 400000, 500000, 600000, 700000]
cuts_dict_bkg = {cut: [] for cut in cuts}#list of adc charge per cluster for all cuts
cuts_dict_SN = {cut: [] for cut in cuts}
cuts_dict_blips = {cut: [] for cut in cuts}

SN = 1

for cluster in clusters:
    if cluster.true_label_ == SN:
        total_charge_sum_SN = 0
        for tp in cluster.tps_:
            total_charge_sum_SN += tp['adc_integral']
        for cut in cuts:
            if total_charge_sum_SN > cut:
                cuts_dict_SN[cut].append(total_charge_sum_SN)
        
    else:
        total_charge_sum_bkg = 0
        for tp in cluster.tps_:
            total_charge_sum_bkg += tp['adc_integral']
        for cut in cuts:
            if total_charge_sum_bkg > cut:
                cuts_dict_bkg[cut].append(total_charge_sum_bkg)
    
                


cluster_sizes_bkg = []
cluster_sizes_SN = []
cluster_fraction = []

total_clusters_bkg = len(cuts_dict_bkg[0])
print(total_clusters_bkg)

# Loop through cuts_dict_bkg
for cut_bkg in cuts_dict_bkg:
    cluster_sizes_bkg.append(len(cuts_dict_bkg[cut_bkg]))
    cluster_fraction.append(len(cuts_dict_bkg[cut_bkg]) / total_clusters_bkg)
    print(f"Cut: {cut_bkg} Size: {len(cuts_dict_bkg[cut_bkg])}")
    print(f'Fraction of Total: {len(cuts_dict_bkg[cut_bkg]) / len(cuts_dict_bkg[0])}')

# Loop through cuts_dict_mt
for cut_SN in cuts_dict_SN:
    cluster_sizes_SN.append(len(cuts_dict_SN[cut_SN]))
    print(f"Cut: {cut_SN} Size: {len(cuts_dict_SN[cut_SN])}")
    

    

fig = plt.figure()
#Just Trues
plt.title("Number of Clusters vs Charge Cut")
plt.plot(cuts, cluster_sizes_bkg, linestyle='-', color='b', label='BKG Clusters')
plt.plot(cuts, cluster_sizes_SN, linestyle='-', color='g', label='SN Clusters')

# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/bkg/clusters_vs_cuts_benchmark.png')
plt.clf()
#log scale
plt.title("Number of Clusters vs Charge Cut")
plt.plot(cuts, cluster_sizes_bkg, linestyle='-', color='b', label='BKG Clusters')
plt.plot(cuts, cluster_sizes_SN, linestyle='-', color='g', label='SN Clusters')
# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters [log]')
plt.legend()
plt.yscale("log")
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/bkg/clusters_vs_cuts_logScale_benchmark.png')
plt.clf()





#MODEL ANALYSIS










exit()
#Fraction
plt.title("Fraction of Background Cut vs Charge Cut")
plt.plot(cuts, cluster_fraction, linestyle='-', color='b', label='# of Clusters')
# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/bkg/clustersFraction_vs_cuts.png')
plt.clf()

