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

import sys
sys.path.append('../python/') 
sys.path.append("/afs/cern.ch/work/h/hakins/private/online-pointing-utils/python")
from cluster import read_root_file_to_clusters

# from utils import save_tps_array, create_channel_map_array
# from hdf5_converter import convert_tpstream_to_numpy 
# from image_creator import save_image, show_image
# from cluster_maker import make_clusters


clusters_bkg, n_events = read_root_file_to_clusters('/afs/cern.ch/work/h/hakins/private/root_cluster_files/es-cc-bkg-truth/X/bkg/X/clusters_tick_limits_3_channel_limits_1_min_tps_to_cluster_1_cut_50000.root') 
clusters_mt, n_events = read_root_file_to_clusters('/afs/cern.ch/work/h/hakins/private/root_cluster_files/es-cc_lab/main_tracks/X/clusters_tick_limits_3_channel_limits_1_min_tps_to_cluster_1_cut_0.root') 
clusters_blips, n_events = read_root_file_to_clusters('/afs/cern.ch/work/h/hakins/private/root_cluster_files/es-cc_lab/blips/X/clusters_tick_limits_3_channel_limits_1_min_tps_to_cluster_1_cut_0.root') 

cuts = [0,25000,50000,55000, 60000, 65000, 70000, 80000, 90000,100000,110000, 120000,130000, 140000, 150000, 160000,170000, 180000, 225000, 250000, 275000, 300000, 325000, 
        350000, 400000, 500000, 600000, 700000]
cuts_dict_bkg = {cut: [] for cut in cuts}#list of adc charge per cluster for all cuts
cuts_dict_mt = {cut: [] for cut in cuts}
cuts_dict_blips = {cut: [] for cut in cuts}



for bkg_cluster in clusters_bkg:
    total_charge_sum_bkg = 0
    for tp_bkg in bkg_cluster.tps_:
        total_charge_sum_bkg += tp_bkg['adc_integral']
        
    for cut in cuts:
        if total_charge_sum_bkg > cut:
            cuts_dict_bkg[cut].append(total_charge_sum_bkg)

for mt_cluster in clusters_mt:
    total_charge_sum_mt = 0

    for tp_mt in mt_cluster.tps_:
        total_charge_sum_mt += tp_mt['adc_integral']
        
    for cut in cuts:
        if total_charge_sum_mt > cut:
            cuts_dict_mt[cut].append(total_charge_sum_mt)

for blip_cluster in clusters_blips:
    total_charge_sum_blip = 0

    for tp_blip in blip_cluster.tps_:
        total_charge_sum_blip += tp_blip['adc_integral']
        
    for cut in cuts:
        if total_charge_sum_blip > cut:
            cuts_dict_blips[cut].append(total_charge_sum_blip)


total_clusters_bkg = 8493526
cluster_sizes_bkg = []
cluster_sizes_mt = []
cluster_sizes_blips = []
cluster_sizes_bkg.append(total_clusters_bkg) #size of zero cut manuelly cause file takes too long to cluster 8 million
cluster_sizes_bkg.append(1013059) #25000 cut
cluster_fraction = []
cluster_fraction.append(1) #size of zero cut manuelly cause file takes too long to cluster 8 million
cluster_fraction.append(1013059/total_clusters_bkg)




# Loop through cuts_dict_bkg
for cut_bkg in cuts_dict_bkg:
    if cut_bkg != 0 and cut_bkg != 25000:
        cluster_sizes_bkg.append(len(cuts_dict_bkg[cut_bkg]))
        cluster_fraction.append(len(cuts_dict_bkg[cut_bkg]) / total_clusters_bkg)
        print(f"Cut: {cut_bkg} Size: {len(cuts_dict_bkg[cut_bkg])}")
        print(f'Fraction of Total: {len(cuts_dict_bkg[cut_bkg]) / len(cuts_dict_bkg[0])}')

# Loop through cuts_dict_mt
for cut_mt in cuts_dict_mt:
    cluster_sizes_mt.append(len(cuts_dict_mt[cut_mt]))
    print(f"Cut: {cut_mt} Size: {len(cuts_dict_mt[cut_mt])}")

# Loop through cuts_dict_blips
for cut_blip in cuts_dict_blips:
    cluster_sizes_blips.append(len(cuts_dict_blips[cut_blip]))
    print(f"Cut: {cut_blip} Size: {len(cuts_dict_blips[cut_blip])}")

fig = plt.figure()
#Just Trues
plt.title("Number of Clusters vs Charge Cut")
plt.plot(cuts, cluster_sizes_bkg, linestyle='-', color='b', label='BKG Clusters')
plt.plot(cuts, cluster_sizes_mt, linestyle='-', color='g', label='Main Track Clusters')
plt.plot(cuts, cluster_sizes_blips, linestyle='-', color='r', label='Blip Clusters')

# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/bkg/clusters_vs_cuts.png')
plt.clf()
#log scale
plt.title("Number of Clusters vs Charge Cut")
plt.plot(cuts, cluster_sizes_bkg, linestyle='-', color='b', label='BKG Clusters')
plt.plot(cuts, cluster_sizes_mt, linestyle='-', color='g', label='Main Track Clusters')
plt.plot(cuts, cluster_sizes_blips, linestyle='-', color='r', label='Blip Clusters')
plt.axhline(y=100, color='r', linestyle='--', linewidth=0.3)
# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters [log]')
plt.legend()
plt.yscale("log")
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/bkg/clusters_vs_cuts_logScale.png')
plt.clf()

#Fraction
plt.title("Fraction of Background Cut vs Charge Cut")
plt.plot(cuts, cluster_fraction, linestyle='-', color='b', label='# of Clusters')
# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/bkg/clustersFraction_vs_cuts.png')
plt.clf()

