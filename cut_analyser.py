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


clusters, n_events = read_root_file_to_clusters('/afs/cern.ch/work/h/hakins/private/root_cluster_files/es-cc-bkg-truth/X/bkg/X/clusters_tick_limits_3_channel_limits_1_min_tps_to_cluster_1_cut_50000.root') #all backgrounds, 8,000,000 clusters

cuts = [0,25000,50000,55000, 60000, 65000, 70000, 80000, 90000,100000,110000, 120000,130000, 140000, 150000, 160000,170000, 180000, 225000, 250000, 275000, 300000, 325000, 
        350000, 400000, 500000, 600000, 700000]
cuts_dict = {cut: [] for cut in cuts}  #list of adc charge per cluster for all cuts


for cluster in clusters:
    total_charge_sum = 0
    #summing ADC integral for each TP set in cluster
    for tp in cluster.tps_:
        total_charge_sum = total_charge_sum + tp['adc_integral']
        
    for cut in cuts:
        if total_charge_sum > cut:
            cuts_dict[cut].append(total_charge_sum)

total_clusters = 8493526
cluster_sizes = []
cluster_sizes.append(total_clusters) #size of zero cut manuelly cause file takes too long to cluster 8 million
cluster_sizes.append(1013059) #25000 cut
cluster_fraction = []
cluster_fraction.append(1) #size of zero cut manuelly cause file takes too long to cluster 8 million
cluster_fraction.append(1013059/total_clusters)
for cut in cuts_dict:
    if cut != 0 and cut != 25000:
        cluster_sizes.append(len(cuts_dict[cut]))
        cluster_fraction.append(len(cuts_dict[cut])/total_clusters)
        print(f"Cut: {cut} Size: {len(cuts_dict[cut])}")
        print(f'Fraction of Total: {len(cuts_dict[cut])/len(cuts_dict[0])}')

fig = plt.figure()
#Just Trues
plt.title("Number of Clusters vs Charge Cut")
plt.plot(cuts, cluster_sizes, linestyle='-', color='b', label='# of Clusters')
# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/bkg/clusters_vs_cuts.png')
plt.clf()
#log scale
plt.title("Number of Clusters vs Charge Cut")
plt.plot(cuts, cluster_sizes, linestyle='-', color='b', label='# of Clusters')
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

'''Cut: 0 Size: 8493526
Fraction of Total: 1.0
Cut: 25000 Size: 1013059
Fraction of Total: 0.11927425665147784
Cut: 50000 Size: 134966
Fraction of Total: 0.015890455860145716
Cut: 60000 Size: 116072
Fraction of Total: 0.013665938033273813
Cut: 70000 Size: 101818
Fraction of Total: 0.011987718645942804
Cut: 80000 Size: 89083
Fraction of Total: 0.010488341355521841
Cut: 100000 Size: 66490
Fraction of Total: 0.007828315354541801
Cut: 120000 Size: 50093
Fraction of Total: 0.005897786149121107
Cut: 140000 Size: 37175
Fraction of Total: 0.004376863036623424
Cut: 150000 Size: 31636
Fraction of Total: 0.003724719274421483
Cut: 160000 Size: 26625
Fraction of Total: 0.0031347405070638506
Cut: 180000 Size: 17813
Fraction of Total: 0.0020972444188667935
Cut: 225000 Size: 4810
Fraction of Total: 0.0005663136840930375
Cut: 250000 Size: 1735
Fraction of Total: 0.00020427323116453637
Cut: 275000 Size: 597
Fraction of Total: 7.028882939782606e-05
Cut: 300000 Size: 312
Fraction of Total: 3.673386058981865e-05
Cut: 325000 Size: 218
Fraction of Total: 2.5666607719809183e-05
Cut: 350000 Size: 167
Fraction of Total: 1.9662034354165748e-05
Cut: 400000 Size: 92
Fraction of Total: 1.0831779404690114e-05
Cut: 500000 Size: 25
Fraction of Total: 2.943418316491879e-06
Cut: 600000 Size: 4
Fraction of Total: 4.709469306387006e-07
Cut: 700000 Size: 2
Fraction of Total: 2.354734653193503e-07'''