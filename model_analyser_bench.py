
import re
import numpy as npy
import matplotlib.pyplot as plt
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
import sys

sys.path.append("/afs/cern.ch/work/h/hakins/private/data-selection-pipeline/python/")

import run_mt_id
sys.path.append('../python/') 
sys.path.append("/afs/cern.ch/work/h/hakins/private/online-pointing-utils/python")

is_benchmark = True
cuts = [50000, 60000, 70000, 80000, 100000, 120000, 140000, 150000, 160000, 180000, 225000, 250000, 275000, 300000, 325000, 
        350000, 400000, 500000]

plot_cuts = []
true_pos=[]
true_neg = []
false_pos = []
false_neg = []
f1s = []
count = 0

cut_counts = 0
for cut in cuts:
    try:
        if not is_benchmark:
            with open(f"/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/{cut}/hyperopt_simple_cnn/aug_coeff_1/metrics.txt",'r') as file:
                lines = file.readlines()
        elif is_benchmark:
            with open(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/{cut}mt_id/metrics.txt",'r') as file:
                lines = file.readlines()
        
    
        matrix_lines = []
        f1 = []
        
        if len(lines) > 8:
            matrix_lines.append(lines[8])
            matrix_lines.append(lines[9])
            f1.append(lines[13])
        else:
            matrix_lines.append(lines[1])
            matrix_lines.append(lines[2])
            f1.append(lines[6])
        # Extract numbers from the collected lines
        matrix_str = ''.join(matrix_lines)
        matrix_str = matrix_str.replace('][', ' ').replace(']', '').replace('[','')
        matrix_values = [float(num) for num in re.split(r'\s+', matrix_str) if num]
        f1_str = ''.join(f1)
        f1_str = f1_str.replace('F1:','').replace('\n','')
        f1_float = float(f1_str)
        true_pos.append(matrix_values[3])
        true_neg.append(matrix_values[0])
        false_pos.append(matrix_values[1])
        false_neg.append(matrix_values[2])
        plot_cuts.append(cut)
        f1s.append(f1_float)
        
    except FileNotFoundError as e:
        print(f"File not Found for cut {cut}")
        cut_counts+=1




print(f'{int((1-(cut_counts/len(cuts)))*100)}% of cuts were succesful')
fig = plt.figure()
#Just Trues
plt.title("Confusion Matrix Values across Different Cuts")
plt.plot(plot_cuts, true_pos, marker='o', linestyle='-', color='b', label='True Positives')
plt.plot(plot_cuts, true_neg, marker='o', linestyle='-', color='g', label='True Negatives')
# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()

# Displaying the plot
if is_benchmark:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion_pos.png')
else:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/models_confusion_pos.png')

plt.clf()

#Just false
plt.title("Confusion Matrix Values across Different Cuts")
plt.plot(plot_cuts, false_pos, marker='o', linestyle='-', color='r', label='False Positives')
plt.plot(plot_cuts, false_neg, marker='o', linestyle='-', color='m', label='False Negatives')

plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
if is_benchmark:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion_neg.png')
else:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/models_confusion_neg.png')
plt.clf()

#true and false
plt.title("Confusion Matrix Values across Different Cuts")
plt.plot(plot_cuts, true_pos, marker='o', linestyle='-', color='b', label='True Positives')
plt.plot(plot_cuts, true_neg, marker='o', linestyle='-', color='g', label='True Negatives')
plt.plot(plot_cuts, false_pos, marker='o', linestyle='-', color='r', label='False Positives')
plt.plot(plot_cuts, false_neg, marker='o', linestyle='-', color='m', label='False Negatives')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
if is_benchmark:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion.png')
else:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/models_confusion.png')



plt.clf()
plt.title("F1 Score across Different Cuts")
plt.plot(plot_cuts, f1s, marker='o', linestyle='-', color='m', label='F1 Score')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
if is_benchmark:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/F1_scores.png')
else:
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/F1_scores.png')
plt.clf()
    
    
#Analyse data and models


'''cuts_dict_bkg = {cut: [] for cut in cuts}#list of adc charge per cluster for all cuts
cuts_dict_mt = {cut: [] for cut in cuts}
cuts_dict_blips = {cut: [] for cut in cuts}
true_labels = []
mt_total = 0
bkg_total = 0

for cluster in clusters:
    true_labels.append(cluster.true_label_)
    if cluster.true_label_ == 100 or cluster.true_label_ == 101:
        mt_total+=1
        total_charge_sum_mt = 0
        for tp in cluster.tps_:
            total_charge_sum_mt += tp['adc_integral']
        for cut in cuts:
            if total_charge_sum_mt > cut:
                cuts_dict_mt[cut].append(total_charge_sum_mt)
    else:
        bkg_total+=1
        total_charge_sum_bkg = 0
        for tp in cluster.tps_:
            total_charge_sum_bkg += tp['adc_integral']
        for cut in cuts:
            if total_charge_sum_bkg > cut:
                cuts_dict_bkg[cut].append(total_charge_sum_bkg)'''
    
                

#101 es
#100 cc

cluster_sizes_bkg = []
cluster_sizes_mt = []
cluster_fraction = []

'''print(f'mt total: {mt_total}')
total_clusters_bkg = bkg_total
print(total_clusters_bkg)

# Loop through cuts_dict_bkg
for cut_bkg in cuts_dict_bkg:
    cluster_sizes_bkg.append(len(cuts_dict_bkg[cut_bkg]))
    cluster_fraction.append(len(cuts_dict_bkg[cut_bkg]) / total_clusters_bkg)
    print(f"Cut: {cut_bkg} Size: {len(cuts_dict_bkg[cut_bkg])}")
    print(f'Fraction of Total: {len(cuts_dict_bkg[cut_bkg]) / bkg_total}')

# Loop through cuts_dict_mt
for cut_mt in cuts_dict_mt:
    cluster_sizes_mt.append(len(cuts_dict_mt[cut_mt]))
    print(f"Cut: {cut_mt} Size: {len(cuts_dict_mt[cut_mt])}")'''
    
true_positives = []
incorrect_mts = []
absolute_true_pos = []
absolute_true_neg = []
absolute_false_pos = []
absolute_false_neg = []


cluster_sizes_mt = [] #total mts in sample
identified_ccs = [] # correctly identified CC MTs
identified_es = [] # correctly identified ES MTs
misidentified_ccs = [] # CC MT identified as Bkg
misidentified_es = [] # ES MT identified as Bkg
false_positives=[] # BKG identified as MT
true_negatives = []
false_negatives = [] #MT identified as Bkg
bkgs = []
blips=[]
total_ccs=[]
totals_es=[]
threshold = .7
for cut in cuts:
     
    labels = np.load(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/{cut}ctds/dataset/dataset_label_process.npy")
    predictions = np.load(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/{cut}mt_id/predictions.npy")
    cc=0
    es=0
    cc_total=0
    es_total=0
    misidentified_mt=0
    blip=0
    bkg=0
    mt=0
    tn=0
    fn = 0

    mt_list  = [100,101]
    for i in range(len(labels)):
        if labels[i] == 100:
            cc_total+=1
        if labels[i] == 101:
            es_total+=1
        if labels[i] == 100 and predictions[i]>threshold:
            cc+=1
        if labels[i] == 101 and predictions[i] > threshold:
            es+=1
        if labels[i] not in mt_list  and predictions[i]>threshold:
            misidentified_mt+=1
        if (labels[i] != 1 and labels[i] not in mt_list) and predictions[i]<threshold :
            tn+=1
        if (labels[i] in mt_list) and predictions[i]<threshold :
            fn+=1
        if labels[i] == 1:
            blip+=1
        if labels[i] != 1 and labels[i] not in mt_list:
            bkg+=1
        if labels[i] in mt_list:
            mt+=1
            
    
    
    true_negatives.append(tn)
    false_negatives.append(fn)
    identified_ccs.append(cc)
    identified_es.append(es)
    false_positives.append(misidentified_mt) 
    blips.append(blip)
    bkgs.append(bkg)
    cluster_sizes_mt.append(mt)
    total_ccs.append(cc_total)
    totals_es.append(es_total)
    
    

    '''with open(f'/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/{cut}mt_id/prediction_results.txt', 'r') as file:
    # Skip the header line
        next(file)
        
        # Read the rest of the lines and convert to integers
        lines = file.readlines()
        tp = int(lines[0].strip())
        tn = int(lines[1].strip())
        fp = int(lines[2].strip())
        fn = int(lines[3].strip())
        all_images_count = int(lines[4].strip())
        identified_mts.append(tp)
        incorrect_mts.append(fp)
        cluster_sizes_mt.append(tp+fn)
    '''
    
'''    identified_mts.append(cluster_sizes_mt[i]*percent)
    incorrect_mts.append(cluster_sizes_bkg[i]*false_pos[i])
    absolute_true_pos.append((cluster_sizes_mt[i]*percent)/mt_total)
    absolute_false_pos.append((cluster_sizes_bkg[i]*false_pos[i]/bkg_total))
    absolute_true_neg.append((cluster_sizes_bkg[i]*true_neg[i])/bkg_total)
    absolute_false_neg.append((cluster_sizes_mt[i]*false_neg[i])/mt_total)
'''
    
plt.clf

'''plt.title("Confusion Matrix Values across Different Cuts")
plt.plot(cuts, absolute_true_pos, marker='o', linestyle='-', color='b', label='True Positives')
plt.plot(cuts, absolute_true_neg, marker='o', linestyle='-', color='g', label='True Negatives')
plt.plot(cuts, absolute_false_pos, marker='o', linestyle='-', color='r', label='False Positives')
plt.plot(cuts, absolute_false_neg, marker='o', linestyle='-', color='m', label='False Negatives')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion_absolute.png')
plt.clf()'''

total_ccs_adjusted = [] #to plot them above the ES counts
for i in range(len(cuts)):
    total_ccs_adjusted.append(total_ccs[i]+identified_es[i])
    
identified_es = np.array(identified_es)
totals_es = np.array(totals_es)
identified_ccs = np.array(identified_ccs)
total_ccs = np.array(total_ccs)
true_negatives = np.array(true_negatives)
false_negatives = np.array(false_negatives)
false_positives  = np.array(false_positives) # FPs
true_positives = identified_ccs+identified_es # TPs
true_positives_es = identified_es
#TP: Main Track identified as a Main Track
#FP: BKG identidied as a Main Track
#TN: BKG identified as a BKG
#FN: Main track identified as a BKG
#normalize
norm_true_positives = true_positives / cluster_sizes_mt[0]
norm_true_negatives = true_negatives /  (blips[0]+bkgs[0])
norm_false_positives = false_positives / (blips[0]+bkgs[0])
norm_false_negatives = false_negatives / cluster_sizes_mt[0] 

norm_true_positives_es = true_positives_es / totals_es[0]

#F1 Score
accuracy = (norm_true_positives+norm_true_negatives)/ (norm_true_negatives+norm_true_positives+norm_false_positives+norm_false_negatives)
recall = (norm_true_positives)/(norm_true_positives+norm_false_negatives)
precicion = norm_true_positives/(norm_true_positives+norm_false_positives)
f1_Score = 2 * (precicion*recall)/(precicion + recall)
#Confusion MAtrix
plt.title("Confusion Matrix Values across Different Cuts")
plt.plot(cuts,norm_true_positives, marker='o', linestyle='-', color='lightgreen', label='True Positive')
plt.plot(cuts, norm_true_negatives, marker='o', linestyle='-', color='lightblue', label='True Negative')
plt.plot(cuts, norm_false_positives, marker='o', linestyle='-', color='lightcoral', label='False Positive')
plt.plot(cuts, norm_false_negatives, marker='o', linestyle='-', color='m', label='False Negative')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion_absolute.png')
plt.clf()
#F1 Score
plt.title("Absolute F1 Score across Different Cuts")
plt.plot(cuts,f1_Score, marker='o', linestyle='-', color='lightcoral', label='True Positive')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/f1_absolute.png')
plt.clf()

#Confusion and F1 for JUST ES Events
plt.title("Confusion Matrix Values across Different Cuts")
plt.plot(cuts,norm_true_positives_es, marker='o', linestyle='-', color='b', label='ES True Positive')
plt.plot(cuts, norm_true_negatives, marker='o', linestyle='-', color='g', label='True Negative')
plt.plot(cuts, norm_false_positives, marker='o', linestyle='-', color='r', label='False Positive')
plt.plot(cuts, norm_false_negatives, marker='o', linestyle='-', color='m', label='False Negative')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion_absolute_ES.png')
plt.clf()





# Convert cuts to strings for labeling
cut_labels = [str(cut) for cut in cuts]
x = np.arange(len(cuts))
bar_width = 0.5
fig, ax = plt.subplots(figsize=(10, 6))  
bars2 = ax.bar(x, identified_es, bar_width,label='ES Identified as MT', color='lightgreen', edgecolor='lightgreen', alpha=0.7)
bars2_hatch = ax.bar(x, totals_es-identified_es,bar_width,bottom=identified_es,label="ES Identified as BKG", color='none', edgecolor='lightgreen',hatch='////',alpha=0.7,zorder=10)
bars1 = ax.bar(x, identified_ccs, bar_width,bottom=totals_es, label='CC Identified as MT', color='blue',edgecolor='blue', alpha=0.45)
bars1_hatch = ax.bar(x, total_ccs-identified_ccs, bar_width,bottom=(identified_ccs+totals_es), label='CC Idenfitiied as BKG', color='none', edgecolor='blue',hatch='////', alpha=0.45,zorder=10)
bars3 = ax.bar(x, false_positives, bar_width, bottom=cluster_sizes_mt, label='BKG+Blip Identified as MT', color='lightcoral',edgecolor='lightcoral', alpha=0.7)




#Overlay the line plot
#ax.plot(x, cluster_sizes_mt, linestyle='-', color='lightblue', marker='o', label='Total MTs')
'''for i in range(len(x)):
    ax.hlines(totals_es[i], x[i] - bar_width/2, x[i] + bar_width/2, colors='lightgreen', alpha=0.7,label="Total ES")
    ax.hlines(total_ccs_adjusted[i], x[i] - bar_width/2, x[i] + bar_width/2, colors='blue', alpha=0.45, label="Total CC (Relative)")
    
ax.plot(x, totals_es, linestyle='-', color='lightgreen', marker='o', label='Total ES')
ax.plot(x, total_ccs_adjusted, linestyle='-', color='blue', marker='o', alpha=0.45,label='Total CCs (Relative)')'''
ax.set_xlabel('Cuts')
ax.set_ylabel('Counts')
ax.set_title('Composition of Sample After MT Identifier at Each Cut')
ax.set_xticks(x)
ax.set_xticklabels(cut_labels, rotation=45, ha='right')  # Rotate x labels
ax.legend()
plt.tight_layout()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/MT_sample_composition.png')

# Isolating ES and CC events
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figsize for better readability
bars1 = ax.bar(x-bar_width/2, identified_ccs, bar_width, label='Identified CC MTs', color='blue',edgecolor='blue', alpha=0.8)
bars1_hatch = ax.bar(x-bar_width/2, total_ccs-identified_ccs, bar_width,bottom=(identified_ccs), label='CC Idenfitiied as BKG', color='none', edgecolor='blue',hatch='////', alpha=0.8,zorder=10)
bars2 = ax.bar(x+bar_width/2, identified_es, bar_width, label='Identified ES MTs', color='lightgreen',edgecolor='green',alpha=0.6)
bars2_hatch = ax.bar(x+bar_width/2, totals_es-identified_es,bar_width,bottom=identified_es,label="ES Identified as BKG", color='none', edgecolor='green',hatch='////',alpha=0.6,zorder=10)


ax.set_xlabel('Cuts')
ax.set_ylabel('Counts')
ax.set_title('Composition of Sample After MT Identifier at Each Cut')
ax.set_xticks(x)
ax.set_xticklabels(cut_labels, rotation=45, ha='right')  # Rotate x labels
ax.legend()
plt.tight_layout()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/MT_ESvCC_composition.png')


fig = plt.figure()
plt.title("Number of Clusters vs Charge Cut")
plt.plot(cuts, bkgs, linestyle='-', color='b', label='BKG')
plt.plot(cuts, blips, linestyle='-', color='g', label='Blips')
plt.plot(cuts,cluster_sizes_mt,linestyle='-', color='red', label='MTs')
plt.xlabel('Cuts')
plt.ylabel('Clusters')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/bkg/clusters_vs_cuts_benchmark.png')



# Adding labels and title
plt.xlabel('Cuts')
plt.ylabel('Clusters [log]')
plt.legend()
plt.yscale("log")
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/bkg/clusters_vs_cuts_logScale_benchmark.png')
plt.clf()


plt.title("True Labels")
plt.hist(labels,bins=103)
plt.yscale('log')
plt.xscale('log')
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/bkg/true_labels.png')




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
