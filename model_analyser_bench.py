
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
cuts = [
    #50000, 60000, 
        70000,80000, 100000, 120000, 140000, 150000, 160000
         , 180000, 225000, 250000, 275000, 300000, 325000, 
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
    

threshold = .7

#nested_lists for multiple plots
true_negatives_nested = []
false_negatives_nested = []
identified_ccs_nested=[]
identified_es_nested=[]
false_positives_nested=[]
total_ccs_nested=[]
totals_es_nested=[]
true_positives_nested=[]
blip_id_as_mt_nested =[]
bkg_id_as_mt_nested = []
#params
decay_rates = []
kernel_sizes = []
learning_rates = []
n_conv_layers = []
n_dense_layers = []
n_dense_units = []
n_filters = []

hyperparameters = { #the best.npy file returns the index for a few of them: n_conv_layers, n_dense layers, n_dense_units, n_filters,kernal size
    "n_conv_layers": [1, 2, 3, 4],
    "n_dense_layers": [2, 3, 4],
    "n_filters": [16, 32, 64],
    "kernel_size": [1, 3, 5],
    "n_dense_units": [32, 64, 128],
    "learning_rate": [0.0001, 0.001],
    "decay_rate": [0.9, 0.999]
}

 


for model_cut in cuts:
    
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
    blip_id_as_mt =[]
    bkg_id_as_mt = []
    
    for bench_cut in cuts:
        if os.path.exists(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/model_{model_cut}_on_cut_{bench_cut}ctds/dataset/dataset_label_process.npy"):
            labels = np.load(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/model_{model_cut}_on_cut_{bench_cut}ctds/dataset/dataset_label_process.npy")
            predictions = np.load(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/model_{model_cut}_on_cut_{bench_cut}mt_id/predictions.npy")
            #print("File loaded successfully.")
        else:
            print(f"-cut_model {model_cut} -cut_bench {bench_cut}")
        #labels = np.load(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/model_{model_cut}_on_cut_{bench_cut}ctds/dataset/dataset_label_process.npy")
        #predictions = np.load(f"/eos/user/h/hakins/dune/ML/mt_identifier/benchmark/model_{model_cut}_on_cut_{bench_cut}mt_id/predictions.npy")
        #hyperparamters
        '''params = np.load(f"/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/{cut}/hyperopt_simple_cnn/aug_coeff_1/hyperopt_simple_cnn_best.npy", allow_pickle=True)
        params_dict = params.item()
        decay_rates.append(params_dict['decay_rate'])
        kernel_sizes.append(hyperparameters['kernel_size'][params_dict['kernel_size']])
        learning_rates.append(params_dict['learning_rate'])
        n_conv_layers.append(hyperparameters['n_conv_layers'][params_dict['n_conv_layers']])
        n_dense_layers.append(hyperparameters['n_dense_layers'][params_dict['n_dense_layers']])
        n_dense_units.append(hyperparameters['n_dense_units'][params_dict['n_dense_units']])
        n_filters.append(hyperparameters['n_filters'][params_dict['n_filters']])'''
        
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
        blip_as_mt=0
        bkg_as_mt=0

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
            if labels[i] == 1  and predictions[i]>threshold:
                blip_as_mt+=1
            if labels[i] != 1 and labels[i] not in mt_list and predictions[i]>threshold:
                bkg_as_mt+=1
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
        blip_id_as_mt.append(blip_as_mt)
        bkg_id_as_mt.append(bkg_as_mt)

    
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
    cluster_sizes_mt = np.array(cluster_sizes_mt)
    blip_id_as_mt = np.array(blip_id_as_mt)
    bkg_id_as_mt = np.array(bkg_id_as_mt)
    bkgs = np.array(bkgs)
    blips = np.array(blips)
    
    true_negatives_nested.append(true_negatives)
    true_positives_nested.append(true_positives)
    false_negatives_nested.append(false_negatives)
    identified_ccs_nested.append(identified_ccs)
    identified_es_nested.append(identified_es)
    false_positives_nested.append(false_positives) 
    total_ccs_nested.append(total_ccs)
    totals_es_nested.append(totals_es)
    blip_id_as_mt_nested.append(blip_id_as_mt)
    bkg_id_as_mt_nested.append(bkg_id_as_mt)
    #TP: Main Track identified as a Main Track
    #FP: BKG identidied as a Main Track
    #TN: BKG identified as a BKG
    #FN: Main track identified as a BKG
    #normalize

plt.figure()
plt.title("Absolute F1 Scores across Different Cuts")
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.ylim(0,1)
colors=['lightcoral','lightblue','lightgreen','blue']
norm_tp_list=[]
norm_tn_list=[]
norm_fp_list=[]
norm_fn_list=[]
for i in range(len(cuts)):
    norm_true_positives = true_positives_nested[i] / cluster_sizes_mt[0]
    norm_true_negatives = true_negatives_nested[i] /  (blips[0]+bkgs[0])
    norm_false_positives = false_positives_nested[i] / (blips[0]+bkgs[0])
    norm_false_negatives = false_negatives_nested[i] / cluster_sizes_mt[0] 
    
    accuracy = (norm_true_positives+norm_true_negatives)/ (norm_true_negatives+norm_true_positives+norm_false_positives+norm_false_negatives)
    recall = (norm_true_positives)/(norm_true_positives+norm_false_negatives)
    precicion = norm_true_positives/(norm_true_positives+norm_false_positives)
    f1_Score = 2 * (precicion*recall)/(precicion + recall)
    plt.plot(cuts,f1_Score, marker='o', linestyle='-', label=f'F1 Score Model {cuts[i]}')

    norm_tp_list.append(norm_true_positives)
    norm_tn_list.append(norm_true_negatives)
    norm_fp_list.append(norm_false_positives)
    norm_fn_list.append(norm_false_negatives)
plt.legend()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_f1_absolute.png')
plt.clf()

#F1 Score
plt.title("Confusion Matrix Values across Different Cuts")
#Confusion MAtrix
for i in range(len(cuts)):
    plt.plot(cuts,norm_tp_list[i], marker='o', linestyle='-', color=colors[0],alpha=1/(i+1), label=f'True Positive model {cuts[i]} ')
    plt.plot(cuts,norm_tn_list[i], marker='o', linestyle='-', color=colors[1],alpha=1/(i+1), label='True Negative')
    plt.plot(cuts,norm_fp_list[i], marker='o', linestyle='-', color=colors[2],alpha=1/(i+1), label='False Positive')
    plt.plot(cuts,norm_fn_list[i], marker='o', linestyle='-', color=colors[3],alpha=1/(i+1), label='False Negative')

#plt.plot(cuts, norm_true_negatives, marker='o', linestyle='-', color='lightblue', label='True Negative')
#plt.plot(cuts, norm_false_positives, marker='o', linestyle='-', color='lightcoral', label='False Positive')
#plt.plot(cuts, norm_false_negatives, marker='o', linestyle='-', color='m', label='False Negative')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
#plt.legend()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion_absolute_crossed.png')
plt.clf()

#True Positives / True Pos + False Neg
cmap = plt.get_cmap("tab20")  # Use a colormap with 20 distinct colors

newfig = plt.figure(figsize=(12, 8))  # Increase figure size for better readability
for i in range(len(cuts)):
    plt.plot(cuts, true_positives_nested[i]/(false_negatives_nested[i]+true_positives_nested[i]), 
             label=f"Model {cuts[i]}", color=cmap(i % 20))  # Ensure colors are distinct and cycle if necessary

plt.xlabel("Cuts")
plt.ylabel("TP / (TP + FN)")
plt.title("TP / (TP + FN) for Different Models at Different Cuts")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Position legend outside the plot
plt.tight_layout()  # Adjust layout to prevent clipping of ylabel and title
plt.ylim(0,1)
plt.savefig("/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/TP_over_FN_plus_TP.png")
plt.clf()
#True Positives / True Pos + False Neg
cmap = plt.get_cmap("tab20")  # Use a colormap with 20 distinct colors

newfig = plt.figure(figsize=(12, 8))  # Increase figure size for better readability
for i in range(len(cuts)):
    plt.plot(cuts, false_positives_nested[i]/(identified_es_nested[i]+false_positives_nested[i]), 
             label=f"Model {cuts[i]}", color=cmap(i % 20))  # Ensure colors are distinct and cycle if necessary

plt.xlabel("Cuts")
plt.ylabel("FP / (FP + TP(ES))")
plt.title("FP / (FP + TP(ES)) for Different Models at Different Cuts")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Position legend outside the plot
plt.tight_layout()  # Adjust layout to prevent clipping of ylabel and title
plt.ylim(0,1)
plt.savefig("/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/FP_over_FP_plus_TP.png")
plt.clf()

#TPs / All
cmap = plt.get_cmap("tab20")  # Use a colormap with 20 distinct colors

newfig = plt.figure(figsize=(12, 8))  # Increase figure size for better readability
for i in range(len(cuts)):
    plt.plot(cuts, true_positives_nested[i]/(true_positives_nested[i]+false_positives_nested[i] + true_negatives_nested[i] + false_negatives_nested[i]), 
             label=f"Model {cuts[i]}", color=cmap(i % 20))  # Ensure colors are distinct and cycle if necessary

plt.xlabel("Cuts")
plt.ylabel("TP / All Clusters")
plt.title("TP / All Clusters for Different Models at Different Cuts")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Position legend outside the plot
plt.tight_layout()  # Adjust layout to prevent clipping of ylabel and title
plt.ylim(0,1)
plt.savefig("/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/TP_over_all.png")
plt.clf()
#TPs
cmap = plt.get_cmap("tab20")  # Use a colormap with 20 distinct colors

newfig = plt.figure(figsize=(12, 8))  # Increase figure size for better readability
for i in range(len(cuts)):
    plt.plot(cuts, true_positives_nested[i], label=f"Model {cuts[i]}", color=cmap(i % 20))  # Ensure colors are distinct and cycle if necessary

plt.xlabel("Cuts")
plt.ylabel("TPs")
plt.title("Number of True Positives vs Cuts")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Position legend outside the plot
plt.tight_layout()  # Adjust layout to prevent clipping of ylabel and title
plt.savefig("/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/TPs.png")
plt.clf()


#TP / TP + FP
cmap = plt.get_cmap("tab20")  # Use a colormap with 20 distinct colors

newfig = plt.figure(figsize=(12, 8))  # Increase figure size for better readability
for i in range(len(cuts)):
    plt.plot(cuts, true_positives_nested[i]/(true_positives_nested[i]+false_positives_nested[i]), label=f"Model {cuts[i]}", color=cmap(i % 20))  # Ensure colors are distinct and cycle if necessary

plt.xlabel("Cuts")
plt.ylabel("TP / (TP + FP)")
plt.title("True Positive / (True Positive + False Positive) vs Cuts")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)  # Position legend outside the plot
plt.tight_layout()  # Adjust layout to prevent clipping of ylabel and title
plt.savefig("/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/TP_TP_plus_FP.png")
plt.clf()

#Top Picks. The same model will be same color, with different metrics different line styles
# Create a plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot with primary y-axis

for i in range(len(cuts)):
    if i == 0 or i == 5 or i == 15:
        ax1.plot(cuts, true_positives_nested[i] / (true_positives_nested[i] + false_positives_nested[i]), label=f"Model {cuts[i]} TP / TP + FP", color=cmap(i%20))
        #ax1.plot(cuts, true_positives_nested[i] / (true_positives_nested[i] + false_positives_nested[i] + true_negatives_nested[i] + false_negatives_nested[i]), linestyle='--', label=f"Model {cuts[i]} TP / All", color=cmap(i%20))
        ax1.plot(cuts, true_positives_nested[i] / (false_negatives_nested[i] + true_positives_nested[i]), linestyle='-.', label=f"Model {cuts[i]} TP / FN + TP", color=cmap(i%20))

# Create secondary y-axis
ax2 = ax1.twinx()
for i in range(len(cuts)):
    if i == 0 or i == 5 or i == 15:
        ax2.plot(cuts, true_positives_nested[i], label=f"Model {cuts[i]} TPs", color=cmap(i%20), linestyle=':')

# Set labels
ax1.set_xlabel('Cuts [MeV]')
ax1.set_ylabel('Proportions')
ax2.set_ylabel('True Positives')

# Add legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2,loc='center right')

ticks = [70000,100000,150000,200000,250000,300000,350000,400000,450000,500000]
ticks_mev = []
for tick in ticks:
    ticks_mev.append(tick*(1.58*np.power(10.0,-5)) - .04)

# Title and layout
plt.xticks([70000,100000,150000,200000,250000,300000,350000,400000,450000,500000], ticks_mev)
#PLACE TICKSMEV HERE
plt.title('Performance Metrics with Different Y Scales')
plt.tight_layout()
plt.savefig("/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/best_models_metrics.png")
plt.clf()

'''#identified ES events and BKG contamination
plt.title("MT Sample Composition")
plt.plot(cuts,identified_es, marker='o', linestyle='-', color='lightgreen', label='ES True Positive')
plt.plot(cuts,totals_es, marker = 'o',linestyle='--', color='lightgreen', label='Total ES')
plt.plot(cuts, norm_true_negatives, marker='o', linestyle='-', color='g', label='True Negative')
plt.plot(cuts, norm_false_positives, marker='o', linestyle='-', color='r', label='False Positive')
plt.plot(cuts, norm_false_negatives, marker='o', linestyle='-', color='m', label='False Negative')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/models_confusion_absolute_ES.png')
plt.clf()
'''


# Convert cuts to strings for labeling
#Makes many plots
'''for i in range(len(cuts)):
    cut_labels = [str(cut) for cut in cuts]
    x = np.arange(len(cuts))
    bar_width = 0.5
    fig, ax = plt.subplots(figsize=(10, 6))  
    bars2 = ax.bar(x, identified_es_nested[i], bar_width,label='ES Identified as MT', color='lightgreen', edgecolor='lightgreen', alpha=0.7)
    bars2_hatch = ax.bar(x, totals_es_nested[i]-identified_es_nested[i],bar_width,bottom=identified_es,label="ES Identified as BKG", color='none', edgecolor='lightgreen',hatch='////',alpha=0.7,zorder=10)
    bars1 = ax.bar(x, identified_ccs_nested[i], bar_width,bottom=totals_es, label='CC Identified as MT', color='blue',edgecolor='blue', alpha=0.45)
    bars1_hatch = ax.bar(x, total_ccs_nested[i]-identified_ccs_nested[i], bar_width,bottom=(identified_ccs+totals_es), label='CC Idenfitiied as BKG', color='none', edgecolor='blue',hatch='////', alpha=0.45,zorder=10)
    bars3 = ax.bar(x, false_positives_nested[i], bar_width, bottom=cluster_sizes_mt, label='BKG+Blip Identified as MT', color='lightcoral',edgecolor='lightcoral', alpha=0.7)
    ax.set_xlabel('Cuts')
    ax.set_ylabel('Counts')
    ax.set_title('Composition of Sample After MT Identifier at Each Cut')
    ax.set_xticks(x)
    ax.set_xticklabels(cut_labels, rotation=45, ha='right')  # Rotate x labels
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/MT_sample_composition.png')'''
#makes one large subplot

fig, axs = plt.subplots(5, 1, figsize=(10, 3 * len(cuts)/2),sharex=True)

for i in range(int(5)):
    x = np.arange(len(cuts))
    bar_width = 0.5

    bars2 = axs[i].bar(x, identified_es_nested[i], bar_width, label='ES Identified as MT', color='lightgreen', edgecolor='lightgreen', alpha=0.7)
    bars2_hatch = axs[i].bar(x, np.subtract(totals_es_nested[i], identified_es_nested[i]), bar_width, bottom=identified_es_nested[i], label="ES Identified as BKG", color='none', edgecolor='lightgreen', hatch='////', alpha=0.7, zorder=10)

    bars1 = axs[i].bar(x, identified_ccs_nested[i], bar_width, bottom=totals_es_nested[i], label='CC Identified as MT', color='blue', edgecolor='blue', alpha=0.45)
    bars1_hatch = axs[i].bar(x, np.subtract(total_ccs_nested[i], identified_ccs_nested[i]), bar_width, bottom=np.add(identified_ccs_nested[i], totals_es_nested[i]), label='CC Identified as BKG', color='none', edgecolor='blue', hatch='////', alpha=0.45, zorder=10)

    bars3 = axs[i].bar(x, blip_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt, label='Blip Identified as MT', color='lightcoral', edgecolor='lightcoral', alpha=0.7)
    bars3_hatch = axs[i].bar(x, blips - blip_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blip_id_as_mt_nested[i], label='Blip Identified as Blip', color='none', edgecolor='lightcoral', hatch='////', alpha=0.45, zorder=10)

    bars4 = axs[i].bar(x, bkg_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blips, label='Bkg Identified as MT', color='orange', edgecolor='orange', alpha=0.4)
    bars4_hatch = axs[i].bar(x, bkgs - bkg_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blips+bkg_id_as_mt_nested[i], label='Bkg Identified as Bkg', color='none', edgecolor='orange', hatch='////', alpha=0.4, zorder=10)

    axs[i].set_ylabel('Counts')
    axs[i].set_ylim(0,200)
    axs[i].set_title(f'Composition of Sample After MT Identifier at {cuts[i]}')
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(cuts, rotation=45, ha='right')  # Rotate x labels
    
plt.legend()
fig.suptitle('Stacked Plots of Composition of Sample After MT Identifier')
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust top spacing for the overall title
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/MT_sample_composition_stacked_1stthird.png')


fig, axs = plt.subplots(5, 1, figsize=(10, 3 * int(len(cuts)/2)), sharex=True)

for i in range(5, 10):
    x = np.arange(len(cuts))
    bar_width = 0.5

    bars2 = axs[i-5].bar(x, identified_es_nested[i], bar_width, label='ES Identified as MT', color='lightgreen', edgecolor='lightgreen', alpha=0.7)
    bars2_hatch = axs[i-5].bar(x, np.subtract(totals_es_nested[i], identified_es_nested[i]), bar_width, bottom=identified_es_nested[i], label="ES Identified as BKG", color='none', edgecolor='lightgreen', hatch='////', alpha=0.7, zorder=10)

    bars1 = axs[i-5].bar(x, identified_ccs_nested[i], bar_width, bottom=totals_es_nested[i], label='CC Identified as MT', color='blue', edgecolor='blue', alpha=0.45)
    bars1_hatch = axs[i-5].bar(x, np.subtract(total_ccs_nested[i], identified_ccs_nested[i]), bar_width, bottom=np.add(identified_ccs_nested[i], totals_es_nested[i]), label='CC Identified as BKG', color='none', edgecolor='blue', hatch='////', alpha=0.45, zorder=10)

    bars3 = axs[i-5].bar(x, blip_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt, label='Blip Identified as MT', color='lightcoral', edgecolor='lightcoral', alpha=0.7)
    bars3_hatch = axs[i-5].bar(x, blips - blip_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blip_id_as_mt_nested[i], label='Blip Identified as Blip', color='none', edgecolor='lightcoral', hatch='////', alpha=0.45, zorder=10)

    bars4 = axs[i-5].bar(x, bkg_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blips, label='Bkg Identified as MT', color='orange', edgecolor='orange', alpha=0.4)
    bars4_hatch = axs[i-5].bar(x, bkgs - bkg_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blips+bkg_id_as_mt_nested[i], label='Bkg Identified as Bkg', color='none', edgecolor='orange', hatch='////', alpha=0.4, zorder=10)

    axs[i-5].set_ylabel('Counts')
    axs[i-5].set_ylim(0,200)
    axs[i-5].set_title(f'Composition of Sample After MT Identifier using Model {cuts[i]}')
    axs[i-5].set_xticks(x)
    axs[i-5].set_xticklabels(cuts, rotation=45, ha='right')  # Rotate x labels
    

fig.suptitle('Stacked Plots of Composition of Sample After MT Identifier')
plt.legend()
plt.tight_layout()

plt.subplots_adjust(top=0.95)  # Adjust top spacing for the overall title
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/MT_sample_composition_stacked_2ndthird.png')
fig, axs = plt.subplots(6, 1, figsize=(10, 3 * int(len(cuts)/2)), sharex=True)

for i in range(10, 16):
    x = np.arange(len(cuts))
    bar_width = 0.5

    bars2 = axs[i-10].bar(x, identified_es_nested[i], bar_width, label='ES Identified as MT', color='lightgreen', edgecolor='lightgreen', alpha=0.7)
    bars2_hatch = axs[i-10].bar(x, np.subtract(totals_es_nested[i], identified_es_nested[i]), bar_width, bottom=identified_es_nested[i], label="ES Identified as BKG", color='none', edgecolor='lightgreen', hatch='////', alpha=0.7, zorder=10)

    bars1 = axs[i-10].bar(x, identified_ccs_nested[i], bar_width, bottom=totals_es_nested[i], label='CC Identified as MT', color='blue', edgecolor='blue', alpha=0.45)
    bars1_hatch = axs[i-10].bar(x, np.subtract(total_ccs_nested[i], identified_ccs_nested[i]), bar_width, bottom=np.add(identified_ccs_nested[i], totals_es_nested[i]), label='CC Identified as BKG', color='none', edgecolor='blue', hatch='////', alpha=0.45, zorder=10)

    bars3 = axs[i-10].bar(x, blip_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt, label='Blip Identified as MT', color='lightcoral', edgecolor='lightcoral', alpha=0.7)
    bars3_hatch = axs[i-10].bar(x, blips - blip_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blip_id_as_mt_nested[i], label='Blip Identified as Blip', color='none', edgecolor='lightcoral', hatch='////', alpha=0.45, zorder=10)

    bars4 = axs[i-10].bar(x, bkg_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blips, label='Bkg Identified as MT', color='orange', edgecolor='orange', alpha=0.4)
    bars4_hatch = axs[i-10].bar(x, bkgs - bkg_id_as_mt_nested[i], bar_width, bottom=cluster_sizes_mt+blips+bkg_id_as_mt_nested[i], label='Bkg Identified as Bkg', color='none', edgecolor='orange', hatch='////', alpha=0.4, zorder=10)

    axs[i-10].set_ylabel('Counts')
    axs[i-10].set_ylim(0,200)
    axs[i-10].set_title(f'Composition of Sample After MT Identifier using Model {cuts[i]}')
    axs[i-10].set_xticks(x)
    axs[i-10].set_xticklabels(cuts, rotation=45, ha='right')  # Rotate x labels
    

fig.suptitle('Stacked Plots of Composition of Sample After MT Identifier')
plt.legend()
plt.tight_layout()

plt.subplots_adjust(top=0.95)  # Adjust top spacing for the overall title
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/MT_sample_composition_stacked_3rdthird.png')
plt.clf()

#Key Plots


    

'''# Isolating ES and CC events
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
plt.savefig('/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/benchmark/MT_ESvCC_composition.png')'''


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
'''# Plotting hyperparameters as a function of cuts
plt.figure(figsize=(30, 12))

# Decay rate
plt.subplot(2, 4, 1)
plt.plot(cuts, decay_rates, marker='o')
plt.title('Decay Rate')
plt.xlabel('Cut')
plt.ylabel('Decay Rate')

# Kernel size
plt.subplot(2, 4, 2)
plt.plot(cuts, kernel_sizes, marker='o')
plt.title('Kernel Size')
plt.xlabel('Cut')
plt.ylabel('Kernel Size')

# Learning rate
plt.subplot(2, 4, 3)
plt.plot(cuts, learning_rates, marker='o')
plt.title('Learning Rate')
plt.xlabel('Cut')
plt.ylabel('Learning Rate')

# Number of convolutional layers
plt.subplot(2, 4, 4)
plt.plot(cuts, n_conv_layers, marker='o')
plt.title('Number of Conv Layers')
plt.xlabel('Cut')
plt.ylabel('Number of Conv Layers')

# Number of dense layers
plt.subplot(2, 4, 5)
plt.plot(cuts, n_dense_layers, marker='o')
plt.title('Number of Dense Layers')
plt.xlabel('Cut')
plt.ylabel('Number of Dense Layers')

# Number of dense units
plt.subplot(2, 4, 6)
plt.plot(cuts, n_dense_units, marker='o')
plt.title('Number of Dense Units')
plt.xlabel('Cut')
plt.ylabel('Number of Dense Units')

# Number of filters
plt.subplot(2, 4, 7)
plt.plot(cuts, n_filters, marker='o')
plt.title('Number of Filters')
plt.xlabel('Cut')
plt.ylabel('Number of Filters')

plt.tight_layout()
plt.savefig("/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/hyper-paramter-vs-cuts.png")
plt.show()'''







    

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
