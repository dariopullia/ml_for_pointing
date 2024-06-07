import re
import numpy as npy
import matplotlib.pyplot as plt

cuts = [50000, 60000, 70000, 80000, 100000, 120000, 140000, 150000, 160000, 180000, 225000, 250000, 275000, 300000, 325000, 
        350000, 400000, 500000, 600000, 700000]
plot_cuts = []
true_pos=[]
true_neg = []
false_pos = []
false_neg = []
f1s = []

cut_counts = 0
for cut in cuts:
    try:
        with open(f"/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/{cut}/hyperopt_simple_cnn/aug_coeff_1/metrics.txt",'r') as file:
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
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/models_confusion_pos.png')
plt.clf()

#Just false
plt.title("Confusion Matrix Values across Different Cuts")
plt.plot(plot_cuts, false_pos, marker='o', linestyle='-', color='r', label='False Positives')
plt.plot(plot_cuts, false_neg, marker='o', linestyle='-', color='m', label='False Negatives')

plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
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
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/models_confusion.png')


plt.clf()
plt.title("F1 Score across Different Cuts")
plt.plot(plot_cuts, f1s, marker='o', linestyle='-', color='m', label='F1 Score')
plt.xlabel('Cuts')
plt.ylabel('Fraction')
plt.legend()
plt.savefig(f'/eos/user/h/hakins/dune/ML/mt_identifier/ds-mix-mt-vs-all/plots/F1_scores.png')