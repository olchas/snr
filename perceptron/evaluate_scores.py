import sys

import numpy as np
import pandas as pd

# usage: python evaluate_scores.py ${path_to_scores_file_with_experiment_results}

scores = pd.read_csv(sys.argv[1], sep='\t')

nr_of_nets = []
nr_of_relu = []
nr_of_sigmoid = []
nr_of_softmax = []
nr_of_tanh = []

scores_sorted = scores.sort_values('top_1_accuracy', ascending=False).copy()

accuracy_range = np.arange(0, 1.01, 0.01)

for min_accuracy in accuracy_range:
    
    nr_of_nets.append(len(scores[scores['top_1_accuracy'] > min_accuracy]))
    
    nr_of_relu.append(scores[scores['top_1_accuracy'] > min_accuracy]['model'].apply(lambda x: x.count('relu')).sum())
    nr_of_sigmoid.append(scores[scores['top_1_accuracy'] > min_accuracy]['model'].apply(lambda x: x.count('sigmoid')).sum())
    nr_of_softmax.append(scores[scores['top_1_accuracy'] > min_accuracy]['model'].apply(lambda x: x.count('softmax')).sum())
    nr_of_tanh.append(scores[scores['top_1_accuracy'] > min_accuracy]['model'].apply(lambda x: x.count('tanh')).sum())
    
nr_of_activation_2_accuracy_df = pd.DataFrame({'min_accuracy': accuracy_range,
                                               'nr_of_nets': nr_of_nets,
                                               'relu': nr_of_relu,
                                               'sigmoid': nr_of_sigmoid,
                                               'softmax': nr_of_softmax,
                                               'tanh': nr_of_tanh})

nr_of_activation_2_accuracy_df.to_csv('nr_of_activation_2_accuracy_df.tsv', sep='\t', index=False)

nr_of_relu = []
nr_of_sigmoid = []
nr_of_softmax = []
nr_of_tanh = []
    
for i, row in scores_sorted.iterrows():

    if i == 0:
        nr_of_relu.append(str(row['model']).count('relu'))
        nr_of_sigmoid.append(str(row['model']).count('sigmoid'))
        nr_of_softmax.append(str(row['model']).count('softmax'))
        nr_of_tanh.append(str(row['model']).count('tanh'))

    else:
        nr_of_relu.append(str(row['model']).count('relu') + nr_of_relu[-1])
        nr_of_sigmoid.append(str(row['model']).count('sigmoid') + nr_of_sigmoid[-1])
        nr_of_softmax.append(str(row['model']).count('softmax') + nr_of_softmax[-1])
        nr_of_tanh.append(str(row['model']).count('tanh') + nr_of_tanh[-1])
        
nr_of_activation_2_score_df = pd.DataFrame({'relu': nr_of_relu,
                                            'sigmoid': nr_of_sigmoid,
                                            'softmax': nr_of_softmax,
                                            'tanh': nr_of_tanh})

nr_of_activation_2_score_df.to_csv('nr_of_activation_2_score.tsv', sep='\t')
