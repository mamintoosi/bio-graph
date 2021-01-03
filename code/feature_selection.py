# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
from itertools import combinations, chain
import pandas as pd
import numpy as np
from bio_graph_utils import compute_metrics
from tqdm import tqdm

#  disable warning: A value is trying to be set on a copy of a slice from a DataFrame.
pd.options.mode.chained_assignment = None  # default='warn'

data_dir = 'data/'
output_dir = 'results/'
working_dir = 'data/'
working_file_name = 'AC_Met_Plant.xlsx'
GT_file_name = 'LR_Met_Plant.xlsx'
node_objects = 'Plant'
edge_objects = 'Met'

min_count = 5
minFreq = 5

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def obj_fun(true_list,pred_list):
    index = np.arange(1,20,1)
    [apk,mapk,mark] = compute_metrics(true_list, pred_list)
    return np.mean(apk)

file_name = 'results/{}_{}_{}_{}_{}_{}.xlsx'.format(working_file_name[:2],GT_file_name[:2],node_objects,\
    edge_objects,str(min_count),str(minFreq))
df = pd.read_excel(file_name, sheet_name='gf_df', engine="openpyxl" , index_col=0)

GT_file = working_dir+GT_file_name
true_df = pd.read_excel(GT_file, engine="openpyxl") 
true_list = list(true_df[node_objects].unique())

n_features = df.shape[1]-1

# features = np.array([[0, 1, 1],
#                      [1, 0, 0],
#                      [1, 1, 1]])
# features = df.iloc[:,:-1].values

# n = features.shape[1]

# get all combinations, we will use this as indices for the columns later
indices = list(powerset(range(n_features)))
# remove the empty subset
indices.pop(0)

# print(indices)
# data = []
scores = []

with tqdm(total=len(indices)) as progress_bar:
    for idx in indices:
        # print()
        cols = list(idx)
        cols_s = list(idx) # columns with feature_sum
        cols_s.append(-1)
        selected_df = df.iloc[:,cols_s]
        # print(cols,cols_s,selected_df.shape)#,cols_s)
        n_cols = selected_df.shape[1]-1
        features_sum = selected_df.iloc[:,:n_cols].sum(axis=1)
        selected_df['features_sum'] = features_sum
        selected_df_sorted = selected_df.sort_values(by='features_sum', ascending=False)
        pred_list = list(selected_df_sorted.index.values)
        # print(obj_fun(true_list,pred_list))
        scores.append(obj_fun(true_list,pred_list))
        # break
        progress_bar.update(1)

best_idx = np.argmax(np.array(scores))
# best_idx,scores[best_idx],indices[best_idx]
f_names = list(df.keys().format())
f_names = list(df.keys().format())
best_features = [f_names[i] for i in indices[best_idx]]
print('Best Features are: ',best_features)