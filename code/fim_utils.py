# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
"""Frequent itemset mining in bio-graphs."""
## توابع معمول موردنیاز برای عملیات گراف

from pandas import ExcelWriter
import pandas as pd
import numpy as np
from orangecontrib.associate.fpgrowth import *
# from itertools import tee
from tqdm import tqdm
import networkx as nx
# from bio_graph_utils import bow_nodes
# import math 
# from sklearn import preprocessing
# import ml_metrics
# # import recmetrics
from sklearn.feature_extraction.text import CountVectorizer
# import matplotlib.pyplot as plt
# from pandas import DataFrame 
# pd.options.display.float_format = "{:.2f}".format
from itertools import chain
# from scipy.sparse import csr_matrix
# from pandas import ExcelWriter
def bow_nodes_int(df):    
    numpy_matrix = df.values
    d =  numpy_matrix.transpose()
    T = [[int(x) for x in row if str(x) != 'nan'] for row in d]
    # T_str = [[str(i)[3:] for i in row ] for row in d]
    # T = [[int(i) for i in row if i != ''] for row in T_str]

    newlist = list(chain(*T))
    print('Number of unique elements of columns:', len(np.unique(newlist)))
    corpus = [None] * len(T)
    for i in range(len(T)):
        listToStr = ' '.join([str(elem) for elem in T[i]]) 
        corpus[i] = listToStr
    vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(corpus)
    print('Corpus size: ',X.shape)
    bow = X.toarray()
    # در فرم هدر نام گیاه، میشه نام متابولیت ها
    featureNames = vectorizer.get_feature_names()
    return T, bow, featureNames


def fim_bio(subG, dfct, minFreq, node_objects,edge_objects,output_dir,working_file_name):
# def fim_bio(minFreq,T,bow,featureNames):
    # T,bow,featureNames = bow_nodes_int(df_subG)
    # bow = dfct.values.transpose()

    bow = dfct.values
    T = [[i for i,x in enumerate(row) if x != 0] for row in bow]
    featureNames = list(dfct.keys().format())
    # featureNames = list(dfct.index.format())

    itemsets = frequent_itemsets(T, minFreq)
    # print(type(itemsets),itemsets)
    freqIS_list = list(itemsets)
    n_freqIS = len(freqIS_list)
    print('Number of Freq. Itemsets:', n_freqIS)

    fim_graph(subG, freqIS_list,minFreq,T,bow,featureNames)
    
    # indicesToRemove  = np.where(sum(G)==0)[0]
    # degreeG = G.sum(axis=0)
    d = subG.degree(weight='weight_fim')
    degreeG_fim = [v[1] for v in d]

    # max_n_best_plants = 100
    # nPlants = np.sum(degreeG != 0)
    # print('nPlants:',nPlants)
    # sorted_nodes = np.sort(degreeG)[::-1]
    sorted_nodes_idx = np.argsort(degreeG_fim)[::-1] # Descending order
    
    # indicesToRemove  = np.where(sum(Gw)==0)[0]
    # degreeG = Gw.sum(axis=0)
    d = subG.degree(weight='weight_fim_w')
    degreeG_fim_w = [v[1] for v in d]
    # sorted_nodes = np.sort(degreeG)[::-1]
    sorted_nodes_idx_w = np.argsort(degreeG_fim_w)[::-1] # Descending order
    # # print(degreeG[sorted_nodes_idx[0]])
    # bestPlants = sorted_nodes_idx[:np.min([nPlants,max_n_best_plants])]
    # print(bestPlants)
    # # print(plantNames[bestPlants])
    # bestPlantNames = [plantNames[x] for x in bestPlants]
    return sorted_nodes_idx, sorted_nodes_idx_w, degreeG_fim, degreeG_fim_w#, G, degreeG

def fim_graph(subG,freqIS_list,minFreq,T,bow,featureNames):
    # ایجاد گراف برای نمایش
    # print(len(featureNames))
    # print(len(T))
    # print(featureNames[0])
    # print(T[0])
    n_freqIS = len(freqIS_list)
    itemFreq = [None] * n_freqIS
    freqIS = iter(freqIS_list)
    for i,item in enumerate(freqIS):
        itemFreq[i] = item[1]

    node_names = list(subG.nodes())
    nCol = len(T)    
    # print('T=',T)
    G = np.zeros([nCol,nCol])  
    Gw = np.zeros([nCol,nCol])  
    print('Computing adjacency Graphs by Frequently Itemsets...\n')
    with tqdm(total=n_freqIS) as progress_bar:
        freqIS = iter(freqIS_list)
        for i,item in enumerate(freqIS):
            set_i = item[0]
            thisFreq = item[1]
            # if i<=3:
            #     print([featureNames[x] for x in set_i])
            if(thisFreq >= minFreq):
                items = [x for x in set_i]
                # print(items)
                commItems_idx = items #[featureNames.index(str(x)) for x in list(items)]
                # if i<=3:
                #     # print('len(featureNames)=',len(featureNames))
                #     # print('items=',items)
                #     commItems_name = [featureNames[x] for x in items]
                #     print(commItems_name)
                w = len(commItems_idx)
                vec = np.zeros((len(bow[0]),), dtype=int)
                vec[commItems_idx] = 1
                commItems = []
                for j in range(len(bow)):
                    row = bow[j]
                    if sum(row&vec)== w:
                        commItems.append(j)
                # if i<=3:
                #     print(commItems)
                for ii in range(len(commItems)):
                    for jj in range(len(commItems)):#range(ii):
                        if(ii != jj):
                            src = commItems[ii]
                            dst = commItems[jj]
                            Gw[src,dst] += w
                            G[src,dst] += 1
            progress_bar.update(1) # update progress

    for ii in range(len(G)):
        for jj in range(len(G)):#range(ii):
            src = node_names[ii]
            dst = node_names[jj]
            subG.add_edge(src,dst, weight_fim_w = Gw[ii,jj])
            subG.add_edge(src,dst, weight_fim = G[ii,jj])

    # return G,Gw