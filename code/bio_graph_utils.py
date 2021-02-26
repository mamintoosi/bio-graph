# -*- coding: utf-8 -*-
#
#    Copyright (C) 2021-2029 by
#    Mahmood Amintoosi <m.amintoosi@gmail.com>
#    All rights reserved.
#    BSD license.
"""Algorithms used in bio-graphs."""

# توابع معمول موردنیاز برای عملیات گراف

from scipy.sparse import csr_matrix
from itertools import chain
from pandas import ExcelWriter
import pandas as pd
import numpy as np
# from orangecontrib.associate.fpgrowth import *
# from itertools import tee
from tqdm.notebook import tqdm
import networkx as nx
import math
from sklearn import preprocessing
import ml_metrics
import recmetrics
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from pandas import DataFrame
pd.options.display.float_format = "{:.2f}".format


# Compute Graph Features
def graph_features(G, weight=None, normalize=True):
    """
    Computer graph nodes attributes

    Parameters
    ----------
    G : nx.Graph

    weight : string or None, optional (default=None)
    The edge attribute that holds the numerical value used as a weight.
    Degree of nodes computed according to the graph nodes' weights
    If None, then each edge has weight 1.

    normalize : boolean
        If True, All columns of the returned DataFrame will be normalized.

    Returns
    -------
    dataframe

    Notes
    -----
    Currently weight parameter is considered only for node degrees. 
    Some other criteria have not accepted weight parameter, 
    some of them such as betweenness, are sensible to weighted edges,
    but the minimum edge value is better for these criteria,
    but the default value is that the strong weight indicate stronger connections.

    """    
    df = pd.DataFrame(index=G.nodes())
    with tqdm(total=6) as progress_bar:
        d = G.degree(weight=weight)
        df['degree'] = ([v[1] for v in d]) 
        progress_bar.update(1)
        df['degree_cent'] = pd.Series(nx.degree_centrality(G))
        progress_bar.update(1)
        df['betweenness'] = pd.Series(nx.betweenness_centrality(G))
        progress_bar.update(1)
        df['closeness'] = pd.Series(nx.closeness_centrality(G))
        progress_bar.update(1)

        # ظاهرا هر چه کمتر باشه بهتره
        df['eccentricity'] = -pd.Series(nx.eccentricity(G))
        progress_bar.update(1)
        df['eigenvector'] = pd.Series(nx.eigenvector_centrality(G))
        progress_bar.update(1)

    if(normalize):
        min_max_scaler = preprocessing.MinMaxScaler()
        numpy_matrix = df.values
        X = min_max_scaler.fit_transform(numpy_matrix)
        for i, col in enumerate(df.columns):
            df.loc[:, col] = X[:, i]
    # محاسبه مجموع ویژگی ها
    features_sum = df.sum(axis=1)
    df['features_sum'] = features_sum

    return df

# Computing Baog of Word


def bow_nodes(df):
    numpy_matrix = df.values
    d = numpy_matrix.transpose()
    T = [[int(x) for x in row if str(x) != 'nan'] for row in d]
    # T_str = [[str(i)[3:] for i in row ] for row in d]
    # T_int = [[int(i) for i in row if i != ''] for row in T_str]
    # T = [[str(i)[3:] for i in row ] for row in d]
    # T = [[int(i) for i in row if i != ''] for row in T]

    newlist = list(chain(*T))
    print('Number of unique elements of columns:', len(np.unique(newlist)))
    corpus = [None] * len(T)
    for i in range(len(T)):
        listToStr = ' '.join([str(elem) for elem in T[i]])
        corpus[i] = listToStr
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(corpus)
    print('Corpus size: ', X.shape)
    bow = X.toarray()
    # در فرم هدر نام گیاه، میشه نام متابولیت ها
    featureNames = vectorizer.get_feature_names()
    return T, bow, featureNames

# Make graph from Bag of Words


def make_graph_from_bow(bow):
    # ایجاد ماتریس مجاورتی گراف
    n_nodes = bow.shape[0]
    M = np.zeros((n_nodes, n_nodes))
    # M = csr_matrix((n_nodes,n_nodes))
    print('Computing Adjacency Matrix...')
    with tqdm(total=n_nodes*n_nodes) as progress_bar:
        for i in range(n_nodes):
            for j in range(n_nodes):
                M[i, j] = sum(bow[i] & bow[j])
                progress_bar.update(1)

    G = nx.Graph(M)
    return G

# Finding largest connected component of G


def largest_con_com(df, G):
    conComp = list(nx.connected_components(G))
    n_con_comp = [len(x) for x in conComp]
    idx = np.argsort(n_con_comp)
    maxIdx = idx[-1]
    # print(maxIdx,n_con_comp[maxIdx])
    con_comp_indices = list(conComp[maxIdx])
    # print(con_comp_indices)
    subG = G.subgraph(nodes=con_comp_indices).copy()
    # در اینجا فهرست نودهای متصب نام گره ها هستند
    # node_names = [df.index.format()[x] for x in con_comp_indices]
    # mapping = dict(zip(con_comp_indices, node_names))
    # subG = nx.relabel_nodes(subG, mapping)
    return subG

def make_graph_from_df(df,node_objects,edge_objects):
    dfct = pd.crosstab(df[node_objects], df[edge_objects])
    bow = dfct.values
    M = bow.dot(bow.T)
    np.fill_diagonal(M,0)
    G = nx.Graph(M)
    node_names = dfct.index.format()
    mapping = dict(zip(np.arange(len(node_names)), node_names))
    G = nx.relabel_nodes(G, mapping)
    return G,dfct,bow



def rank_using_graph_features(subG, min_count, node_objects, edge_objects, data_dir, output_dir, working_file_name):
    # print('Computing features...\n')
    gf_df = graph_features(subG)  # graph features data frame
    # print(gf_df)

    # file_name = data_dir+node_objects+"_features_min_count_" + \
    #     str(min_count)+"_"+working_file_name
    # # file_name
    # writer = ExcelWriter(file_name)
    # gf_df.to_excel(writer, 'features')  # , index=False)
    # writer.save()
    # مرتب سازی بر حسب مجموع ویژگی ها
    gf_df_sorted = gf_df.sort_values(by='features_sum', ascending=False)

    # file_name = output_dir+node_objects+"_features_min_count_" + \
    #     str(min_count)+"_"+working_file_name
    # file_name
    # writer = ExcelWriter(file_name)
    # gf_df_sorted.to_excel(writer, node_objects)  # , index=False)
    # writer.save()

    return gf_df, gf_df_sorted


def compute_metrics(true_list, recom_list, apk_ranges=np.arange(1, 50, 2), mapk_ranges=np.arange(1, 50, 2)):
    apk = []
    for K in apk_ranges:
        apk.extend([ml_metrics.apk(true_list, recom_list, k=K)])
    mapk = []
    # با مراجعه به سورس تابع زیر مشخص شد که این برای کار ما درست نیست
    # دلیل در انتهای فایل
    # for K in mapk_ranges:
    #     mapk.extend([ml_metrics.mapk([true_list], [recom_list], k=K)])
        # mapk.extend([recmetrics.novelty(true_list, recom_list, k=K)]) # فرقی نکرد!!

    #     به همان دلیل بالا بخش زیر هم مناسب کار ما نیست
    # mark = []
    # for K in mapk_ranges:
    #     mark.extend([recmetrics.mark(true_list, recom_list, k=K)])

    # و چون 
    # _ark
    # رو نشد فراخوانی کنم
    # لذا هر کدام را داخل کروشه می‌گذاریم
    ark = []
    for K in apk_ranges:
        ark.extend([recmetrics.mark([true_list], [recom_list], k=K)])

    return [apk, ark]


def AC_df_to_2_col():
    file_name = 'data/AC_restructured.xlsx'
    output_file_name = 'data/AC_met_plant.xlsx'
    df = pd.read_excel(file_name, engine="openpyxl")
    df2 = pd.DataFrame(columns=['Met', 'Plant'])

    with tqdm(total=df.shape[1]) as progress_bar:
        for name, values in df.iteritems():
            mets = [x for x in values if str(x) != 'nan']
            for m in mets:
                df2.loc[len(df2.index)+1] = [m, name]
            progress_bar.update(1)

    writer = ExcelWriter(output_file_name)
    df2.to_excel(writer, 'AC', index=False)
    writer.save()


def AC_df_2_col_to_spread():
    file_name = 'data/AC_met_plant.xlsx'
    output_file_name = 'data/AC_met_plant_spread.xlsx'
    df = pd.read_excel(file_name, engine="openpyxl")
    met_names = df['Met'].unique()
    plant_names = df['Plant'].unique()
    n_row = len(plant_names)
    # print(n_row)
    df_sheet_names = DataFrame(columns=['PlantNames'], index=np.arange(n_row))
    df_sheet_names['PlantNames'] = plant_names
    n_row = len(df['Plant'].unique())
    # print(n_row)
    df_spread = DataFrame(columns=met_names, index=np.arange(n_row))
    df_spread_id = DataFrame(columns=met_names, index=np.arange(n_row))
    with tqdm(total=len(met_names)) as progress_bar:
        for name in met_names:
            nan_list = list(np.full(n_row, np.nan))
            col_list = df[df['Met'] == name].Plant.values
            nan_list[:len(col_list)] = col_list
            df_spread[name] = nan_list

            nan_list = list(np.full(n_row, np.nan))
            plant_list = df[df['Met'] == name].Plant.values
            col_list = df_sheet_names.index[df_sheet_names['PlantNames'].isin(plant_list)].tolist()
            nan_list[:len(col_list)] = col_list
            df_spread_id[name] = nan_list
            progress_bar.update(1)

    # df_spread.iloc[:, [0, 1, 2, -3, -2, -1]].head(10)
    writer = ExcelWriter(output_file_name)
    df_spread_id.to_excel(writer, 'metName_plantID', index=False)
    df_spread.to_excel(writer, 'metName_plantName', index=False)
    df_sheet_names.to_excel(writer, 'PlantNames', index=False)
    writer.save()


def LR_df_to_2_col():
    file_name = 'data/403.xlsx'
    output_file_name = 'data/LR_met_plant.xlsx'
    df = pd.read_excel(file_name, engine="openpyxl")

    df2 = pd.DataFrame(columns=['Met', 'Plant'])
    with tqdm(total=df.shape[1]) as progress_bar:
        for name, values in df.iteritems():
            mets = [x for x in values if str(x) != 'nan']
            for m in mets:
                df2.loc[len(df2.index)+1] = [m, name]
            progress_bar.update(1)

    writer = ExcelWriter(output_file_name)
    df2.to_excel(writer, 'LR', index=False)
    writer.save()


def LR_df_2_col_to_spread():
    file_name = 'data/LR_met_plant.xlsx'
    output_file_name = 'data/LR_met_plant_spread.xlsx'
    df = pd.read_excel(file_name, engine="openpyxl")
    met_names = df['Met'].unique()
    # n_row = len(df['Plant'].unique())
    # print(n_row)
    # df_spread = DataFrame(columns=met_names, index=np.arange(n_row))
    # for name in met_names:
    #     nan_list = list(np.full(n_row, np.nan))
    #     col_list = df[df['Met'] == name].Plant.values
    #     nan_list[:len(col_list)] = col_list
    #     df_spread[name] = nan_list

    plant_names = df['Plant'].unique()
    n_row = len(plant_names)
    # print(n_row)
    df_sheet_names = DataFrame(columns=['PlantNames'], index=np.arange(n_row))
    df_sheet_names['PlantNames'] = plant_names
    n_row = len(df['Plant'].unique())
    # print(n_row)
    df_spread = DataFrame(columns=met_names, index=np.arange(n_row))
    df_spread_id = DataFrame(columns=met_names, index=np.arange(n_row))
    with tqdm(total=len(met_names)) as progress_bar:
        for name in met_names:
            nan_list = list(np.full(n_row, np.nan))
            col_list = df[df['Met'] == name].Plant.values
            nan_list[:len(col_list)] = col_list
            df_spread[name] = nan_list

            nan_list = list(np.full(n_row, np.nan))
            plant_list = df[df['Met'] == name].Plant.values
            col_list = df_sheet_names.index[df_sheet_names['PlantNames'].isin(plant_list)].tolist()
            nan_list[:len(col_list)] = col_list
            df_spread_id[name] = nan_list
            progress_bar.update(1)

    # df_spread.iloc[:, [0, 1, 2, -3, -2, -1]].head(10)
    writer = ExcelWriter(output_file_name)
    df_spread_id.to_excel(writer, 'metName_plantID', index=False)
    df_spread.to_excel(writer, 'metName_plantName', index=False)
    df_sheet_names.to_excel(writer, 'PlantNames', index=False)
    writer.save()

# df_2_col_to_spread('AC','Plant')
# df_2_col_to_spread('LR','Plant')
# df_2_col_to_spread('AC','Met')
def df_2_col_to_spread(file_prefix='LR',col_name='Met'):
    if col_name=='Met':
        row_name = 'Plant'
    else:
        row_name = 'Met'
    # file_name = 'data/'+file_prefix+'_'+col_name+'_'+row_name+'.xlsx'
    file_name = 'data/'+file_prefix+'_Met_Plant.xlsx'
    output_file_name = 'data/'+file_prefix+'_'+col_name+'_'+row_name+'_spread.xlsx'
    df = pd.read_excel(file_name, engine="openpyxl")
    col_names = df[col_name].unique()

    row_names = df[row_name].unique()
    n_row = len(row_names)
    # print(n_row)
    df_sheet_names = DataFrame(columns=[row_name+'Names'], index=np.arange(n_row))
    df_sheet_names[row_name+'Names'] = row_names
    n_row = len(df[row_name].unique())
    # print(n_row)
    df_spread = DataFrame(columns=col_names, index=np.arange(n_row))
    df_spread_id = DataFrame(columns=col_names, index=np.arange(n_row))
    i = 0
    with tqdm(total=len(col_names)) as progress_bar:
        for name in col_names:
            nan_list = list(np.full(n_row, np.nan))
            col_list = df[df[col_name] == name][row_name].values
            nan_list[:len(col_list)] = col_list
            df_spread[name] = nan_list

            nan_list = list(np.full(n_row, np.nan))
            row_list = df[df[col_name] == name][row_name].values
            col_list = df_sheet_names.index[df_sheet_names[row_name+'Names'].isin(row_list)].tolist()
            col_list = [int(x) for x in col_list]
            nan_list[:len(col_list)] = col_list
            df_spread_id[name] = nan_list
            # df_spread_id[name] = pd.to_numeric(df_spread_id[name], downcast='integer', errors='ignore')
            # df_spread_id[name] = df_spread_id[name].apply(lambda x: int(x) if x == x else "")
            progress_bar.update(1)
    # print(df_spread_id.iloc[:10,:3])
    # df_spread.iloc[:, [0, 1, 2, -3, -2, -1]].head(10)
    writer = ExcelWriter(output_file_name)
    df_spread_id.to_excel(writer, col_name+'Name_'+row_name+'ID', index=False)
    df_spread.to_excel(writer, col_name+'Name_'+row_name+'Name', index=False)
    df_sheet_names.to_excel(writer, row_name+'Names', index=False)
    writer.save()

# from orangecontrib.associate.fpgrowth import *  
# import Orange
# from Orange.data.pandas_compat import table_from_frame
# # data = Orange.data.Table(df.values)
# data = table_from_frame(df)

# دلیل نادرستی استفاده از 
# mapk
# actual = [1,2,3] #[[1,2,3],[3,4,5]]
# predicted = [3,4,5]#[[1,2,4],[5,6,7],[8,9]]
# for a,p in zip(actual, predicted):
#     print(a,p)
# این تابع برای وقتی است که یک لیست از جوابهای درست و توصیه شده دیتافریم
# مثل خروجی لاتک 
# tag-recommendation