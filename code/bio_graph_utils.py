# Author: Mahmood Amintoosi
# توابع معمول موردنیاز برای عملیات گراف

from scipy.sparse import csr_matrix
from itertools import chain
from pandas import ExcelWriter
import pandas as pd
import numpy as np
# from orangecontrib.associate.fpgrowth import *
# from itertools import tee
from tqdm import tqdm
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
def graph_features(G, normalize=True):
    """
    Computer graph nodes attributes

    Parameters
    ----------
    G : nx.Graph
    normalize : boolean
        If True, All columns of the returned DataFrame will be normalized.

    Returns
    -------
    dataframe
    """    
    df = pd.DataFrame(index=G.nodes())
    # deg = G.degree()
    # deg_list = np.zeros((len(deg)))
    # for i,x in enumerate(G.degree()):
    #     deg_list[i] = x[1]
    with tqdm(total=5) as progress_bar:
        # df['degree'] = deg_list #pd.Series(deg_list)
        df['degree_cent'] = pd.Series(nx.degree_centrality(G))
        df['betweenness'] = pd.Series(nx.betweenness_centrality(G))
        df['closeness'] = pd.Series(nx.closeness_centrality(G))

        # ظاهرا هر چه کمتر باشه بهتره
        df['eccentricity'] = 1-pd.Series(nx.eccentricity(G))
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
    subG = G.subgraph(nodes=con_comp_indices).copy()
    node_names = [df.keys().format()[x] for x in con_comp_indices]
    mapping = dict(zip(con_comp_indices, node_names))
    subG = nx.relabel_nodes(subG, mapping)
    return subG


def rank_using_graph_features(subG, min_count, node_objects, edge_objects, data_dir, output_dir, working_file_name):
    print('Computing features...\n')
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
    for K in mapk_ranges:
        mapk.extend([ml_metrics.mapk(true_list, recom_list, k=K)])
    mark = []
    for K in mapk_ranges:
        mark.extend([recmetrics.mark(true_list, recom_list, k=K)])
    return [apk, mapk, mark]


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