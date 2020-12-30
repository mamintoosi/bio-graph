# Author: Mahmood Amintoosi
## کاوش الگوهای پرتکرار

from pandas import ExcelWriter
import pandas as pd
import numpy as np
# from orangecontrib.associate.fpgrowth import *
# from itertools import tee
from tqdm import tqdm
import networkx as nx
from pandas import DataFrame 
pd.options.display.float_format = "{:.2f}".format
import random

from bio_graph_utils import *
from fim_utils import fim_bio


data_dir = 'data/'
output_dir = 'results/'
working_dir = 'data/'

# working_file_name = 'AC_restructured.xlsx'
# GT_file_name = '403.xlsx'
# node_objects = 'plant'
# edge_objects = 'met'

working_file_name = 'AC_Plant_Met_spread.xlsx'
GT_file_name = 'LR_Plant_Met_spread.xlsx'
node_objects = 'plant'
edge_objects = 'met'

# working_file_name = 'LR_met_plant_spread.xlsx'
# GT_file_name = 'AC_met_plant_spread.xlsx'
# node_objects = 'met'
# edge_objects = 'plant'

working_file = working_dir+working_file_name
df = pd.read_excel(working_file, engine="openpyxl") 
for col in df.columns:
   df[col] = df[col].apply(lambda x: float(x) if not pd.isna(x) else x)
    # df[col] = pd.to_numeric(df[col], downcast='integer', errors='ignore')
#    df[col] = df[col].apply(lambda x: int(x) if isinstance(x, float) else x)

# نگهداری ستونهای با بیش از ۵ عنصر
min_count = 4
# حداقل تعداد تکرار برای مجموعه اقلام مکرر
minFreq = 4

# نگهداری ستونهای با بیش از مین‌کانت عنصر
cols_to_preserve = np.where((df.count()>=min_count).values == True)[0]
df = df.iloc[:,cols_to_preserve]
# این قسمت باید با ساب گراف ان ایکس بازنویسی باشه
# subgraph
# ایجاد دیتافریمها هم میتونه بازنویسی بشه که گراف مستقیم از دیتافریم خونده بشه

# در فرم هدر نام گیاه، میشه نام گیاهان
col_names = list(df.head(0))
n_nodes = len(col_names)
# print('Sahpe of dataframe: ',df.shape)
# print(len(plantNames))
# print(df.iloc[:,[0,1,2,-3,-2,-1]].head(10))

# ایجاد کیسه کلمات
T, bow, featureNames = bow_nodes(df)

# ایجاد گراف از روی کیسه کلمات
G = make_graph_from_bow(bow)

# پیدا کردن بزرگترین زیرگراف همبند
print('Computing the largest connected graph...\n')
subG = largest_con_com(df,G)
print('Number of nodes sub graph:', len(subG.nodes()))
# nx.draw_shell(subG,with_labels=True)

gf_df, gf_df_sorted = rank_using_graph_features(subG,min_count,node_objects,edge_objects,data_dir,output_dir,working_file_name)
# df[list(subG)]
# مجموعه اقلام مکرر
df_subG = df[list(subG)]
T, bow, featureNames = bow_nodes(df_subG)

sorted_nodes_idx, sorted_nodes_idx_w, G, degreeG = fim_bio(minFreq,T,bow,featureNames)
# sorted_nodes_idx, sorted_nodes_idx_w, G, degreeG = fim_bio(minFreq,df_subG,node_objects,edge_objects,output_dir,working_file_name)
list_subG = list(subG)
# print([list_subG[x] for x in sorted_nodes_idx])
node_names = [df_subG.keys().format()[x] for x in sorted_nodes_idx]
node_names_w = [df_subG.keys().format()[x] for x in sorted_nodes_idx_w]
# mapping = dict(zip(sorted_nodes_idx, node_names))
# new_subG = nx.relabel_nodes(subG,mapping)
# new_df = pd.DataFrame(index = new_subG.nodes())

#استفاده از گراف مجموعه اقلام مکرر
gf_fim_df = gf_df.loc[:, gf_df.columns != 'features_sum']
min_max_scaler = preprocessing.MinMaxScaler()
# numpy_matrix = df.values
X = min_max_scaler.fit_transform(degreeG.reshape(-1,1))
gf_fim_df['degree_fim'] = X#degreeG#pd.Series(degreeG)
features_sum = gf_fim_df.sum(axis=1)
gf_fim_df['features_sum'] = features_sum

# gf_fim_df
gf_fim_df_sorted = gf_fim_df.sort_values(by='features_sum', ascending=False)
# gf_fim_df_sorted

s1 = gf_fim_df_sorted['degree_cent']
s2 = gf_fim_df_sorted['degree_fim']
print('Correlation of degree and degree_fim:',s1.corr(s2))

random_order = node_names.copy()
random.shuffle(random_order)

GT_file = working_dir+GT_file_name
true_df = pd.read_excel(GT_file, engine="openpyxl") 
true_list = list(true_df.keys().values)
index = np.arange(1,50,2)
[apk_fim,mapk_fim,mark_fim] = compute_metrics(true_list, node_names)
[apk_fim_w,mapk_fim_w,mark_fim_w] = compute_metrics(true_list, node_names_w)
[apk_gf,mapk_gf,mark_gf] = compute_metrics(true_list, list(gf_df_sorted.index.values))
[apk_gf_fim,mapk_gf_fim,mark_gf_fim] = compute_metrics(true_list, list(gf_fim_df_sorted.index.values))
[apk_rand,mapk_rand,mark_rand] = compute_metrics(true_list, random_order)
# apk_fim
# [apk_fim_w,mapk_fim_w,mark_fim_w] = compute_metrics(AC_plants_order_by_metabolit_numbers, fim_w_df[1].values)
# [apk_gf,mapk_gf,mark_gf] = compute_metrics(AC_plants_order_by_metabolit_numbers, plants_order_by_features)
# [apk_rt,mapk_rt,mark_rt] = compute_metrics(AC_plants_order_by_metabolit_numbers, RT_plants_order_by_metabolit_numbers)

names = ['Features', 'Frequently Item Set','Frequently Item Set w','Hybrid','Random']#, 'RT']

scores = [mapk_gf, mapk_fim, mapk_fim_w, mapk_gf_fim,mapk_rand]#, mapk_rt]
fig = plt.figure(figsize=(10, 4))
recmetrics.mapk_plot(scores, model_names=names, k_range=index)

scores = [mark_gf, mark_fim, mark_fim_w,mark_gf_fim, mark_rand]
fig = plt.figure(figsize=(10, 4))
recmetrics.mark_plot(scores, model_names=names, k_range=index)

scores = [apk_gf, apk_fim, apk_fim_w,apk_gf_fim, apk_rand]
fig = plt.figure(figsize=(10, 4))
recmetrics.mapk_plot(scores, model_names=names, k_range=index)
png_file_name = 'results/{}_{}_{}_{}_{}_{}.png'.format(working_file_name[:2],GT_file_name[:2],node_objects,\
    edge_objects,str(min_count),str(minFreq))
fig.savefig(png_file_name)
# for col in df.columns:
#     print(set(df[col].dropna()) & set([1. , 2. ,6.]))

file_name = 'results/{}_{}_{}_{}_{}_{}.xlsx'.format(working_file_name[:2],GT_file_name[:2],node_objects,\
    edge_objects,str(min_count),str(minFreq))
writer = ExcelWriter(file_name)
gf_df_sorted.to_excel(writer, sheet_name='gf_df_sorted')  # , index=False)
gf_fim_df_sorted.to_excel(writer, sheet_name='gf_fim_df_sorted')  # , index=False)
gf_df.to_excel(writer, sheet_name='gf_df')  # , index=False)
writer.save()
