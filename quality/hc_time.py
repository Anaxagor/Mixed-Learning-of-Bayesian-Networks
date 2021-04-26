import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import pandas as pd
from bayesian.train_bn import structure_learning, parameter_learning
from preprocess.discretization import get_nodes_type, discretization, inverse_discretization, code_categories
from bayesian.save_bn import save_structure, save_params, read_structure, read_params
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from visualization.visualization import draw_BN
from bayesian.calculate_accuracy import calculate_acc
from experiments.redef_HC import hc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import time

path_dict = {'geo': './datasets/hackathon_processed.csv',
'healthcare':  './datasets/healthcare.csv',
'sangiovese': './datasets/sangiovese.csv',
'mehra': './datasets/mehra.csv',
'social': './datasets/vk_interests_finance.csv'
}

list_for_del = ['healthcare', 'sangiovese', 'mehra']


list_datasets = ['social']
list_method = ['MI', 'LL']

for name in list_datasets:
    #data = pd.read_csv('./datasets/hackathon_processed.csv')
    #data = pd.read_csv('./datasets/healthcare.csv')
    data = pd.read_csv(path_dict[name])
    if name in list_for_del:
        del data['Unnamed: 0']

    columns = data.columns
    if name == 'geo':
        columns = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
    if name == 'social':
        columns = ['age', 'sex', 'relation', 'mean_tr', 'median_tr', 'tr_per_month', 'is_parent', 'is_driver', 'has_pets']

    data = data[columns]
    if (name == 'solial') | (name == 'mehra'):
        data = data[:2000]
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    node_type = get_nodes_type(data)

    columns_for_discrete = []
    for param in columns:
        if node_type[param] == 'cont':
            columns_for_discrete.append(param)
    columns_for_code = []
    for param in columns:
        if node_type[param] == 'disc':
            columns_for_code.append(param) 

    data_coded, label_coder = code_categories(data, 'label', columns_for_code)
    data_discrete, coder = discretization(data_coded, 'kmeans', columns_for_discrete)

    datacol = data_coded.columns.to_list()
        
        
    blacklist = []
    for x in datacol:
        for y in datacol:
            if x != y:
                if (node_type[x] == 'cont') & (node_type[y] == 'disc'):
                    blacklist.append((x, y))
    column_name_dict = dict([(n, i) for i, n in enumerate(datacol)])
    blacklist_new = []
    for pair in blacklist:
        blacklist_new.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
    
    for method in list_method:
        print("Dataset: %s, method: %s" % (name, method))
        start_time = time.time()
        #bn = structure_learning(data_coded, 'HC', 'LL', node_type)
        bn = hc(data_coded, metric = method, black_list=blacklist_new)
        print("Gauss--- %s seconds ---" % (time.time() - start_time))
  
        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            for pa in bn.F[rv]['parents']:
                structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                                list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
        skeleton = dict()
        skeleton['V'] = list(columns) 
        skeleton['E'] = structure
        path = ('HC_%s_%s_gauss') % (name, method)
        save_structure(skeleton, path)

        start_time = time.time()
        #bn = structure_learning(data_discrete, 'HC', 'LL', node_type)
        bn = hc(data_discrete, metric = method, black_list=blacklist_new)
        print("Disc--- %s seconds ---" % (time.time() - start_time))

        structure = []
        nodes = sorted(list(bn.nodes()))
        for rv in nodes:
            for pa in bn.F[rv]['parents']:
                structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                                list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
        skeleton = dict()
        skeleton['V'] = list(columns) 
        skeleton['E'] = structure
        path = ('HC_%s_%s_disc') % (name, method)
        save_structure(skeleton, path)