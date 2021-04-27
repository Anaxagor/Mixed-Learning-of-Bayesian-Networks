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
#list_datasets = ['geo', 'healthcare', 'sangiovese', 'mehra', 'social']
list_datasets = ['mehra']
#list_datasets = ['sangiovese']

list_algo = ['HC'] #['HC', 'evo']
list_method = ['MI'] #['MI','LL']
list_disc_types = ['gauss', 'disc']

for algo in list_algo:
    for dataset in list_datasets:
        for method in list_method:
            for type in list_disc_types:
                file = ('%s_%s_%s_%s') % (algo, dataset, method, type)
                skeleton_Graph = read_structure(file)
                skeleton = dict()
                skeleton['V'] = skeleton_Graph.V 
                skeleton['E'] = skeleton_Graph.E

                data = pd.read_csv(path_dict[dataset])
                if dataset in list_for_del:
                    del data['Unnamed: 0']

                columns = data.columns
                if dataset == 'geo':
                    columns = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
                if dataset == 'social':
                    columns = ['age', 'sex', 'relation', 'mean_tr', 'median_tr', 'tr_per_month', 'is_parent', 'is_driver', 'has_pets']

                data = data[columns]
                if (dataset == 'sangiovese') | (dataset == 'solial') | (dataset == 'mehra'):
                    data = data[:500]
                data.dropna(inplace=True)
                data.reset_index(inplace=True, drop=True)
                node_type = get_nodes_type(data)

                file_param = file + '_param'
                params = parameter_learning(data, node_type, skeleton)
                save_params(params, file_param)
                params = read_params(file_param)
                bn = HyBayesianNetwork(skeleton_Graph, params)
                print(file)
                accuracy_dict, rmse_dict, real_param, pred_param = calculate_acc(bn, data, columns)
                print(accuracy_dict, rmse_dict)

