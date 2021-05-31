import operator
from copy import copy
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from external.libpgm.sampleaggregator import SampleAggregator
from preprocess.discretization import get_nodes_type
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from bayesian.metrics import norm_mean_squared_error




def calculate_acc(bn: HyBayesianNetwork, data: pd.DataFrame, columns: list, cont_error = 'rmse', sample_count = 1000) -> (dict, dict, list, list):
    """Function for calculating of params restoration accuracy

    Args:
        bn (HyBayesianNetwork): fitted BN
        data (pd.DataFrame): test dataset
        columns (list): list of params for restoration

    Returns:
        dict: accuracy score (discrete vars)
        dict: rmse score (continuous vars)
        float: density of BN graph
    """

    accuracy_dict = dict()
    rmse_dict = dict()
    pred_param = [[0 for j in range(data.shape[0])] for i in range(len(columns))]
    real_param = [[0 for j in range(data.shape[0])] for i in range(len(columns))]
    data = data[columns]
    node_type = get_nodes_type(data)
    for i in range(data.shape[0]):
        test = dict(data.iloc[i, :])
        for n, key in enumerate(columns):
            train_dict = copy(test)
            train_dict.pop(key)
            try:
                if node_type[key] == 'disc':
                    agg = SampleAggregator()
                    sample = agg.aggregate(bn.randomsample(sample_count, train_dict))
                    sorted_res = sorted(sample[key].items(), key=operator.itemgetter(1), reverse=True)
                    pred_param[n][i] = sorted_res[0][0]
                    real_param[n][i] = test[key]
                    if isinstance(real_param[n][i], np.int64):
                        pred_param[n][i] = int(pred_param[n][i])
                if node_type[key] == 'cont':
                    sample = pd.DataFrame(bn.randomsample(sample_count, train_dict))
                    if (data[key] > 0).any():
                        sample = sample.loc[sample[key] >= 0]
                    if sample.shape[0] == 0:
                        print(i)
                    else:
                        pred = np.mean(sample[key].values)
                        pred_param[n][i] = pred
                        real_param[n][i] = test[key]
            except Exception as ex:
                print(i)
    for n, key in enumerate(columns):
        if node_type[key] == 'disc':
            accuracy_dict[key] = round(accuracy_score(real_param[n], pred_param[n]), 2)
        if node_type[key] == 'cont':
            if cont_error == 'rmse':
                rmse_dict[key] = round(mean_squared_error(real_param[n], pred_param[n], squared=False), 2)
            if cont_error == 'nrmse':
                rmse_dict[key] = round(norm_mean_squared_error(real_param[n], pred_param[n], squared=False), 2)
    return accuracy_dict, rmse_dict, real_param, pred_param

