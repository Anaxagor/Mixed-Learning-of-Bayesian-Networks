import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from copy import copy
import math
from typing import List
import numpy as np
import pandas as pd
from pyBN.utils.independence_tests import mutual_information
from preprocess.discretization import get_nodes_type, code_categories
from preprocess.numpy_pandas import loc_to_DataFrame, get_type_numpy
    
def query_filter(data: pd.DataFrame, columns: List, values: List):
    """
    Filters the data according to the column-value list
            Arguments
    ----------
    *data* : pandas.DataFrame

    Returns
    -------
    *data_trim* : pandas.DataFrame
        Filtered data.

    Effects
    -------
    None
    """
    data_copy = copy(data)
    filter_str = '`' + str(columns[0]) + '`' + ' == ' + str(values[0])
    if len(columns) == 1:
        return data_copy.query(filter_str)
    else:
        for i in range(1, len(columns)):
            filter_str += ' & ' + '`' + str(columns[i]) + '`' + ' == ' + str(values[i])
        data_trim = data_copy.query(filter_str)
        return data_trim


def entropy_gauss(data):
    """
    Calculate entropy for Gaussian multivariate distributions.
            Arguments
    ----------
    *data* : numpy array

    Returns
    -------
    *entropy* : a float
        The entropy for Gaussian multivariate distributions.

    Effects
    -------
    None
    """
    if len(data[0]) > 1:
        if data.shape[0] == 1:
            var = np.var(data)
            if var > 1e-7:
                return 0.5 * (1 + math.log(var))
            else:
                return 0.0
        else:    
            var = np.linalg.det(np.cov(data))
            if var > 1e-7:
                return 0.5 * (1+ math.log(var))
            else:
                return 0.0
    else: 
        return 0.0


def mi_gauss(data, conditional=False):
    """
    Calculate Mutual Information based on entropy. 
    In the case of continuous uses entropy for Gaussian multivariate distributions.
            Arguments
    ----------
    *data* : pandas.DataFrame

    Returns
    -------
    *MI* : a float
        The Mutual Information

    Effects
    -------
    None

    Notes
    -----
    - Need to preprocess data with code_categories
    """
    if type(data) is np.ndarray:
        return mi_gauss(loc_to_DataFrame(data), conditional)
    elif type(data) is pd.DataFrame:
        nodes_type = get_nodes_type(data)
        if conditional:
            #Hill-Climbing does not use conditional MI, but other algorithms may require it
            #At the moment it counts on condition of the last row in the list of columns
            print('Warning: conditional == True')
            nodes_type_trim = copy(nodes_type)
            data_trim = copy(data)
            list_keys = list(nodes_type_trim.keys)
            del nodes_type_trim[list_keys[-1]]
            del data[list_keys[-1]]
            return (mi_gauss(data, nodes_type) - mi_gauss(data_trim, nodes_type))
        else:
            column_disc = []
            for key in nodes_type:
                if nodes_type[key] == 'disc':
                    column_disc.append(key)
            column_cont = []
            for key in nodes_type:
                if nodes_type[key] == 'cont':
                    column_cont.append(key)
            data_disc = data[column_disc]
            data_cont = data[column_cont]
            
            if len(column_cont) == 0:
                return(mutual_information(data_disc.values,conditional = False))
            else:
                H_gauss = entropy_gauss(data_cont.values.T)
                if len(column_disc) > 0:
                    dict_comb = {}
                    comb_prob = {}
                    for i in range(len(data_disc)):
                        row = data_disc.iloc[i]
                        comb = ''
                        for _, val in row.items():
                            comb = comb + str(val) + ', '
                        if not comb in dict_comb:
                            dict_comb[comb] = row
                            comb_prob[comb] = 1
                        else:
                            comb_prob[comb] += 1
                    comb_prob = {key: val/len(data_disc) for key, val in comb_prob.items()}
                    H_cond = 0.0
                    for key in list(dict_comb.keys()):
                        filtered_data = query_filter(data, column_disc, list(dict_comb[key]))
                        filtered_data = filtered_data[column_cont]
                        H_cond += comb_prob[key] * entropy_gauss(filtered_data.values.T)
                else:
                    H_cond = 0.0
                return(H_gauss-H_cond)
    


            
    

if __name__ == "__main__":
    data = pd.read_csv('./data/input_data/daks_processed.csv')
    #columns = ['Period', 'Tectonic regime', 'Hydrocarbon type']
    #columns = ['Gross', 'Netpay','Porosity']
    columns = ['Gross', 'Netpay', 'Period']
    data_test = data[columns]

    node_type = get_nodes_type(data_test)
    columns_for_discrete = []
    for param in columns:
        if node_type[param] == 'cont':
            columns_for_discrete.append(param)
    columns_for_code = []
    for param in columns:
        if node_type[param] == 'disc':
            columns_for_code.append(param)        

    data_coded, code_dict = code_categories(data_test, "label", columns_for_code)
    print(mi_gauss(data_coded.values))
