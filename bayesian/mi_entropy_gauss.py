import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from copy import copy
import math
from typing import List
import numpy as np
import pandas as pd
from external.pyBN.utils.independence_tests import mutual_information, entropy
from preprocess.discretization import get_nodes_type
from preprocess.numpy_pandas import loc_to_DataFrame
from preprocess.graph import edges_to_dict, nodes_from_edges

from preprocess.discretization import get_nodes_type, code_categories


    
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


def entropy_gauss(pd_data):
    """
    Calculate entropy for Gaussian multivariate distributions.
            Arguments
    ----------
    *data* : pd.DataFrame

    Returns
    -------
    *entropy* : a float
        The entropy for Gaussian multivariate distributions.

    Effects
    -------
    None
    """
    if not isinstance(pd_data, pd.Series):
        data = copy(pd_data).values.T 
    else:
        data = np.array(copy(pd_data)).T
    if (data.ndim == 1) | (copy(data).T.ndim == 1)|(len(data[0]) == 1)|(len(data.T[0]) == 1):
        var = np.var(data)
        if var > 1e-7:
            return 0.5 * (1 + math.log(var))
        else:
            return 0.0
    elif data.size == 0: 
        return 0.0
    elif (len(data[0]) == 1) | (len(copy(data).T[0]) == 1):
        return 0.0
    else:    

        var = np.linalg.det(np.cov(data))        
        if var > 1e-7:
            return 0.5 * (1+ math.log(var))
        else:
            return 0.0

  

def entropy_all(data):
    """
        For one varibale, H(X) is equal to the following:
            -1 * sum of p(x) * log(p(x))
        For two variables H(X|Y) is equal to the following:
            sum over x,y of p(x,y)*log(p(y)/p(x,y))
        For three variables, H(X|Y,Z) is equal to the following:
            -1 * sum of p(x,y,z) * log(p(x|y,z)),
                where p(x|y,z) = p(x,y,z)/p(y)*p(z)
    Arguments
    ----------
    *data* : pd.DataFrame
    Returns
    -------
    *H* : entropy value"""
    if type(data) is np.ndarray:
        return entropy_all(loc_to_DataFrame(data))
    elif isinstance(data, pd.Series):
        return(entropy_all(pd.DataFrame(data)))
    elif (type(data) is pd.DataFrame):
        nodes_type = get_nodes_type(data)
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
                return(entropy(data_disc.values))
        elif len(column_disc) == 0:
            return entropy_gauss(data_cont)
        else:      
            H_disc = entropy(data_disc.values)
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
                H_cond += comb_prob[key] * entropy_gauss(filtered_data)

            return(H_disc + H_cond)



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
    elif isinstance(data, pd.Series):
        return(mi_gauss(pd.DataFrame(data)))
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
            del data_trim[list_keys[-1]]
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
            elif len(column_disc) == 0:
                if len(column_cont) == 1:
                    return entropy_gauss(data_cont)
                else:
                    """data_one = data_cont[column_cont[0]]
                    column_cont_trim = copy(column_cont)
                    del column_cont_trim[0]
                    data_cont_trim = data[column_cont_trim]
                    H_gauss = entropy_gauss(data_one)+entropy_gauss(data_cont_trim)-entropy_gauss(data_cont)"""
                    H_gauss = entropy_gauss(data_cont)
                    H_cond = 0.0
            else:
                H_gauss = entropy_gauss(data_cont)
                
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
                    H_cond += comb_prob[key] * entropy_gauss(filtered_data)
                
            return(H_gauss-H_cond)

def mi(edges: list, data: pd.DataFrame):
    """
    Bypasses all nodes and summarizes scores, 
    taking into account the parent-child relationship.
            Arguments
    ----------
    *edges* : list
    *data* : pd.DataFrame

    Returns
    -------
    *sum_score* : float

    Effects
    -------
    None
    """
    parents_dict = edges_to_dict(edges)
    sum_score = 0.0
    nodes_with_edges = parents_dict.keys()
    for var in nodes_with_edges:
        child_parents = [var]
        child_parents.extend(parents_dict[var])
        sum_score += mi_gauss(copy(data[child_parents]))
    nodes_without_edges = list(set(data.columns).difference(set(nodes_with_edges)))
    for var in nodes_without_edges:
        sum_score += mi_gauss(copy(data[var]))
    return sum_score


if __name__ == '__main__':
    data = pd.read_csv('./datasets/hackathon_processed.csv')
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    #columns = ['Period', 'Tectonic regime', 'Hydrocarbon type']
    #columns = ['Gross', 'Netpay','Porosity']
    #columns = ['Gross', 'Netpay', 'Period']
    #columns = data.columns
    columns = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Hydrocarbon type', 'Gross','Netpay','Porosity','Permeability', 'Depth']
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
    data_coded

    print(mi_gauss(data_coded))
    print(mi_gauss(data_coded.values))
    print(entropy_all(data_coded))
    print(entropy_all(data_coded.values))

    edges = [('Netpay', 'Structural setting'), 
    ('Porosity', 'Hydrocarbon type')]

    print(mi(edges, data_coded[nodes_from_edges(edges)]))
    print(mi(edges, data_coded))
