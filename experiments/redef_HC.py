"""
**********************
Greedy Hill-Climbing
for Structure Learning
**********************

Code for Searching through the space of
possible Bayesian Network structures.

Various optimization procedures are employed,
from greedy search to simulated annealing, and 
so on - mostly using scipy.optimize.

Local search - possible moves:
- Add edge
- Delete edge
- Invert edge

Strategies to improve Greedy Hill-Climbing:
- Random Restarts
    - when we get stuck, take some number of
    random steps and then start climbing again.
- Tabu List
    - keep a list of the K steps most recently taken,
    and say that the search cannt reverse (undo) any
    of these steps.
"""
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from scipy.optimize import *
import numpy as np
import pandas as pd
from heapq import *
from copy import copy, deepcopy

from external.pyBN.classes.bayesnet import BayesNet
from external.pyBN.learning.parameter.mle import mle_estimator
from external.pyBN.learning.parameter.bayes import bayes_estimator
from external.pyBN.learning.structure.score.info_scores import log_likelihood, AIC
from bayesian.mi_entropy_gauss import mi_gauss
from external.pyBN.utils.graph import would_cause_cycle
from pomegranate import BayesianNetwork
import matplotlib.pyplot as plt
from external.pyBN.learning.structure.score.info_scores import info_score
from preprocess.discretization import get_nodes_type, discretization, code_categories
from bayesian.save_bn import save_params, save_structure, read_params, read_structure
from bayesian.train_bn import parameter_learning
from bayesian.calculate_accuracy import calculate_acc
from external.libpgm.hybayesiannetwork import HyBayesianNetwork
from experiments.redef_info_scores import log_lik_local, BIC_local as mutual_information


def hc(data, metric='LL', max_iter=100, debug=False, init_nodes=None, restriction=None, init_edges=None, remove_geo_edges=True, black_list=None):
    """
    Greedy Hill Climbing search proceeds by choosing the move
    which maximizes the increase in fitness of the
    network at the current step. It continues until
    it reaches a point where there does not exist any
    feasible single move that increases the network fitness.

    It is called "greedy" because it simply does what is
    best at the current iteration only, and thus does not
    look ahead to what may be better later on in the search.

    For computational saving, a Priority Queue (python's heapq) 
    can be used	to maintain the best operators and reduce the
    complexity of picking the best operator from O(n^2) to O(nlogn).
    This works by maintaining the heapq of operators sorted by their
    delta score, and each time a move is made, we only have to recompute
    the O(n) delta-scores which were affected by the move. The rest of
    the operator delta-scores are not affected.

    For additional computational efficiency, we can cache the
    sufficient statistics for various families of distributions - 
    therefore, computing the mutual information for a given family
    only needs to happen once.

    The possible moves are the following:
        - add edge
        - delete edge
        - invert edge

    Arguments
    ---------
    *data* : pd.DataFrame
        The data from which the Bayesian network
        structure will be learned.

    *metric* : a string
        Which score metric to use.
        Options:
            - AIC
            - BIC / MDL
            - LL (log-likelihood)

    *max_iter* : an integer
        The maximum number of iterations of the
        hill-climbing algorithm to run. Note that
        the algorithm will terminate on its own if no
        improvement is made in a given iteration.

    *debug* : boolean
        Whether to print(the scores/moves of the)
        algorithm as its happening.

    *init_nodes* : a list of initialize nodes (number of nodes according to the dataset)

    *restriction* : a list of 2-tuples
        For MMHC algorithm, the list of allowable edge additions.

    Returns
    -------
    *bn* : a BayesNet object

    """
    nrow = data.shape[0]
    ncol = data.shape[1]
    
    names = range(ncol)

    # INITIALIZE NETWORK W/ NO EDGES
    # maintain children and parents dict for fast lookups
    c_dict = dict([(n,[]) for n in names])
    p_dict = dict([(n,[]) for n in names])
    if init_edges:
        for edge in init_edges:
            c_dict[edge[0]].append(edge[1])
            p_dict[edge[1]].append(edge[0])

    score_list = []
    # COMPUTE INITIAL LIKELIHOOD SCORE	
    value_dict = dict([(n, np.unique(data.values[:,i])) for i,n in enumerate(names)])
    bn = BayesNet(c_dict)
    columns = data.columns
    node_type = get_nodes_type(data)
    columns_for_discrete = []
    for param in columns:
        if node_type[param] == 'cont':
            columns_for_discrete.append(param)
    columns_for_code = []
    for param in columns:
        if node_type[param] == 'disc':
            columns_for_code.append(param) 
    
    
    
    
    #mle_estimator(bn, data_discreted.values)
    #max_score = info_score(bn, data_discreted.values, metric)

    data = data.values
 

    
    

    # CREATE EMPIRICAL DISTRIBUTION OBJECT FOR CACHING
    #ED = EmpiricalDistribution(data,names)

    

    _iter = 0
    improvement = True

    while improvement:
        improvement = False
        max_delta = 0

        if debug:
            print('ITERATION: ' , _iter)
        
        ### TEST ARC ADDITIONS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v not in c_dict[u] and u!=v and not would_cause_cycle(c_dict, u, v) and len(p_dict[v]) != 3:
                    # FOR MMHC ALGORITHM -> Edge Restrictions
                    if (init_nodes is None or  not(v in init_nodes)) and (restriction is None or (u,v) in restriction) and (black_list is None or not((u,v) in black_list)):
                        # SCORE FOR 'V' -> gaining a parent
                        old_cols = (v,) + tuple(p_dict[v]) # without 'u' as parent
                        mi_old = mutual_information(data[:,old_cols])
                        new_cols = old_cols + (u,) # with'u' as parent
                        mi_new = mutual_information(data[:,new_cols])
                        delta_score = nrow * (mi_old - mi_new)
                        
                        if delta_score > max_delta:
                            if debug:
                                print('Improved Arc Addition: ' , (u,v))
                                print('Delta Score: ' , delta_score)
                            max_delta = delta_score
                            max_operation = 'Addition'
                            max_arc = (u,v)
                            

        # ### TEST ARC DELETIONS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v in c_dict[u]:
                    # SCORE FOR 'V' -> losing a parent
                    old_cols = (v,) + tuple(p_dict[v]) # with 'u' as parent
                    mi_old = mutual_information(data[:,old_cols])
                    new_cols = tuple([i for i in old_cols if i != u]) # without 'u' as parent
                    mi_new = mutual_information(data[:,new_cols])
                    delta_score = nrow * (mi_old - mi_new)

                    if (delta_score > max_delta):
                        if init_edges == None:
                            if debug:
                                print('Improved Arc Deletion: ' , (u,v))
                                print('Delta Score: ' , delta_score)
                            max_delta = delta_score
                            max_operation = 'Deletion'
                            max_arc = (u,v)
                        else:
                            if (u, v) in init_edges:
                                if remove_geo_edges:
                                    if debug:
                                        print('Improved Arc Deletion: ' , (u,v))
                                        print('Delta Score: ' , delta_score)
                                    max_delta = delta_score
                                    max_operation = 'Deletion'
                                    max_arc = (u,v)
                            else:
                                if debug:
                                    print('Improved Arc Deletion: ' , (u,v))
                                    print('Delta Score: ' , delta_score)
                                max_delta = delta_score
                                max_operation = 'Deletion'
                                max_arc = (u,v)

        # ### TEST ARC REVERSALS ###
        for u in bn.nodes():
            for v in bn.nodes():
                if v in c_dict[u] and not would_cause_cycle(c_dict,v,u, reverse=True) and len(p_dict[u])!=3 and (init_nodes is None or not (u in init_nodes)) and (restriction is None or (v,u) in restriction) and (black_list is None or not((v,u) in black_list)):
                    # SCORE FOR 'U' -> gaining 'v' as parent
                    old_cols = (u,) + tuple(p_dict[v]) # without 'v' as parent
                    mi_old = mutual_information(data[:,old_cols])
                    new_cols = old_cols + (v,) # with 'v' as parent
                    mi_new = mutual_information(data[:,new_cols])
                    delta1 = -1* nrow * (mi_old - mi_new)
                    # SCORE FOR 'V' -> losing 'u' as parent
                    old_cols = (v,) + tuple(p_dict[v]) # with 'u' as parent
                    mi_old = mutual_information(data[:,old_cols])
                    new_cols = tuple([u for i in old_cols if i != u]) # without 'u' as parent
                    mi_new = mutual_information(data[:,new_cols])
                    delta2 = nrow * (mi_old - mi_new)
                    # COMBINED DELTA-SCORES
                    delta_score = delta1 + delta2

                    if (delta_score > max_delta):
                        if init_edges == None:
                            if debug:
                                print('Improved Arc Reversal: ' , (u,v))
                                print('Delta Score: ' , delta_score)
                            max_delta = delta_score
                            max_operation = 'Reversal'
                            max_arc = (u,v)
                        else:
                            if (u, v) in init_edges:
                                if remove_geo_edges:
                                    if debug:
                                        print('Improved Arc Reversal: ' , (u,v))
                                        print('Delta Score: ' , delta_score)
                                    max_delta = delta_score
                                    max_operation = 'Reversal'
                                    max_arc = (u,v)
                            else:
                                if debug:
                                    print('Improved Arc Reversal: ' , (u,v))
                                    print('Delta Score: ' , delta_score)
                                max_delta = delta_score
                                max_operation = 'Reversal'
                                max_arc = (u,v)






                        

        if max_delta != 0:
            improvement = True
            u,v = max_arc
            if max_operation == 'Addition':
                if debug:
                    print('ADDING: ' , max_arc , '\n')
                c_dict[u].append(v)
                p_dict[v].append(u)
                
            elif max_operation == 'Deletion':
                if debug:
                    print('DELETING: ' , max_arc , '\n')
                c_dict[u].remove(v)
                p_dict[v].remove(u)
                
            elif max_operation == 'Reversal':
                if debug:
                    print('REVERSING: ' , max_arc, '\n')
                c_dict[u].remove(v)
                p_dict[v].remove(u)
                c_dict[v].append(u)
                p_dict[u].append(v)
            
        else:
            if debug:
                print('No Improvement on Iter: ' , _iter)

        ### TEST FOR MAX ITERATION ###
        _iter += 1
        if _iter > max_iter:
            if debug:
                print('Max Iteration Reached')
            break

    
    bn = BayesNet(c_dict)
    
    
    return bn

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
    
    healthcare = pd.read_csv('./datasets/sangiovese.csv')
    del healthcare['Unnamed: 0']
    #healthcare.to_csv('../datasets/healthcare1.csv')
    """healthcare = healthcare.iloc[0:500]
    columns = healthcare.columns
    print(columns)
    healthcare.dropna(inplace=True)
    healthcare.reset_index(inplace=True, drop=True)
    data_test = healthcare"""

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
    if columns_for_discrete != []:
        data_discereted, est = discretization(data_coded, "kmeans", columns_for_discrete)
    else:
        data_discereted = data_coded
    
    
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
    

    """column_name_dict = dict([(n, i) for i, n in enumerate(columns_for_code)])
    bn = hc(data_coded[columns_for_code])
    structure = []
    nodes = sorted(list(bn.nodes()))
    for rv in nodes:
        for pa in bn.F[rv]['parents']:
            structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                            list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
    skeleton_disc = dict()
    skeleton_disc['V'] = columns_for_code
    
    skeleton_disc['E'] = structure
    print(skeleton_disc)

    column_name_dict = dict([(n, i) for i, n in enumerate(columns_for_discrete)])
    bn = hc(data_discereted[columns_for_discrete])
    structure = []
    nodes = sorted(list(bn.nodes()))
    for rv in nodes:
        for pa in bn.F[rv]['parents']:
            structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                            list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
    skeleton_cont = dict()
    skeleton_cont['V'] = columns_for_discrete
    
    skeleton_cont['E'] = structure
    print(skeleton_cont)"""



    column_name_dict = dict([(n, i) for i, n in enumerate(list(columns))])
    
    bn = hc(data_coded, black_list=blacklist_new)
    structure = []
    nodes = sorted(list(bn.nodes()))
    for rv in nodes:
        for pa in bn.F[rv]['parents']:
            structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                            list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
    skeleton = dict()
    skeleton['V'] = list(columns)
    
    skeleton['E'] = structure

    #skeleton = {'V': ['Tectonic regime', 'Period', 'Lithology', 'Hydrocarbon type', 'Structural setting', 'Gross', 'Netpay', 'Porosity', 'Permeability', 'Depth'], 'E': [['Tectonic regime', 'Depth'], ['Tectonic regime', 'Netpay'], ['Period', 'Lithology'], ['Hydrocarbon type', 'Permeability'], ['Hydrocarbon type', 'Gross'], ['Hydrocarbon type', 'Porosity'], ['Hydrocarbon type', 'Period'], ['Hydrocarbon type', 'Structural setting'], ['Hydrocarbon type', 'Lithology'], ['Hydrocarbon type', 'Tectonic regime'], ['Structural setting', 'Lithology']]}
    print(skeleton)

    # save_structure(skeleton, 'test')
    # skel = read_structure('test')

    

    # params = parameter_learning(data_test, node_type, skeleton)
    # save_params(params, 'test_param')
    # params = read_params('test_param')
    # bn = HyBayesianNetwork(skel, params)
    # print(calculate_acc(bn, data_test, columns))

    """save_structure(skeleton, 'sangiovese')
    skel = read_structure('sangiovese')

    params = parameter_learning(data_test, node_type, skeleton)
    save_params(params, 'sangiovese_param')
    params = read_params('sangiovese_param')
    bn = HyBayesianNetwork(skel, params)
    print(calculate_acc(bn, data_test, columns))"""












