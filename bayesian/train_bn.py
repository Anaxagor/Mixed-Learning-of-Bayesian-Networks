import itertools
from copy import copy
import math
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import K2Score
from pomegranate import DiscreteDistribution, ConditionalProbabilityTable
from scipy.stats import norm
from sklearn import linear_model
from bayesian.structure_score import MIG
from sklearn import mixture
from external.pyBN.learning.structure.score.hill_climbing import hc as hc_method


import datetime
import os
import random
import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc
from pgmpy.models import BayesianModel
from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.chains.chain import Chain
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import project_root
from pgmpy.estimators import K2Score
from bayesian.mi_entropy_gauss import mi
from sklearn.svm import SVR
from sklearn.cluster import KMeans





random.seed(1)
np.random.seed(1)

def K2(chain: Chain, reference_data: pd.DataFrame) -> float:
    nodes = reference_data.columns.to_list()
    graph, labels = chain_as_nx_graph(chain)
    struct = []
    for pair in graph.edges():
        struct.append((str(labels[pair[0]]), str(labels[pair[1]])))
    BN_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in BN_model.nodes():
            no_nodes.append(node)
    
    score = K2Score(reference_data).score(BN_model) #+ 1000*len(no_nodes)
    return score


def MI(chain: Chain, reference_data: pd.DataFrame) -> float:
    nodes = reference_data.columns.to_list()
    graph, labels = chain_as_nx_graph(chain)
    struct = []
    for pair in graph.edges():
        struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
    score = mi(struct, reference_data)
    return score


def run_BN_evo_K2(data: pd.DataFrame, max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5), is_visualise=False, with_tuning=False) -> Chain: 
    available_nodes = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
    composer_requirements = GPComposerRequirements(
            primary=available_nodes,
            secondary=available_nodes, max_arity=6,
            max_depth=3, pop_size=20, num_of_generations=50,
            crossover_prob=0.8, mutation_prob=0.9, max_lead_time=max_lead_time, add_single_model_chains=False)
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)
    task = Task(TaskTypesEnum.regression)
    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(K2).with_optimiser_parameters(optimiser_parameters)
    composer = builder.build()
    chain_evo_composed = composer.compose_chain(data=data)
    return chain_evo_composed


def run_BN_evo_MI(data: pd.DataFrame, max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5), is_visualise=False, with_tuning=False) -> Chain: 
    available_nodes = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
    composer_requirements = GPComposerRequirements(
            primary=available_nodes,
            secondary=available_nodes, max_arity=6,
            max_depth=3, pop_size=20, num_of_generations=50,
            crossover_prob=0.8, mutation_prob=0.9, max_lead_time=max_lead_time, add_single_model_chains=False)
    scheme_type = GeneticSchemeTypesEnum.steady_state
    optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)
    task = Task(TaskTypesEnum.regression)
    builder = GPComposerBuilder(task).with_requirements(composer_requirements).with_metrics(MI).with_optimiser_parameters(optimiser_parameters)
    composer = builder.build()
    chain_evo_composed = composer.compose_chain(data=data)
    return chain_evo_composed
    






def structure_learning(data: pd.DataFrame, search: str, score: str, node_type: dict, init_nodes: list = None,
                       white_list: list = None,
                       init_edges: list = None, remove_init_edges: bool = True, black_list: list = None) -> dict:
    """Function for bayesian networks structure learning

    Args:
        data (pd.DataFrame): input encoded and discretized data
        search (str): search strategy (HC, evo)
        score (str): algorith of learning (K2, MI, MI_mixed)
        node_type (dict): dictionary with node types (discrete or continuous)
        init_nodes (list, optional): nodes with no parents. Defaults to None.
        white_list (list, optional): allowable edges. Defaults to None.
        init_edges (list, optional): start edges of graph (set user). Defaults to None.
        remove_init_edges (bool, optional): flag that allow to delete start edges (or not allow). Defaults to True.
        black_list (list, optional): forbidden edges. Defaults to None.

    Returns:
        dict: dictionary with structure (values are lists of nodes and edges)
    """
    blacklist = []
    datacol = data.columns.to_list()
    if init_nodes:
        blacklist = [(x, y) for x in datacol for y in init_nodes if x != y]
    for x in datacol:
        for y in datacol:
            if x != y:
                if (node_type[x] == 'cont') & (node_type[y] == 'disc'):
                    blacklist.append((x, y))
    if black_list:
        blacklist = blacklist + black_list

    skeleton = dict()
    skeleton['V'] = datacol

    if search == 'HC':
        if score == "MI":
            column_name_dict = dict([(n, i) for i, n in enumerate(datacol)])
            blacklist_new = []
            for pair in blacklist:
                blacklist_new.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            if white_list:
                white_list_old = copy(white_list)
                white_list = []
                for pair in white_list_old:
                    white_list.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            if init_edges:
                init_edges_old = copy(init_edges)
                init_edges = []
                for pair in init_edges_old:
                    init_edges.append((column_name_dict[pair[0]], column_name_dict[pair[1]]))
            bn = hc_method(data.values, restriction=white_list, init_edges=init_edges, remove_geo_edges=remove_init_edges, black_list=blacklist_new)
            structure = []
            nodes = sorted(list(bn.nodes()))
            for rv in nodes:
                for pa in bn.F[rv]['parents']:
                    structure.append([list(column_name_dict.keys())[list(column_name_dict.values()).index(pa)],
                                  list(column_name_dict.keys())[list(column_name_dict.values()).index(rv)]])
            skeleton['E'] = structure
        if score == "K2":
            hc_K2Score = HillClimbSearch(data, scoring_method=K2Score(data))
            if init_edges == None:
                best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
                else:
                    best_model_K2Score = hc_K2Score.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
            structure = [list(x) for x in list(best_model_K2Score.edges())]
            skeleton['E'] = structure
        if score == 'MI_mixed':
            hc_mi_mixed = HillClimbSearch(data, scoring_method=MIG(data=data))
            if init_edges == None:
                best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list)
            else:
                if remove_init_edges:
                    startdag = DAG()
                    startdag.add_nodes_from(nodes=datacol)
                    startdag.add_edges_from(ebunch=init_edges)
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         start_dag=startdag)
                else:
                    best_model_mi_mixed = hc_mi_mixed.estimate(black_list=blacklist, white_list=white_list,
                                                         fixed_edges=init_edges)
            structure = [list(x) for x in list(best_model_mi_mixed.edges())]
            skeleton['E'] = structure
    if search == 'evo':

        if score == "MI":
            chain = run_BN_evo_MI(data)
            graph, labels = chain_as_nx_graph(chain)
            struct = []
            for pair in graph.edges():
                struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
            skeleton['E'] = struct
       
        if score == "K2":
            chain = run_BN_evo_K2(data)
            graph, labels = chain_as_nx_graph(chain)
            struct = []
            for pair in graph.edges():
                struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
            skeleton['E'] = struct

    return skeleton

def parameter_learning(data: pd.DataFrame, node_type: dict, skeleton: dict) -> dict:
    """Function for parameter learning for hybrid BN
    Args:
        data (pd.DataFrame): input dataset
        node_type (dict): dictionary with node types (discrete or continuous)
        skeleton (dict): structure of BN
    Returns:
        dict: dictionary with parameters of distributions in nodes
    """
    datacol = data.columns.to_list()
    node_data = dict()
    node_data['Vdata'] = dict()
    for node in datacol:
        children = []
        parents = []
        for edge in skeleton['E']:
            if (node in edge):
                if edge.index(node) == 0:
                    children.append(edge[1])
                if edge.index(node) == 1:
                    parents.append(edge[0])
        if (node_type[node] == "disc") & (len(parents) == 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            cprob = list(dict(sorted(dist.items())).values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": None}
        if (node_type[node] == "disc") & (len(parents) != 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            dist = ConditionalProbabilityTable.from_samples(data[parents + [node]].values)
            params = dist.parameters[0]
            cprob = dict()
            for i in range(0, len(params), len(vals)):
                probs = []
                for j in range(i, (i + len(vals))):
                    probs.append(params[j][-1])
                combination = [str(x) for x in params[i][0:len(parents)]]
                cprob[str(combination)] = probs
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": None}
        if (node_type[node] == "cont") & (len(parents) == 0):
            # mean_base, std = norm.fit(data[node].values)
            # variance = std ** 2
            mean_base = np.mean(data[node].values)
            variance = np.var(data[node].values)
            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "type": "lg", "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "type": "lg", "children": None}
        if (node_type[node] == "cont") & (len(parents) != 0):
            disc_parents = []
            cont_parents = []
            for parent in parents:
                if node_type[parent] == 'disc':
                    disc_parents.append(parent)
                else:
                    cont_parents.append(parent)

            if (len(disc_parents) == 0):
                # mean_base, std = norm.fit(data[node].values)
                # variance = std ** 2
                # mean_base = np.mean(data[node].values)
                # variance = np.var(data[node].values)
                #model = linear_model.BayesianRidge()
                model = linear_model.LinearRegression()
                predict = []
                if len(parents) == 1:
                    model.fit(np.transpose([data[parents[0]].values]), data[node].values)
                    predict = model.predict(np.transpose([data[parents[0]].values]))
                else:
                    model.fit(data[parents].values, data[node].values)
                    predict = model.predict(data[parents].values)
                variance = (RSE(data[node].values, predict)) ** 2
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "type": "lg",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": model.intercept_, "mean_scal": list(model.coef_),
                                                "parents": parents, "variance": variance, "type": "lg",
                                                "children": None}
            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    mean_base = np.nan
                    variance = np.nan
                    predict = []
                    # if new_data.shape[0] != 0:
                    #     #mean_base, std = norm.fit(new_data[node].values)
                    #     #variance = std ** 2
                    #     mean_base = np.mean(new_data[node].values)
                    #     variance = np.var(new_data[node].values)
                    if new_data.shape[0] != 0:
                        #model = linear_model.BayesianRidge()
                        model = linear_model.LinearRegression()
                        if len(cont_parents) == 1:
                            model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                            predict = model.predict(np.transpose([new_data[cont_parents[0]].values]))
                        else:
                            model.fit(new_data[cont_parents].values, new_data[node].values)
                            predict = model.predict(new_data[cont_parents].values)
                        key_comb = [str(x) for x in comb]
                        mean_base = model.intercept_
                        variance = (RSE(new_data[node].values, predict)) ** 2
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base,
                                                  'mean_scal': list(model.coef_)}
                    else:
                        scal = list(np.full(len(cont_parents), np.nan))
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    if new_data.shape[0] != 0:
                        #mean_base, std = norm.fit(new_data[node].values)
                        #variance = std ** 2
                        mean_base = np.mean(new_data[node].values)
                        variance = np.var(new_data[node].values)
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                    else:
                        mean_base = np.nan
                        variance = np.nan
                        key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}

    return node_data


def parameter_learning_mix(data: pd.DataFrame, node_type: dict, skeleton: dict) -> dict:
    """Function for parameter learning for hybrid BN

    Args:
        data (pd.DataFrame): input dataset
        node_type (dict): dictionary with node types (discrete or continuous)
        skeleton (dict): structure of BN

    Returns:
        dict: dictionary with parameters of distributions in nodes
    """
    datacol = data.columns.to_list()
    node_data = dict()
    node_data['Vdata'] = dict()
    cont_columns = []
    for key in node_type.keys():
        if node_type[key] == 'cont':
            cont_columns.append(key)
    gmm_params = dict()
    N = 5
    gmm = mixture.GaussianMixture(n_components=N, covariance_type='full')
    gmm.fit(data[cont_columns].values)
    for index, column in enumerate(cont_columns):
        means = []
        for component in range (N):
            means.append(gmm.means_[component][index])
        gmm_params[column] = means
            
        
    for node in datacol:
        children = []
        parents = []
        for edge in skeleton['E']:
            if (node in edge):
                if edge.index(node) == 0:
                    children.append(edge[1])
                if edge.index(node) == 1:
                    parents.append(edge[0])

        if (node_type[node] == "disc") & (len(parents) == 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            cprob = list(dict(sorted(dist.items())).values())
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": None, "vals": vals,
                                            "type": "discrete", "children": None}
        if (node_type[node] == "disc") & (len(parents) != 0):
            numoutcomes = int(len(data[node].unique()))
            dist = DiscreteDistribution.from_samples(data[node].values)
            vals = sorted([str(x) for x in list(dist.parameters[0].keys())])
            dist = ConditionalProbabilityTable.from_samples(data[parents + [node]].values)
            params = dist.parameters[0]
            cprob = dict()
            for i in range(0, len(params), len(vals)):
                probs = []
                for j in range(i, (i + len(vals))):
                    probs.append(params[j][-1])
                combination = [str(x) for x in params[i][0:len(parents)]]
                cprob[str(combination)] = probs
            if (len(children) != 0):
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": children}
            else:
                node_data['Vdata'][node] = {"numoutcomes": numoutcomes, "cprob": cprob, "parents": parents,
                                            "vals": vals, "type": "discrete", "children": None}
        if (node_type[node] == "cont") & (len(parents) == 0):
            # mean_base, std = norm.fit(data[node].values)
            # variance = std ** 2
            mean_base = np.mean(data[node].values)
            variance = np.var(data[node].values)
            if (len(children) != 0):
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "type": "lg", "children": children}
            else:
                node_data['Vdata'][node] = {"mean_base": mean_base, "mean_scal": [], "parents": None,
                                            "variance": variance, "type": "lg", "children": None}
        if (node_type[node] == "cont") & (len(parents) != 0):
            disc_parents = []
            cont_parents = []
            for parent in parents:
                if node_type[parent] == 'disc':
                    disc_parents.append(parent)
                else:
                    cont_parents.append(parent)
            

            if (len(disc_parents) == 0) & (len(cont_parents) != 0):
                # mean_base, std = norm.fit(data[node].values)
                # variance = std ** 2
                # mean_base = np.mean(data[node].values)
                # variance = np.var(data[node].values)
                #model = linear_model.BayesianRidge()
                #model = linear_model.LinearRegression()
                #model = SVR(kernel='linear', C=100, gamma='auto')
                #gmm = mixture.GaussianMixture(n_components=10, covariance_type='full')
                
                # predict = []
                # if len(parents) == 1:
                #     model.fit(np.transpose([data[parents[0]].values]), data[node].values)
                #     predict = model.predict(np.transpose([data[parents[0]].values]))
                # else:
                #     model.fit(data[parents].values, data[node].values)
                #     predict = model.predict(data[parents].values)
               
                #gmm.fit(data[parents+[node]].values)

                #variance = (RSE(data[node].values, predict)) ** 2
                # means_parent = [[] for i in range (N)]#[list(l)[0:-1] for l in list(gmm.means_)]
                # for c in range(N):
                #     for c_p in parents:
                #         means_parent[c].append(gmm_params[c_p][c])

                # mean_node = gmm_params[node]#[list(l)[-1] for l in list(gmm.means_)]
                mean_node = []
                means_parent = []
                #labels = gmm.predict(data[parents+[node]].values)
                labels = gmm.predict(data[cont_columns].values)
                diff_data = copy(data)
                diff_data['labels'] = labels
                variances = []
                for i in range(0,N,1):
                    sample = diff_data.loc[diff_data['labels'] == i]
                    variances.append(np.var(sample[node].values))
                    mean_node.append(np.mean(sample[node].values))
                    parent_one = []
                    for p in parents:
                        parent_one.append(np.mean(sample[p].values))
                    means_parent.append(parent_one)
                # for var in variances:
                #     if (str(var) == 'nan'):
                #         variances[variances.index(var)] = 0

                if (len(children) != 0):
                    node_data['Vdata'][node] = {"mean_base": mean_node, "mean_scal": means_parent,
                                                "parents": parents, "variance": variances, "type": "lg",
                                                "children": children}
                else:
                    node_data['Vdata'][node] = {"mean_base": mean_node, "mean_scal": means_parent,
                                                "parents": parents, "variance": variances, "type": "lg",
                                                "children": None}



                # if (len(children) != 0):
                #     node_data['Vdata'][node] = {"mean_base": model.intercept_[0], "mean_scal": list(model.coef_[0]),
                #                                 "parents": parents, "variance": variance, "type": "lg",
                #                                 "children": children}
                # else:
                #     node_data['Vdata'][node] = {"mean_base": model.intercept_[0], "mean_scal": list(model.coef_[0]),
                #                                 "parents": parents, "variance": variance, "type": "lg",
                #                                 "children": None}
            if (len(disc_parents) != 0) & (len(cont_parents) != 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    #predict = []
                    # if new_data.shape[0] != 0:
                    #     #mean_base, std = norm.fit(new_data[node].values)
                    #     #variance = std ** 2
                    #     mean_base = np.mean(new_data[node].values)
                    #     variance = np.var(new_data[node].values)
                    key_comb = [str(x) for x in comb]
                    if new_data.shape[0] != 0:
                        #model = linear_model.BayesianRidge()
                        #model = linear_model.LinearRegression()
                        #model = SVR(kernel='linear', C=100, gamma='auto')
                        #if new_data.shape[0] > N:
                            #gmm = mixture.GaussianMixture(n_components=10, covariance_type='full')
                            
                            # if len(cont_parents) == 1:
                            #     model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                            #     predict = model.predict(np.transpose([new_data[cont_parents[0]].values]))
                            # else:
                            #     model.fit(new_data[cont_parents].values, new_data[node].values)
                            #     predict = model.predict(new_data[cont_parents].values)
                            # mean_base = model.intercept_[0]
                            # variance = (RSE(new_data[node].values, predict)) ** 2
                            #gmm.fit(new_data[cont_parents+[node]].values)
                        means_parent = []#[list(l)[0:-1] for l in list(gmm.means_)]
                        mean_node = []#[list(l)[-1] for l in list(gmm.means_)]
                        labels = gmm.predict(new_data[cont_columns].values)
                        sample = copy(new_data)
                        sample['labels'] = labels
                        variances = []
                        for i in range(0,N,1):
                            sample_small = sample.loc[sample['labels'] == i]
                            variances.append(np.var(sample_small[node].values))
                            mean_node.append(np.mean(sample_small[node].values))
                            parent_one = []
                            for p in cont_parents:
                                parent_one.append(np.mean(sample_small[p].values))
                            means_parent.append(parent_one)
                        # for var in variances:
                        #     if (str(var) == 'nan'):
                        #         variances[variances.index(var)] = 0
                        hycprob[str(key_comb)] = {'variance': variances, 'mean_base': mean_node, 'mean_scal': means_parent}
                        #if new_data.shape[0] <= N:
                            # model = linear_model.LinearRegression()
                            # predict = []
                            # if len(cont_parents) == 1:
                            #     model.fit(np.transpose([new_data[cont_parents[0]].values]), new_data[node].values)
                            #     predict = model.predict(np.transpose([new_data[cont_parents[0]].values]))
                            # else:
                            #     model.fit(new_data[cont_parents].values, new_data[node].values)
                            #     predict = model.predict(new_data[cont_parents].values)
                            # mean_base = model.intercept_
                            # variance = (RSE(new_data[node].values, predict)) ** 2
                            # hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': list(model.coef_)}
                    else:
                        mean_base = np.nan
                        variance = np.nan
                        scal = list(np.full(len(cont_parents), np.nan))
                        #key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': scal}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}
            if (len(disc_parents) != 0) & (len(cont_parents) == 0):
                hycprob = dict()
                values = []
                combinations = []
                for d_p in disc_parents:
                    values.append(np.unique(data[d_p].values))
                for xs in itertools.product(*values):
                    combinations.append(list(xs))
                for comb in combinations:
                    mask = np.full(len(data), True)
                    for col, val in zip(disc_parents, comb):
                        mask = (mask) & (data[col] == val)
                    new_data = data[mask]
                    key_comb = [str(x) for x in comb]
                    if new_data.shape[0] != 0:
                        #mean_base, std = norm.fit(new_data[node].values)
                        #variance = std ** 2
                        # mean_base = np.mean(new_data[node].values)
                        # variance = np.var(new_data[node].values)
                        # if new_data.shape[0] > 6:
                        #     gmm = mixture.GaussianMixture(n_components=5, covariance_type='spherical')
                            
                        #     gmm.fit(np.transpose([new_data[node].values]))
                        #     mean_node = [list(l)[0] for l in list(gmm.means_)]
                        #     labels = gmm.predict(np.transpose([new_data[node].values]))
                        #     sample = copy(new_data)
                        #     sample['labels'] = labels
                        #     variances = []
                        #     for i in range(0,5,1):
                        #         sample_small = sample.loc[sample['labels'] == i]
                        #         variances.append(sample_small[node].var())
                            
                        #     hycprob[str(key_comb)] = {'variance': variances, 'mean_base': mean_node, 'mean_scal': []}
                        # else:
                        mean_base = np.mean(new_data[node].values)
                        variance = np.var(new_data[node].values)
                            #key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                        
                    else:
                        mean_base = np.nan
                        variance = np.nan
                        #key_comb = [str(x) for x in comb]
                        hycprob[str(key_comb)] = {'variance': variance, 'mean_base': mean_base, 'mean_scal': []}
                if (len(children) != 0):
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": children,
                                                "hybcprob": hycprob}
                else:
                    node_data['Vdata'][node] = {"parents": parents, "type": "lgandd", "children": None,
                                                "hybcprob": hycprob}

    return node_data



def RSE(y_true, y_predicted):
   
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))

    rse = math.sqrt(RSS / (len(y_true)))
    return rse