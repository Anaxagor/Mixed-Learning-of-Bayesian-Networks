import datetime
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from networkx.algorithms.cycles import simple_cycles
from pgmpy.estimators import K2Score
from pgmpy.models import BayesianModel

from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.chains.chain_validation import has_no_self_cycled_nodes
from fedot.core.chains.graph import GraphObject
from fedot.core.chains.graph_node import PrimaryGraphNode, SecondaryGraphNode
from fedot.core.composer.gp_composer.gp_composer import ChainGenerationParams, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import (
    GPChainOptimiser,
    GPChainOptimiserParameters,
    GeneticSchemeTypesEnum)
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.log import default_log

random.seed(1)
np.random.seed(1)


def project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent


def k2_metric(network: GraphObject, data: pd.DataFrame):
    nodes = data.columns.to_list()
    graph, labels = chain_as_nx_graph(network)
    struct = []
    for pair in graph.edges():
        struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
    bn_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in bn_model.nodes():
            no_nodes.append(node)

    return [random.random()]
    score = K2Score(data).score(bn_model)
    return [score]


def _has_no_duplicates(graph):
    _, labels = chain_as_nx_graph(graph)
    list_of_nodes = []
    for node in labels.values():
        list_of_nodes.append(str(node))
    if len(list_of_nodes) != len(set(list_of_nodes)):
        raise ValueError('Chain has duplicates')
    return True


def _has_disc_parents(graph):
    node_types = {'Tectonic regime': 'disc',
                  'Period': 'disc',
                  'Lithology': 'disc',
                  'Structural setting': 'disc',
                  'Hydrocarbon type': 'disc',
                  'Gross': 'cont',
                  'Netpay': 'cont',
                  'Porosity': 'cont',
                  'Permeability': 'cont',
                  'Depth': 'cont'}
    graph, labels = chain_as_nx_graph(graph)
    for pair in graph.edges():
        if (node_types[str(labels[pair[1]])] == 'disc') & (node_types[str(labels[pair[0]])] == 'cont'):
            raise ValueError(f'Discrete node has cont parent')
    return True


def _has_no_cycle(graph: GraphObject):
    nx_graph, _ = chain_as_nx_graph(graph)
    cycled = list(simple_cycles(nx_graph))
    if len(cycled) > 0:
        raise ValueError('Chain has cycle')
    return True


def run_bayesian(max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5)):
    data = pd.read_csv(f'{project_root()}\\data\\geo_encoded.csv')
    nodes_types = ['Tectonic regime', 'Period', 'Lithology',
                   'Structural setting', 'Gross', 'Netpay',
                   'Porosity', 'Permeability', 'Depth']
    rules = [has_no_self_cycled_nodes, _has_no_cycle, _has_no_duplicates, _has_disc_parents]

    requirements = GPComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types, max_arity=3,
        max_depth=3, pop_size=20, num_of_generations=50,
        crossover_prob=0.8, mutation_prob=0.9, max_lead_time=max_lead_time)

    optimiser_parameters = GPChainOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth])

    chain_generation_params = ChainGenerationParams(
        chain_class=GraphObject,
        primary_node_func=PrimaryGraphNode,
        secondary_node_func=SecondaryGraphNode,
        rules_for_constraint=rules)

    optimizer = GPChainOptimiser(
        chain_generation_params=chain_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_chain=None,
        log=default_log(logger_name='Bayesian', verbose_level=4))

    optimized_network = optimizer.optimise(partial(k2_metric, data=data))

    optimized_network.show()


if __name__ == '__main__':
    run_bayesian()
