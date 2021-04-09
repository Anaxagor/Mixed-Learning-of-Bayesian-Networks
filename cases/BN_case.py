import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


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

random.seed(1)
np.random.seed(1)

def K2(chain: Chain, reference_data: pd.DataFrame) -> float:
    nodes = reference_data.columns.to_list()
    graph, labels = chain_as_nx_graph(chain)
    struct = []
    for pair in graph.edges():
        struct.append([str(labels[pair[0]]), str(labels[pair[1]])])
    BN_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in BN_model.nodes():
            no_nodes.append(node)
    
    score = K2Score(reference_data).score(BN_model) #+ 1000*len(no_nodes)
    return score

def run_credit_scoring_problem(max_lead_time: datetime.timedelta = datetime.timedelta(minutes=5), is_visualise=False, with_tuning=False):
    dataset_to_compose = pd.read_csv('C:\\Users\\anaxa\\Documents\\Projects\\Mixed-Learning-of-Bayesian-Networks\\cases\\data\\geo_encoded.csv')
    #dataset_to_compose = dataset_to_compose.drop(columns=['Tectonic regime', 'Netpay', 'Gross','Lithology', 'Porosity', 'Permeability'])
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
    #builder = builder.with_cache('BN_case')
    composer = builder.build()
    chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                is_visualise=True)
    chain_evo_composed.show()
if __name__ == '__main__':
    run_credit_scoring_problem(is_visualise=True)
                               


