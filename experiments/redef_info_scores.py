import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import pandas as pd
from copy import copy

from bayesian.mi_entropy_gauss import mi_gauss as mutual_information, entropy_all as entropy
from preprocess.discretization import get_nodes_type, code_categories
from preprocess.numpy_pandas import loc_to_DataFrame, get_type_numpy



def info_score(bn, data, metric='BIC'):
	if metric.upper() == 'LL':
		score = log_likelihood(bn, data)
	"""elif metric.upper() == 'BIC':
		score = BIC(bn, data)
	elif metric.upper() == 'AIC':
		score = AIC(bn, data)
	else:
		score = BIC(bn, data)"""

	return score
	

##### INFORMATION-THEORETIC SCORING FUNCTIONS #####

def log_likelihood(bn, data):
	"""
	Determining log-likelihood of the parameters
	of a Bayesian Network. This is a quite simple
	score/calculation, but it is useful as a straight-forward
	structure learning score.

	Semantically, this can be considered as the evaluation
	of the log-likelihood of the data, given the structure
	and parameters of the BN:
		- log( P( D | Theta_G, G ) )
		where Theta_G are the parameters and G is the structure.

	However, for computational reasons it is best to take
	advantage of the decomposability of the log-likelihood score.
	
	As an example, if you add an edge from A->B, then you simply
	need to calculate LOG(P'(B|A)) - Log(P(B)), and if the value
	is positive then the edge improves the fitness score and should
	therefore be included. 

	Even more, you can expand and manipulate terms to calculate the
	difference between the new graph and the original graph as follows:
		Score(G') - Score(G) = M * I(X,Y),
		where M is the number of data points and I(X,Y) is
		the marginal mutual information calculated using
		the empirical distribution over the data.

	In general, the likelihood score decomposes as follows:
		LL(D | Theta_G, G) = 
			M * Sum over Variables ( I ( X , Parents(X) ) ) - 
			M * Sum over Variables ( H( X ) ),
		where 'I' is mutual information and 'H' is the entropy,
		and M is the number of data points

	Moreover, it is clear to see that H(X) is independent of the choice
	of graph structure (G). Thus, we must only determine the difference
	in the mutual information score of the original graph which had a given
	node and its original parents, and the new graph which has a given node
	and new parents.

	NOTE: This assumes the parameters have already
	been learned for the BN's given structure.

	LL = LL - f(N)*|B|, where f(N) = 0

	Arguments
	---------
	*bn* : a BayesNet object
		Must have both structure and parameters
		instantiated.
	Notes
	-----
	NROW = data.shape[0]
	mi_score = 0
	ent_score = 0
	for rv in bn.nodes():
		cols = tuple([bn.V.index(rv)].extend([bn.V.index(p) for p in bn.parents(rv)]))
		mi_score += mutual_information(data[:,cols])
		ent_score += entropy(data[:,bn.V.index(rv)])
	
	return NROW * (mi_score - ent_score)
	"""

	NROW = data.shape[0]
	mi_score = 0
	ent_score = 0
	for rv in bn.nodes():
		l1 = (bn.V.index(rv),)
		l = tuple([bn.V.index(p) for p in bn.parents(rv)])
		
		cols = l1 + l
		mi_score += mutual_information(data[:,cols])
		ent_score += entropy(data[:,bn.V.index(rv)])
	
	return (NROW * (mi_score - ent_score))
	#return ((1/nrow)*(np.sum(np.log((1e+7+bn.flat_cpt())))))

def log_lik_local(data):
    NROW = data.shape[0]
    if isinstance(data, pd.DataFrame):
        return (NROW * (mutual_information(data) - entropy(data.iloc[:,0])))
    elif isinstance(data, pd.Series):
        return 0.0
    elif isinstance(data, np.ndarray):
        return (NROW * (mutual_information(data) - entropy(data[:,0])))
	#return ((1/nrow)*(np.sum(np.log((1e+7+bn.flat_cpt())))))

def BIC_local(data):
	NROW = data.shape[0]
	log_score = log_lik_local(data)
	penalty = 0.5 * num_params(data) * np.log(NROW)
	return log_score - penalty

def num_params(data):
	if isinstance(data, pd.DataFrame):
		return num_params(data.values)
	if isinstance(data, pd.Series):
		print(np.array(data))
		return num_params(np.array(copy(data)))
	if isinstance(data, np.ndarray):
		node_type = get_type_numpy(data)
		columns_for_discrete = []
		columns_for_code = []
		for param in node_type.keys():
			if node_type[param] == 'cont':
				columns_for_discrete.append(param)
			if node_type[param] == 'disc':
				columns_for_code.append(param)
		prod = 1
		for var in columns_for_code:
			prod *= len(np.unique(np.array(data[:,var])))
		if columns_for_discrete != []:
			k = len(columns_for_discrete)
			prod *= int(k*(k+1)/2) + k
		return prod
	else:
		print('Num_params: Unexpected data type')
		print(data)
		pass
	

def BIC(bn, data):
	"""
	Bayesian Information Criterion.

	BIC = LL - f(N)*|B|, where f(N) = log(N)/2

	"""
	log_score = log_likelihood(bn, data)
	penalty = 0.5 * bn.num_params() * np.log(max(bn.num_edges(),1))
	return log_score - penalty



if __name__ == '__main__':
    data = pd.read_csv('./datasets/hackathon_processed.csv')
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    columns = ['Period', 'Tectonic regime', 'Hydrocarbon type']
    #columns = ['Gross', 'Netpay','Porosity']
    #columns = ['Gross', 'Netpay', 'Period']
    #columns = data.columns
    #columns = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Hydrocarbon type', 'Gross','Netpay','Porosity','Permeability', 'Depth']
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

    print(log_lik_local(data_coded['Period']))
    print(log_lik_local(data_coded[['Period', 'Tectonic regime']]))
    print(log_lik_local(data_coded[['Period', 'Hydrocarbon type']]))



















