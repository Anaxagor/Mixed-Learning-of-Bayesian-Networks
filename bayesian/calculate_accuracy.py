import operator
from copy import copy

import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from core.bayesian.discretization import get_nodes_type
from core.bayesian.restoration import predict_params
from core.api.networks.utils import to_networkx
from external.libpgm.hybayesiannetwork import HyBayesianNetwork


def calculate_acc(bn: HyBayesianNetwork, data: pd.DataFrame, columns: list) -> (dict, dict, float):
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
                result = predict_params(train_dict, bn, node_type)
                if node_type[key] == 'disc':
                    pred_param[n][i] = max(result[key].items(), key=operator.itemgetter(1))[0]
                    real_param[n][i] = test[key]
                if node_type[key] == 'cont':
                    if str(result[key][0]) == 'nan':
                        continue
                    else:
                        pred_param[n][i] = result[key][0]
                        real_param[n][i] = test[key]
            except Exception as ex:
                continue
    for n, key in enumerate(columns):
        if node_type[key] == 'disc':
            accuracy_dict[key] = round(accuracy_score(real_param[n], pred_param[n]), 2)
        if node_type[key] == 'cont':
            rmse_dict[key] = round(mean_squared_error(real_param[n], pred_param[n], squared=False), 2)
    structure = dict()
    structure['V'] = bn.V
    structure['E'] = bn.E
    G = to_networkx(structure)
    density = round(nx.density(G), 2)
    return accuracy_dict, rmse_dict, density

# def pome_to_pgmpy(bnfit: BayesianNetwork):
#     """
#     Convert the network from Pomegranate format to pgmpy.
#     """

#     def temp_func2(temp1):
#         temp2=[t[-1] for t in temp1]
#         temp3=[x[-2] for x in temp1]
#         numvalues=len(set(temp3))
#         values=sorted(list(set(temp3)))
#         totallen=len(temp3)
#         temp4=[temp2[i:totallen:numvalues] for i in range(0,numvalues)] 
#         return numvalues,values,temp4

#     edge_list = []
#     for i in range(bnfit.node_count()):
#         for element in list(bnfit.structure[i]):
#             edge_list.append((element, i))
#     bn_new2 = BayesianModel(edge_list)
#     t1=list(bn_new2.nodes)
#     t2=[int(bnfit.states[i].name) for i in range(len(bnfit.states))]
#     t3=list(set(t2)-set(t1))
#     if t3:
#         bn_new2.add_nodes_from(t3)

#     len_of_nodes=bnfit.node_count()
#     listofcpds=[]
#     for x in range(0, len_of_nodes):
#         if bnfit.structure[x]:
#             t1=temp_func2(bnfit.states[x].distribution.parameters[0])
#             evidence=list(bnfit.structure[x])
#             len_evidence=len(bnfit.structure[x])
#             listofcpds.append([x,t1[0],t1[1],evidence,t1[2]])
#         else:
#             t1=bnfit.states[x].distribution.parameters[0]
#             keys=sorted(t1)
#             val=[[t1[i]] for i in sorted(t1)]       
#             listofcpds.append([x,len(val),keys,[],val]) 


#     for x in range(0, len_of_nodes):
#         if bnfit.structure[x]:
#             cpd_d_sn = TabularCPD(variable=listofcpds[x][0], variable_card=listofcpds[x][1], values=listofcpds[x][4],  evidence=listofcpds[x][3], evidence_card=[listofcpds[p1][1] for p1 in listofcpds[x][3]], state_names={p1:listofcpds[p1][2] for p1 in [listofcpds[x][0]]+listofcpds[x][3]})
#             bn_new2.add_cpds(cpd_d_sn)
#         else:
#             cpd_d_sn = TabularCPD(variable=listofcpds[x][0], variable_card=listofcpds[x][1], values=listofcpds[x][4], state_names={listofcpds[x][0]: listofcpds[x][2]})
#             bn_new2.add_cpds(cpd_d_sn)
#     return(bn_new2)


# def calculate_acc(bn: BayesianNetwork, data_test: pd.DataFrame, verbose=False):
#     """
#     Сalculates the accuracy of the received network on a test dataset. 
#     If a directed acyclic graph has at least one node with 3 parents, 
#     then the inference occurs through the pgmpy library. 
#     The network should be converted to pgmpy format for this. 
#     The inference occurs through the Pomegranate library if there are no such nodes. 

#     Input:
#     -bn
#     Input fitted pomegranate Bayesian network

#     -data_test
#     Dataset for checking accuracy

#     -verbose
#     Progress output


#     Output:
#     1) The first object is a DataFrame with acuracy: 
#         param	acc
#     0	Depth	0.765957
#     1	Gross	0.531915
#     2	Netpay	0.468085
#     3	Porosity	0.553191
#     4	Permeability	0.765957
#     5	Structural setting	1.000000
#     6	Period	1.000000
#     7	Lithology	0.936170
#     8	Tectonic regime	0.978723
#     9	Hydrocarbon type	1.000000

#     2) The second object is a DataFrame with restored value: 

#         Depth	Gross	Netpay	Porosity	Permeability	Structural setting	Period	Lithology	Tectonic regime	Hydrocarbon type
#     0	2	6	1	3	2	31	1	2	14	5
#     1	3	1	3	3	2	31	21	2	24	5
#     2	3	1	3	3	1	31	21	2	24	5
#     3	0	1	4	0	2	31	24	10	14	2
#     ...

#     3) The third object is a density of a graph.
#     d=m/(n(n−1)), where n is the number of nodes and m is the number of edges in graph.

#     4) Elapsed time in sec.

#     """    

#     pgmpy_is_true=any(i >= 3 for i in list(map(len,bn.structure)))
#     if pgmpy_is_true:
#         t = time.process_time()
#         l1 = list(bn.structure)
#         edges1 = [(y, x[0]) for x in enumerate(l1) for y in x[1]]
#         G = nx.DiGraph()
#         G.add_edges_from(edges1)
#         out_density = nx.density(G)


#         bn_new=pome_to_pgmpy(bn)
#         bn_new_infer = VariableElimination(bn_new)

#         ifdiff=len(data_test.columns.to_list())-len(bn_new.nodes())
#         if ifdiff:
#             print("The number of parameters in the fitted model does not match the specified data")

#         test_datai = data_test.copy()
#         test_datai.columns = [i for i in range(len(test_datai.columns))]
#         res=pd.DataFrame(columns=['param','acc'])
#         t3=pd.DataFrame()

#         for column_iterator in range(len(test_datai.columns)):
#             time1 = time.process_time()
#             test_data = test_datai.copy()
#             test_data=test_data.drop(columns=test_datai.columns[column_iterator])
#             test_dict=test_data.to_dict('records')
#             t2 = pd.DataFrame(columns=[test_datai.columns[column_iterator]])

#             for rows_iterator in range(len(test_datai)):
#                 bnq = bn_new_infer.map_query(variables=[test_datai.columns[column_iterator]], evidence=test_dict[rows_iterator],show_progress=False)
#                 t2=t2.append(bnq,ignore_index=True)

#             t2=t2.astype(int)
#             t3=pd.concat([t3,t2],axis=1)
#             res=res.append({'param': test_datai.columns[column_iterator], 'acc': accuracy_score(list(t2[test_datai.columns[column_iterator]]),list(test_datai[test_datai.columns[column_iterator]]))},ignore_index=True)

#             if verbose:
#                 print("Calculation is over for:",res['param'][column_iterator],"accuracy:",res['acc'][column_iterator], "time:", time.process_time() - time1)

#         elapsed_time = time.process_time() - t
#         res['param']=list(data_test.columns)
#         t3.columns=list(data_test.columns)

#     else:
#         t = time.process_time()
#         l1 = list(bn.structure)
#         edges1 = [(y, x[0]) for x in enumerate(l1) for y in x[1]]
#         G = nx.DiGraph()
#         G.add_edges_from(edges1)
#         out_density = nx.density(G)

#         res = pd.DataFrame(columns=['param', 'acc'])
#         t3 = pd.DataFrame()

#         if verbose:
#             elapsed_time1 = time.process_time() - t
#             print("Start calc:", elapsed_time1)

#         for column_iterator in range(len(data_test.columns)):
#             time1=time.process_time()
#             test_data = data_test.copy()
#             test_data[test_data.columns[column_iterator]] = None
#             t2 = pd.DataFrame(columns=[test_data.columns[column_iterator]])
#             t1 = bn.predict(test_data.values)
#             t2 = [x[column_iterator] for x in t1]
#             t4 = pd.DataFrame(t2, columns=[data_test.columns[column_iterator]])
#             t4 = t4.astype(int)
#             t3 = pd.concat([t3, t4], axis=1)
#             res = res.append({'param': data_test.columns[column_iterator],
#                           'acc': accuracy_score(t2, list(data_test[data_test.columns[column_iterator]]))},
#                          ignore_index=True)

#             if verbose:
#                 print("Calculation is over for:", res['param'][column_iterator], "accuracy:", res['acc'][column_iterator],"time:", time.process_time() - time1)

#         elapsed_time = time.process_time() - t

#     return res, t3, out_density, elapsed_time
