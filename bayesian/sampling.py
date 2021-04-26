import pandas as pd

from external.libpgm.hybayesiannetwork import HyBayesianNetwork


def generate_synthetics(bn: HyBayesianNetwork, n: int = 1000, evidence: dict = None) -> pd.DataFrame:
    """Function for sampling from BN

    Args:
        bn (HyBayesianNetwork): learnt BN
        n (int, optional): number of samples (rows). Defaults to 1000.
        evidence (dict): dictionary with values of params that initialize nodes

    Returns:
        pd.DataFrame: final sample
    """
    sample = pd.DataFrame()

    if evidence:
        sample = pd.DataFrame(bn.randomsample(5 * n, evidence=evidence))
        cont_nodes = []
        for key in bn.nodes.keys():
            if (str(type(bn.nodes[key])).split('.')[1] == 'lg') | (str(type(bn.nodes[key])).split('.')[1] == 'lgandd'):
                cont_nodes.append(key)
        sample.dropna(inplace=True)
        sample = sample.loc[(sample.loc[:, cont_nodes].values >= 0).all(axis=1)]
        sample.reset_index(inplace=True, drop=True)
    else:
        sample = pd.DataFrame(bn.randomsample(5 * n))
        cont_nodes = []
        for key in bn.nodes.keys():
            if (str(type(bn.nodes[key])).split('.')[1] == 'lg') | (str(type(bn.nodes[key])).split('.')[1] == 'lgandd'):
                cont_nodes.append(key)
        sample.dropna(inplace=True)
        sample = sample.loc[(sample.loc[:, cont_nodes].values >= 0).all(axis=1)]
        sample.reset_index(inplace=True, drop=True)


    

    # final_sample = pd.DataFrame()

    # i = 0
    # while i < n:
    #     sample = pd.DataFrame(bn.randomsample(1))
    #     flag = True
    #     for node in cont_nodes:
    #         if (sample.loc[0,node] < 0) | (str(sample.loc[0,node]) == 'nan'):
    #             flag = False
    #     if flag:
    #         final_sample = pd.concat([final_sample, sample])
    #         i = i + 1
    #     else:
    #         continue
    return sample
