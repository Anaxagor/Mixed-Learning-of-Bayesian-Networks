import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from bayesian.mi_gaussian import mi_gauss
from pgmpy.estimators.StructureScore import StructureScore

class MIG(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for Bayesian structure scoring for BayesianModels with MI for Gaussian.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        """
        super(MIG, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'"""

        list_var = [variable]
        list_var.extend(parents)
        
        score = mi_gauss(self.data[list_var])

        return score


    
   
    