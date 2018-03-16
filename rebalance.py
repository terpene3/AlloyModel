import numpy as np
import sklearn.linear_model as linear_model

"""
rebalance.py
Implements methods for splitting training data to improve class balance
Implements methods for models that take this balanced data as inputs
"""

def splitClasses(yin):
    """
    On a dimension basis in yin make lists of observations that are 0 and 1
    :param yin: 2D numpy array
    :return: list of lists where index of first list corresponds to a collumn in y
    zeros contains index of elements that have a zero y entry at the corresponding column
    nonzeros contains the elements that do not
    """
    nonzero = [[] for x in range(yin.shape[1])]
    zero = [[] for x in range(yin.shape[1])]
    for idx, instance in enumerate(yin):
        for vidx, value in enumerate(instance):
            if value == 0:
                zero[vidx] += [idx]
            elif value == 1:
                nonzero[vidx] += [idx]
            else:
                raise ValueError("Value not 0 or 1 in row: " + str(idx) + "value: ", value)
    return zero, nonzero

def splitTrivial(yin):
    null = []
    nonnull = []
    for idx, y in enumerate(yin):
        if np.sum(y) > 2:
            nonnull.append(idx)
        else:
            null.append(idx)
    return null, nonnull


"""
Class for a logistic regression with a output of dimension size
Implemented using size separate 1D logistic regressions
"""
class ndlogistic():
    def __init__(self, size, cvector = None, penalty ='l2'):
        self.size = size
        if cvector:
            self.model = [linear_model.LogisticRegression(penalty= penalty, C=cvector[i])
                          for i in range(size)]
            if size != len(cvector):
                print("warning size does not match length of cvector")
        else:
            self.model = [linear_model.LogisticRegression()
                         for i in range(size)]
    def train(self, xin, trainindices, yin):
        """
        trains the model
        :param xin: xtrain data
        :param trainindices: indices of the train data to use
        :param yin: ytrain data. ytrain is a list of np arrays
        each array contain the truth data for a dimension of the train data
        :return: none
        """
        if len(yin) != self.size:
            raise ValueError("yin shape does not match model shape")
        for i in range(self.size):
            self.model[i].fit(xin.iloc[trainindices[i]], yin[i])

    def predict(self, xin, indices = None):
        """
        predicts for xin
        :param xin: pd dataframe
        :param indices: indicies of xin to use (optional)
        a list of lists with the top dimension being the same as self.size
        :return: a list of prediction for each dimension if indicies are used
        otherwise returns a np array of the predictions
        """
        if indices:
            output = []
        else:
            output = np.zeros([xin.shape[0], self.size])

        for i in range(self.size):
            if indices:
                output.append(self.model[i].predict(xin.iloc[indices[i]]))
            else:
                output[: , i] = self.model[i].predict(xin)

        return output