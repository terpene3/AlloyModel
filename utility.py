import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

"""
utility.py
Implements helper functions
"""


def pLoad(name):
    """
    Loads a file saved using pickle named name.csv
    :param name: filename string
    :return: loaded data
    """
    with open(name + ".csv", 'rb') as handle:
        f = pickle.load(handle)
        return f
    
def pSave(data, name):
    """
    Saves data using pickle to name.csv
    :param data: data to be saved
    :param name: file name string
    :return: none
    """
    with open(name +".csv", "wb") as handle:
        pickle.dump(data, handle)


def convertNPtoString(npin):
    """
    :param npin: list of floats or a 1D np array
    :return: list of strings with exponent and 3 decimal places
    """
    out = []
    for elem in npin:
        out +=['{0:1.3e}'.format(elem)]
    return out

def splitStabilityVector(stabilityVector):
    """

    :param stabilityVector: Converts the vector to an array
    Forms a regex search for 1 digit numbers with 1 decimal
    Requires that all vectors are the same length
    Checks that all values are [0,1]
    :return: An np array of the split vector
    """
    split = np.asarray([re.findall(r"(\d\.\d)", elem) for elem in stabilityVector], dtype = "float16")
    if np.amax(split) > 1:
        raise ValueError("Value in array is > 1")

    return split

def augment(Xin, yin):
    """
    Automatically matches and reverses collumns
    Drops stability vector
    :param input: input array from pandas with header
    Require that Xin has an even number of collumns, and alternating A,B columns
    :return: returns an augmented array based on symmetry between A and B for Xin and yin
    """


    Xindropped = Xin.drop(["stabilityVec"], axis = 1)


    columns = Xindropped.columns.values.tolist()

    if len(columns) % 2:
        raise  ValueError("number of columns not divisible by 2")

    if len(Xin) != len(yin):
        raise ValueError("number of columns not divisible by 2")

    XinSwapped = Xindropped.copy()
    yout = np.zeros([yin.shape[0] * 2, yin.shape[1]])
    yout[0 : yin.shape[0]] = yin
    yout[yin.shape[0] : 2* yin.shape[0]] = yin[:,::-1]

    for idx in range(0, len(columns), 2):
        #check to make sure columns match

        def extractAB(string):
            """
            Extracts x in formula_x_restofstring or restofstring_x
            where x is 'A' or 'B'
            :param string: input string
            :return: False if no match found, otherwise (the extracted A or B, restofstring)
            In the event both match, formula_x_restofstring has priority
            """
            #try formula_x
            groups = re.search(r"formula([AB])($|.+)", string)
            if groups:
                return groups.group(1), groups.group(2)
            #try _x
            groups = re.search(r"(.+)_([AB])$", string)
            if groups:
                return groups.group(2), groups.group(1)

            return False

        matches1 = extractAB(columns[idx])
        matches2 = extractAB(columns[idx + 1])

        #check that matches succeeded, have both an A and B, and the rest of the strings match
        if (matches1 and matches2) and (matches1[0] != matches2[0]) and matches1[1] == matches2[1]:
                columns[idx], columns[idx + 1] = columns[idx + 1], columns[idx]
        else:
            raise ValueError("invalid headers found in " + columns[idx] + " " + columns[idx + 1])
        XinSwapped.columns = columns

        Xout = pd.concat([Xindropped,XinSwapped])
    return Xout, yout


def crossValidate(model, xtrain, ytrain, param_name, param_range, subplotidx=111):
    """
    Performs and plots 5-fold cross-validation of param_name over param_range
    :param model: scikit learn model
    :param xtrain: train features pd or np array
    :param ytrain: train truth values np array
    :param param_name: string of parameter to cross-validate
    :param param_range: list of value to try to parameter
    :param subplotidx: index of subplot to plot as (optional)
    :return: (train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)
    """


    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html

    # perform 5 fold CV
    train_scores, test_scores = validation_curve(
        model, xtrain, ytrain, param_name=param_name, param_range=param_range,
        cv=5, scoring="mean_squared_error", n_jobs=4)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plot the CV results
    plt.subplot(subplotidx)
    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("neg. MSE")

    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    if subplotidx == 111:
        plt.show()
    return (train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

def buildStabilityVectors(vectors):
    """
    Formats the stability vectors for output based on vectors inputs
    :param vectors: a np array of stability vectors without the first and last index
    :return: list of strings
    """
    outstrings = []
    for vector in vectors:
        outstring = "[1.0,"
        for elem in vector:
            outstring += '{0:1.1f}'.format(elem) + ","

        outstring += "1.0]"
        outstrings.append(outstring)
    return outstrings
