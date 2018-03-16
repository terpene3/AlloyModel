import pandas as pd
import numpy as np

"""
featureGen.py
implements methods to create new features
generates new features based on average features, the absolute different,
and the relative values
"""

def featuregenerate(featuresIn, feature1, feature2, newFeatureName, featuretype, convertNameToData=True, debug=False):
    """
    Generates new augmentation features based on absolute differences, averages, and fractions of feature 1 and feature2
    :param featuresIn: Pandas dataframe
    :param feature1: name of first feature to make new feature using
    :param feature2: name of second feature to make new feature using
    :param newFeatureName: name of new feature
    :param featuretype: type of augmentation to be performed
    :param convertNameToData:
    :param debug: Bool of whether to print debug output
    :return:
    """
    if convertNameToData:
        feature1 = featuresIn[feature1]
        feature2 = featuresIn[feature2]

    if featuretype == "absdiff":
        newfeature = np.abs(feature1 - feature2)

    elif featuretype == "avg":
        newfeature = (feature1 + feature2) / 2

    elif featuretype == "frac":
        newfeature = feature1 / feature2

    else:
        raise ValueError("Feature type ", featuretype, "is invalid")

    newfeature = pd.DataFrame(newfeature, columns=[newFeatureName])
    if debug:
        display(pd.concat([feature1, feature2, newfeature], axis=1))
    newframe = pd.concat([featuresIn, newfeature], axis=1)
    return newframe


def valenceFeatureGeneration(featuresIn):
    """
    Generates new features based on valance electron counts
    :param featuresIn: Pandas dataframe
    :return: Pandas dataframe with new features appended
    """

    Sa = 'formulaA_elements_NsValence'
    Sb = 'formulaB_elements_NsValence'
    Pa = 'formulaA_elements_NpValence'
    Pb = 'formulaB_elements_NpValence'
    Da = 'formulaA_elements_NdValence'
    Db = 'formulaB_elements_NdValence'
    Fa = 'formulaA_elements_NfValence'
    Fb = 'formulaB_elements_NfValence'
    Na = 'formulaA_elements_NValance'
    Nb = 'formulaB_elements_NValance'

    # make new features based on average numbers of valence electrons for S, P, D, F, and total
    featuresIn = featuregenerate(featuresIn, Sa, Sb, 'avg' + '_NsValence', 'avg')
    featuresIn = featuregenerate(featuresIn, Pa, Pb, 'avg' + '_NpValence', 'avg')
    featuresIn = featuregenerate(featuresIn, Da, Db, 'avg' + '_NdValence', 'avg')
    featuresIn = featuregenerate(featuresIn, Fa, Fb, 'avg' + '_NfValence', 'avg')
    featuresIn = featuregenerate(featuresIn, Na, Nb, 'avg' + '_NValence', 'avg')

    # make new features based on average fraction of S, P, D, F electrons
    featuresIn = featuregenerate(featuresIn, 'avg_NsValence', 'avg_NValence', 'frac_NsValence', 'frac')
    featuresIn = featuregenerate(featuresIn, 'avg_NpValence', 'avg_NValence', 'frac_NpValence', 'frac')
    featuresIn = featuregenerate(featuresIn, 'avg_NdValence', 'avg_NValence', 'frac_NdValence', 'frac')
    featuresIn = featuregenerate(featuresIn, 'avg_NfValence', 'avg_NValence', 'frac_NfValence', 'frac')

    return featuresIn


def newFeatures(featuresIn):
    """
    Generates new augmentation features based on absolute differences, averages, and fractions of valence electrons
    :param featuresIn: Pandas dataframe
    :return: Pandas dataframe with new features appended
    """

    # new feature as average atomic weight
    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_AtomicWeight', 'formulaB_elements_AtomicWeight',
                                 'avg_AtomicWeight', 'avg', True)
    # new feature as average row
    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_Row', 'formulaB_elements_Row',
                                 'avg_Row', 'avg', True)
    # new feature as average column
    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_Column', 'formulaB_elements_Column',
                                 'avgs_Column', 'avg', True)
    # new feature as average atomic number
    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_Number', 'formulaB_elements_Number',
                                 'avg_Number', 'avg', True)
    # new feature as absolute difference in atomic number
    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_Number', 'formulaB_elements_Number',
                                 'absd_Number', 'absdiff', True)

    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_CovalentRadius', 'formulaB_elements_CovalentRadius',
                                 'avgs_CovalentRadius', 'avg', True)

    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_CovalentRadius', 'formulaB_elements_CovalentRadius',
                                 'absd_CovalentRadius', 'absdiff', True)

    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_Electronegativity',
                                 'formulaB_elements_Electronegativity',
                                 'avg_Electronegativity', 'avg', True)

    featuresIn = featuregenerate(featuresIn, 'formulaA_elements_Electronegativity',
                                 'formulaB_elements_Electronegativity',
                                 'absd_Electronegativity', 'absdiff', True)

    featuresIn = valenceFeatureGeneration(featuresIn)

    return featuresIn