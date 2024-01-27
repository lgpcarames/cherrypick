#!/usr/bin/env python

"""Tests for `cherrypick` package."""

import sys
import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer
sys.path.insert(0, r'/cherrypick')
from cherrypick import (threshold_score,
    __get_features_threshold_score__,
    __best_threshold_classification__,
    __set_difficulty_group__,
    __generate_stats_success__,
      cherry_score
    )

data = load_breast_cancer()



# Test threshold_score function
def test_threshold_score():
    predictions = [0.77770241, 0.23754122, 0.82427853, 0.9657492 , 0.97260111,
       0.45344925, 0.60904246, 0.77552651, 0.64161334, 0.72201823,
       0.03503652, 0.29844947, 0.05851249, 0.85706094, 0.37285403,
       0.67984795, 0.25627995, 0.34758122, 0.00941277, 0.35833378,
       0.94909418, 0.21789901, 0.31939137, 0.91777239, 0.03190367,
       0.06508454, 0.629829  , 0.87381344, 0.00871573, 0.74657724]
    target = [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

    result = threshold_score(predictions, target)

    assert result['precision'] == pytest.approx(0.8, abs=1e-4)
    assert result['recall'] == pytest.approx(0.5, abs=1e-4)
    assert result['accuracy'] == pytest.approx(0.5, abs=1e-4)
    assert result['f-score'] == pytest.approx(0.61538461538461541, abs=1e-4)
    assert result['roc-auc'] == pytest.approx(0.5902777777777778, abs=1e-4)
    assert result['threshold'] == pytest.approx(0.6090424627612779, abs=1e-4)


# Test _get_features_threshold_score_ function
def test_get_features_threshold_score():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data,
                      columns=data.feature_names) # Create a test DataFrame
    df['target'] = data.target
    variables = data.feature_names  # Define the list of variables
    target =   'target' # Define the target variable
    result = __get_features_threshold_score__(df, variables, target)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == len(variables)
    assert all(col in result.columns for col in ['variable', 'threshold_variable'])




# Test _best_threshold_classification_ function
def test_best_threshold_classification():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data,
                      columns=data.feature_names) # Create a test DataFrame
    df['target'] = data.target
    variables = data.feature_names  # Define the list of variables
    target =   'target' # Define the target variable
    result = __best_threshold_classification__(df, variables, target)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape


# Test _set_difficulty_group_ function
def test_set_difficulty_group():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data,
                      columns=data.feature_names) # Create a test DataFrame
    df['target'] = data.target
    variables = data.feature_names  # Define the list of variables
    target =   'target' # Define the target variable

    classified_df = __best_threshold_classification__(df=df, variables=variables, target=target)

    result = __set_difficulty_group__(classified_df, target)
    assert isinstance(result, pd.DataFrame)
    assert 'success_rate' in result.columns
    assert 'difficulty_group' in result.columns
    assert result['difficulty_group'].nunique() == 3


# Test _generate_stats_success_ function
def test_generate_stats_success():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data,
                      columns=data.feature_names) # Create a test DataFrame
    df['target'] = data.target
    variables = data.feature_names  # Define the list of variables
    target =   'target' # Define the target variable

    classified_df = __best_threshold_classification__(df=df, variables=variables, target=target)

    difficulty_df = __set_difficulty_group__(df=classified_df, target=target)

    result = __generate_stats_success__(df=difficulty_df, variables=variables, target=target)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == len(variables)
    assert all(col in result.columns for col in ['variable', 'success_rate_g0', 'success_rate_g1'])



# Test cherry_score function
def test_cherry_score():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data,
                      columns=data.feature_names) # Create a test DataFrame
    df['target'] = data.target
    variables = data.feature_names  # Define the list of variables
    target =   'target' # Define the target variable
    result = cherry_score(df, variables, target)
    assert isinstance(result, pd.DataFrame),\
        'the outcome is not a dataframe type'
    assert result.shape[0] == len(variables),\
    'the num of rows in the outcome is not equal to the num of variables'
    assert all(col in result.columns for col in ['variable', 'cherry_score']),\
        'the cols variables and/or cherry_score are not present in the cols of outcome'
    assert isinstance(cherry_score(df[[variables[0]]+[target]], [variables[0]], target), pd.DataFrame),\
        'the outcome type when you have only an unique input col is not a dataframe'



# Run the tests
if __name__ == '_main_':
    pytest.main()
