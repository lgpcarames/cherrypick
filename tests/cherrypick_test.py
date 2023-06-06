#!/usr/bin/env python

"""Tests for `cherrypick` package."""


import unittest
import pandas as pd
import sys
sys.path.insert(0, r'C:/Users/Lucas/Documents/feature_selector/cherrypick')

import cherrypick as cp

import pytest
from cherrypick import (
    threshold_score,
    _get_features_threshold_score_,
    _best_threshold_classification_,
    _set_difficulty_group_,
    _generate_stats_sucess_,
    generate_cherry_score,
)


# Test threshold_score function
def test_threshold_score():
    predictions = [0.2, 0.4, 0.6, 0.8]
    target = [0, 0, 1, 1]
    result = threshold_score(predictions, target)
    assert result['precision'] == pytest.approx(0.6667, abs=1e-4)
    assert result['recall'] == pytest.approx(0.5, abs=1e-4)
    assert result['acuracia'] == pytest.approx(0.5, abs=1e-4)
    assert result['f-score'] == pytest.approx(0.5714, abs=1e-4)
    assert result['roc-auc'] == pytest.approx(0.75, abs=1e-4)
    assert result['threshold'] == pytest.approx(0.6, abs=1e-4)


# Test _get_features_threshold_score_ function
def test_get_features_threshold_score():
    df = ...  # Create a test DataFrame
    variables = ...  # Define the list of variables
    target = ...  # Define the target variable
    result = _get_features_threshold_score_(df, variables, target)
    assert ...  # Make assertions on the result


# Test _best_threshold_classification_ function
def test_best_threshold_classification():
    df = ...  # Create a test DataFrame
    variables = ...  # Define the list of variables
    target = ...  # Define the target variable
    result = _best_threshold_classification_(df, variables, target)
    assert ...  # Make assertions on the result


# Test _set_difficulty_group_ function
def test_set_difficulty_group():
    df = ...  # Create a test DataFrame
    target = ...  # Define the target variable
    result = _set_difficulty_group_(df, target)
    assert ...  # Make assertions on the result


# Test _generate_stats_sucess_ function
def test_generate_stats_sucess():
    df = ...  # Create a test DataFrame
    variables = ...  # Define the list of variables
    target = ...  # Define the target variable
    result = _generate_stats_sucess_(df, variables, target)
    assert ...  # Make assertions on the result


# Test generate_cherry_score function
def test_generate_cherry_score():
    df = ...  # Create a test DataFrame
    variables = ...  # Define the list of variables
    target = ...  # Define the target variable
    result = generate_cherry_score(df, variables, target)
    assert ...  # Make assertions on the result


# Run the tests
if _name_ == '_main_':
    pytest.main()