#!/usr/bin/env python

"""Tests for `cherrypick` package."""


import unittest
import pandas as pd
import sys
sys.path.insert(0, r'C:\\Users\\Lucas\\Documents\\feature_selector\\cherrypick')

from cherrypick import cherrypick

class TestCherrypick(unittest.TestCase):
    """Tests for `cherrypick` package."""

    # def setUp(self):
    #     """Set up test fixtures, if any."""

    # def tearDown(self):
    #     """Tear down test fixtures, if any."""

    # def test_000_something(self):
    #     """Test something."""

    
    def testa_fit_gera_dic_e_rev_dic(self):

        ce = cherrypick.CatEncoder()

        ce.fit(pd.DataFrame({'variavel': ['variavel_0', 'variavel_1', 'variavel_2']}))

        self.assertIsNotNone(ce.dic)

if __name__=='__main__':
    unittest.main()