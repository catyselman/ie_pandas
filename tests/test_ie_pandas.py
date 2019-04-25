#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import os
#import sys
import pytest
import numpy as np
from ie_pandas.ie_pandas import DataFrame


def test_initialize_dataframe_errors():
    
    with pytest.raises(TypeError):
        DataFrame('test')

    with pytest.raises(TypeError):
        DataFrame(['test'])

    with pytest.raises(TypeError):
        DataFrame({'1': np.array([np.array([1]), 2, 3, 4])})

    with pytest.raises(TypeError):
        DataFrame({'1': np.array([None, 2, 3, 4])})


@pytest.mark.parametrize("dictionary, dataSize, colSize", [
	({'1': np.array([1, 2]), '2': np.array([2, 3])}, 4, 2),
   	({'1': np.array([1, 2, 3, 4]), '2': np.array([2, 3, 4, 5]),
         '3': np.array(['ff', 'dfdfa', 'dfdfb', 'cdfd'])}, 12, 3),
])


def test_initialize_dataframe_numpy(dictionary, dataSize, colSize):

    myDF =  DataFrame(dictionary)

    assert myDF.data.size == dataSize
    assert len(myDF.types) == colSize
    assert myDF.colNames.size == colSize


@pytest.mark.parametrize("dictionary, dataSize, colSize", [
	({'1': [1, 2], '2': [2, 3]},
           4, 2),
   	({'1': [1, 2, 3, 4], '2': [2, 3, 4, 5],
      '3': ['ff', 'dfdfa', 'dfdfb', 'cdfd']}, 12, 3),
])


def test_initiaze_dataframe_list(dictionary, dataSize, colSize):

    myDF = DataFrame(dictionary)

    assert myDF.data.size == dataSize
    assert len(myDF.types) == colSize
    assert myDF.colNames.size == colSize
