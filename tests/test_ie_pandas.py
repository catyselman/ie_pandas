#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
from ie_pandas import DataFrame

npDict1 = { '1': np.array([1,2]), '2': np.array([2,3])}
npDict2 = { '1': np.array([1,2,3,4]), '2': np.array([2,3,4,5]), '3': np.array(['ff','dfdfa','dfdfb','cdfd'])}

listDict1 = { '1': [1,2], '2': [2,3]}
listDict2 = { '1': [1,2,3,4], '2': [2,3,4,5], '3': ['ff','dfdfa','dfdfb','cdfd']}

def test_initialize_dataframe_errors():
    
    with pytest.raises(TypeError):
        DataFrame('test')

    with pytest.raises(TypeError):
        DataFrame(['test'])
        
    with pytest.raises(TypeError):
        DataFrame({ '1': np.array([np.array([1]),2,3,4])})
        
    with pytest.raises(TypeError):
        DataFrame({ '1': np.array([None,2,3,4])}) 
        
    with pytest.raises(Exception):
        DataFrame({ '1': np.array([1,2,3,4]), '2': np.array([1,2,3])}) 
        
        
@pytest.mark.parametrize("dictionary, dataSize, colSize", [
    (npDict1, 4, 2),
    (npDict2, 12, 3),
])

def test_initialize_dataframe_numpy(dictionary, dataSize, colSize):
    
    myDF =  DataFrame(dictionary)
    
    assert myDF.data.size == dataSize
    assert len(myDF.types) == colSize
    assert myDF.colNames.size == colSize

@pytest.mark.parametrize("dictionary, dataSize, colSize", [
    (listDict1, 4, 2),
    (listDict2, 12, 3),
])

def test_initiaze_dataframe_list(dictionary, dataSize, colSize):
    myDF =  DataFrame(dictionary)
    
    assert myDF.data.size == dataSize
    assert len(myDF.types) == colSize
    assert myDF.colNames.size == colSize

def test_get_item_errors():
    with pytest.raises(IndexError):
        myDF = DataFrame(npDict1)
        x = myDF['4']


@pytest.mark.parametrize("dictionary, arg, expected", [
    (npDict1, '2', [2, 3]),
    (npDict2, '3', ['ff','dfdfa','dfdfb','cdfd']),
    (listDict1, '2', [2, 3]),
    (listDict2, '3', ['ff','dfdfa','dfdfb','cdfd']),
])

def test_get_item(dictionary, arg, expected):
    myDF =  DataFrame(dictionary)

    response = myDF[arg]

    assert isinstance(response, np.ndarray)
    assert response.size == len(expected)

    for i in range(len(expected)):
        assert response[i] == expected[i]

@pytest.mark.parametrize("dictionary, arg, value", [
    (npDict1, '2', [2, 4]), ## Exiting col
    (npDict2, '4', ['a','b','c','d']), ## Nonexiting col
    (listDict1, '2', [2, 3]), ## Exiting col
    (listDict2, '5', ['a','b','c', 'd']), ## Nonexiting col
])

def test_set_item(dictionary, arg, value):
    myDF = DataFrame(dictionary)
    myDF[arg] = value
    np.testing.assert_array_equal(myDF[arg],np.array(value))

def test_get_row_errors():
    with pytest.raises(IndexError):
        myDF = DataFrame(npDict1)
        x = myDF.get_row(5)

@pytest.mark.parametrize("dictionary, arg, expected", [
    (npDict1, 0, [1,2]),
    (npDict2, 1, [2, 3, 'dfdfa']),
    (listDict1, 1, [2, 3]),
    (listDict2, 2, [3, 4, 'dfdfb']),
])

def test_get_row(dictionary, arg, expected):
    myDF = DataFrame(dictionary)
    assert myDF.get_row(arg) == expected
