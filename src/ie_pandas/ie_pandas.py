#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class DataFrame:


    def __init__(self, valuesDictionary):

        # Creation from a dictionary of lists and dictionary of NumPy arrays
        if not isinstance(valuesDictionary, dict):
            raise TypeError('DataFrame must be initalized with a dictionary' +
                            ' of lists of Numpy arrays.')

        #  Support for int, float, and bool, string (non-numerical) columns
        supportedTypes = (int, float, str, bool, np.int_, np.float_,
                          np.str_, np.bool_)

        for value in list(valuesDictionary.values()):
            if (not isinstance(value, np.ndarray)) and (not isinstance(value, list)):
                raise TypeError('All values of dictionary must be a list or' +
                                ' numpy arrays.')
            if not all(isinstance(val, supportedTypes) for val in value):
                raise TypeError('All values in the data dictionary must be' + 
                                ' integers, floats, string or booleans.')

        # Saving column names
        self.colNames = np.array(list(valuesDictionary.keys()))

        valList = list(valuesDictionary.values())

        # Check types and save as a variable
        self.types = []
        for column in valList:
            column = np.array(column)
            self.types.append(column.dtype.type)

        # Save data as a numpy array
        self.data = np.transpose(np.array(valList, dtype='O'))

    def __repr__(self):
        return "Data:\n{0}\n Colnames: {1}.".format(self.data, self.colNames)