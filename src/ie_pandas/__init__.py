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
        self.colNames = np.array(list(map(str, list(valuesDictionary.keys()))))

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


    #Dictionary-style access to columns (`df["col_name"]`), which should return NumPy arrays in
    #all cases, and should allow modification (read and write)

    ## Read
        ## Check return val
        ## Cover cases with multiple arguments (if needed?)
    def __getitem__(self, arg):
        try:
            colIndex = np.where(self.colNames==arg)[0][0]
        except:
            raise IndexError("Column '{0}' does not exist in DataFrame.".format(arg))
        return self.data[:, colIndex]

    ## Write
    def __setitem__(self, arg, value):
        if(arg in self.colNames):
            colIndex = np.where(self.colNames==arg)[0][0]
            self.data[:, colIndex] = value
            self.types[colIndex] = np.array(value).dtype.type
        else:
            self.colNames = np.append(self.colNames, str(arg))
            self.data = np.column_stack((self.data, value))
            self.types.append(np.array(value).dtype.type)

#Method .get_row(index) that returns a list of values corresponding to the row
    #    Cover cases with multiple arguments (if needed?)
    def get_row(self, index):
        if(index not in range(self.data.shape[0])):
            raise IndexError('Index {0} out of range.'.format(index))
        return list(self.data[index])

#Methods .sum(), .median(), .min() and .max() that, ignoring the
#non-numerical columns, return a list of values corresponding to applying
#the function to each numerical column

    def aggFunction(self, function):
        trans = np.transpose(self.data)
        returnVal = []
        for i in range(len(self.colNames)):
            if(np.issubdtype(self.types[i], np.number)):
                returnVal.append(function(trans[i]))
        return returnVal

    def sum(self):
        return self.aggFunction(np.sum)

    def median(self):
        return self.aggFunction(np.median)

    def max(self):
        return self.aggFunction(np.max)

    def min(self):
        return self.aggFunction(np.min)

    def mean(self):
        return self.aggFunction(np.mean)

    def var(self):
        return self.aggFunction(np.var)

    def std(self):
        return self.aggFunction(np.std)
    
#Find Index of minimum value
    def argmin(self):
        return self.aggFunction(np.argmin)

#Find Index of maximum value
    def argmax(self):
        return self.aggFunction(np.argmax)




