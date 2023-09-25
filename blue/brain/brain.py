import numpy as np
import pandas as pd

import scipy
from scipy.linalg import fractional_matrix_power
from scipy.linalg import logm, expm

#IN linalg.py
#__all__ = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv', 'inv',
#           'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'slogdet', 'det',
#           'svd', 'eig', 'eigh', 'lstsq', 'norm', 'qr', 'cond', 'matrix_rank',
#           'LinAlgError', 'multi_dot']

__all__ = ['upper_vectorization','lower_vectorization', 'double_vectorization','geodesic_distance', 'ID_rate', 'ID_rate2', 'fhd_map',
        'geodesic_k']


def fhd_map(x):
    if x >= 0.25:
        y = '3_fhp'
    elif x>0 and x<0.25:
        y = '2_fha'
    else:
        y = '1_fhn'
    return y

def fhd_numeric_map(x):
    if x >= 0.25:
        y = 2
    elif x>0 and x<0.25:
        y = 1
    else:
        y = 0
    return y

## TDO: count edges assertion
def upper_vectorization(A:np.array):
    triangle = np.triu(A, 1)
    vector = triangle[np.triu_indices(triangle.shape[0], 1)]

    return vector

def lower_vectorization(A:np.array):
    triangle = np.tril(A, -1)
    vector = triangle[np.tril_indices(triangle.shape[0], -1)]

    return vector

def double_vectorization(A:np.array):
    up_triangle = np.triu(A, 1)
    up_vector = up_triangle[np.triu_indices(up_triangle.shape[0],1)]

    low_triangle = np.tril(A, -1)
    low_vector = low_triangle[np.tril_indices(low_triangle.shape[0], -1)]

    vector = np.hstack([up_vector, low_vector])  

    return vector



def geodesic_distance(a,b):
    
    a_ = fractional_matrix_power(a, -0.5)
    q = a_@ b @a_
    #eye = np.diagonal(q)
    #print(eye)
    #trace_q = np.trace(q)
    #print(trace_q)
    q = logm(q)
    trace = np.trace(q**2)
    #print(trace)
    dist = np.sqrt(trace)
    
    return dist

def geodesic_k(a,b):

    q = np.linalg.solve(a,b)
    e, _ = np.linalg.eig(q)
    dist = np.sqrt(np.sum(np.log(e)**2))

    return dist



def ID_rate(A, distance=False):
    matches = []
    for i in range(len(A)):
        # TODO: optimize if distance comparison)
        if distance:
            j = np.argmin(A[i])
        else:
            j = np.argmax(A[i])
        if i == j:
            matches.append(1)
        else:
            print('Missmatch:', i,j)
            matches.append(0)

    id_rate = np.sum(matches)/len(matches)

    return id_rate


def ID_rate2(A, distance=False, verbose=False):
    matches = []
    for i in range(len(A)):
        # TODO: optimize if distance comparison)
        if distance:
            j = np.argmin(A[i])
        else:
            j = np.argmax(A[i])
        if i == j:
            matches.append(1)
        else:
            matches.append(0)
            if verbose:
                print('Missmatch:', i,j)

    forward_rate = np.sum(matches)/len(matches)
    print(forward_rate)

    matches = []
    for j in range(len(A.T)):
        # TODO: optimize if distance comparison)
        if distance:
            i = np.argmin(A.T[j])
        else:
            i = np.argmax(A.T[j])
        if j == i:
            matches.append(1)
        else:
            matches.append(0)
            if verbose:
                print('Missmatch:', j,i)

    backward_rate = np.sum(matches)/len(matches)
    print(backward_rate)

    id_rate = (forward_rate + backward_rate) / 2

    return np.round(id_rate,3)

def matrix_from_vector(v, n):
    matrix = np.zeros((n, n))
    i_upp = np.triu_indices(n, 1)
    matrix[i_upp] = v
    i_low = np.tril_indices(n, -1)
    matrix[i_low] = matrix.T[i_low]
    
    return matrix
