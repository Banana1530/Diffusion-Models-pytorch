import pandas as pd
import numpy as np
import os
from scipy.sparse import csc_matrix

import time
# import imp

import cvxpy
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels


# NOTE: update this path to point to your xpress license
# %set_env XPAUTH_PATH=/Users/christiankroer/opt/xpauth.xpr


# %set_env XPRESS=/Users/christiankroer/opt/
# import xpress as xp
# xp.controls.outputlog = 0


def eisenberg_gale(valuations, B = None):
    if np.min(valuations[:, 2]) < 1e-7:
        warnings.warn('The provided valuations are badly scaled.')

    num_buyers = int(np.max(valuations[:, 0])) + 1
    num_items = int(np.max(valuations[:, 1])) + 1
    num_pairs = valuations.shape[0]

    if B is None:
        B = np.ones(num_buyers)
    elif B.size != num_buyers:
        raise RuntimeError('The provided budget vector is of the wrong size.')
    elif np.min(B) < 0:
        raise RuntimeError('The provided budget vector has a negative entry.')

    supply = np.ones(num_items)

    P_rows = valuations[:, 1].astype(int)
    P_cols = np.arange(num_pairs)
    P_vals = np.ones(shape=(num_pairs,))
    P = csc_matrix((P_vals, (P_rows, P_cols)))

    U_rows = valuations[:, 0].astype(int)
    U_cols = np.arange(num_pairs)
    U_vals = valuations[:, 2]
    U = csc_matrix((U_vals, (U_rows, U_cols)))

    x = cvxpy.Variable(shape=(num_pairs,), name='x')
    constraints = [P @ x <= supply]
    constraints.append(x >= 0)
    l = cvxpy.Variable(shape=(num_buyers,), nonneg=True, name='l')
    objective = cvxpy.Maximize(B @ cvxpy.log(U @ x + l) - cvxpy.sum(l))

    prob = cvxpy.Problem(objective, constraints)
    prob.__dict__['metadata'] = {'capacities': supply, 'P': P, 'U': U}
    return prob


def items_per_buyer(valuations, skip_sort=False):
    if not skip_sort:
        valuations = valuations[np.argsort(valuations[:, 0]), :]
    d = np.diff(np.hstack([[-1], valuations[:, 0]]))
    startlocs = np.where(d == 1)[0]
    # ^ starting index for each buyer
    startlocs = np.hstack([startlocs, [valuations.shape[0]]])
    # ^ add imaginary "final" buyer
    itemct = np.diff(startlocs)
    return itemct


def buyers_per_item(valuations, skip_sort=False):
    if not skip_sort:
        valuations = valuations[np.argsort(valuations[:, 1]), :]
    d = np.diff(np.hstack([[-1], valuations[:, 1]]))
    startlocs = np.where(d == 1)[0]
    # ^ starting index for each item, in this sorted order
    startlocs = np.hstack([startlocs, [valuations.shape[0]]])
    # ^ add imaginary "final" item
    buyerct = np.diff(startlocs)
    return buyerct


def results_by_item(valuations, allocations, prices):
    order = np.argsort(valuations[:, 1])
    valuations = valuations[order, :]
    allocations = allocations[order]
    widths = buyers_per_item(valuations, skip_sort=True)
    start = 0
    item2result = []
    for i, w in enumerate(widths):
        stop = start + w
        curr_buyers = valuations[start:stop, 0].astype(int)
        curr_values = valuations[start:stop, 2]
        curr_allocs = allocations[start:stop]
        d = {'buyers': curr_buyers,
             'valuations': curr_values,
             'allocations': curr_allocs,
             'price': prices[i] if prices is not None else None}
        item2result.append(d)
        start = stop
    return item2result


def results_by_buyer(valuations, allocations, prices):
    order = np.argsort(valuations[:, 0])
    valuations = valuations[order, :]
    allocations = allocations[order]
    widths = items_per_buyer(valuations, skip_sort=True)
    start = 0
    buyer2result = []
    for w in widths:
        stop = start + w  # inclusive of "start + w".
        curr_items = valuations[start:stop, 1].astype(int)
        curr_value = valuations[start:stop, 2]
        curr_alloc = allocations[start:stop]
        curr_prices = prices[curr_items] if prices is not None else None
        d = {'items': curr_items,
             'valuations': curr_value,
             'allocations': curr_alloc,
             'prices': curr_prices}
        buyer2result.append(d)
        start = stop
    return buyer2result


def convert_dense_valuation(v):
    num_buyers = v.shape[0]
    num_items = v.shape[1]
    buyer_index_vec = np.repeat(np.arange(num_buyers), num_items)
    item_index_vec = np.tile(np.arange(num_items), num_buyers)
    utility_vec = np.ravel(v, order='C')  # C, versus Fortran order.
    # convert the above vectors into columns
    buyer_index_vec = buyer_index_vec.reshape((-1, 1))
    item_index_vec = item_index_vec.reshape((-1, 1))
    utility_vec = utility_vec.reshape((-1, 1))
    # horizontally stack those to obtain a matrix in the required format.
    valuations = np.hstack([buyer_index_vec, item_index_vec, utility_vec])
    return valuations

class fppe_obj:
    def __init__(self, x, be,pr,u,mu,de):
        self.x = x
        self.be = be
        self.pr = pr
        # self.al = al
        self.u = u 
        self.mu = mu 
        self.de = de


def fppe(V, b):


    # solve the EG
    vals = convert_dense_valuation(V)
    prob = eisenberg_gale(vals, b)
    # obj = prob.solve(verbose=False) # fails with 200 buyer items
    obj = prob.solve(solver='MOSEK', verbose=False) # need license, apply it quick; thousands of buyer-items
    runtime = prob.solver_stats.solve_time


    # extract the most basic variables, x and p
    model_variables = prob.variables()
    x = model_variables[0].value
    capacity_constraint = prob.constraints[0]
    p = capacity_constraint.dual_value


    # calculate the rest of variables
    # item_res = results_by_item(vals, x, p)
    buyer_res = results_by_buyer(vals, x, p)
    allocation = np.zeros(V.shape)
    for i in range(V.shape[0]):
        for res_idx in range(buyer_res[i]["items"].size):
            j = buyer_res[i]["items"][res_idx]
            allocation[i, j] = buyer_res[i]["allocations"][res_idx]
            
    beta = b / ((V * allocation).sum(axis=1) + b - allocation @ p)
    u = b / beta
    mu = np.multiply(allocation,V) # buyer x item
    de = b - np.sum(allocation * (p), axis= 1)
    return  fppe_obj(allocation, beta, p, u, mu ,de)



