import numpy as np
import math
import random
import scipy
from scipy import special

import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path_length

import gurobipy as gp
from gurobipy import GRB


def random_laplace_noise(eps):
    """
        This function generates a random planer laplace noise with the given epsilon as the scale.
        Privacy level: eps.

        Parameters
        ----------
        eps : float
            The scale of targeted planer laplace distribution (>0).

        Returns
        -------
        x : float
            x coordinate of the generated laplace noise.
        y : float
            y coordinate of the generated laplace noise.

    """
    # generate polar coordinates
    theta = np.random.uniform(0, 2 * math.pi)  # this shall be the angular coordinate

    p = random.random()  # draw a random sample from unif(0, 1)
    r = -1 / eps * (scipy.special.lambertw((p - 1) / math.e, k=-1, \
                                           tol=1e-8).real + 1)  # this shall be the radial coordinate

    # convert polar coordinates to cartesian coordinates
    x, y = r * math.cos(theta), r * math.sin(theta)

    return x, y


def batch_laplace_noise(batch_size, eps):
    """
        This function generates random planer laplace noise with the given epsilon in batch.

        Parameters
        ----------
        batch_size: int
            the number of planner laplace noise to be generated.
        eps : float
            The scale of targeted planer laplace distribution (>0).

        Returns
        -------
        noise : numpy.ndarray
            a numpy array of shape (batch_size, 2) containing a number of planer laplace noise coordinates.

    """
    noise = np.zeros((batch_size, 2))
    for i in range(batch_size):
        noise[i, :] = random_laplace_noise(eps)
    return noise


def generate_spanner(x_list, dQ, stretch=1.1):
    """
        This function generates a graph spanner with the given stretch factor (dilation factor). For the formal
        definition of graph spanner: https://en.wikipedia.org/wiki/Geometric_spanner.

        Parameters
        ----------
        x_list : list
            The list of locations for generating the spanner. Eg. [(1, 3), (-2, 2), (4, 1)]
        dQ : function
            The distance metric used for assigning weights to edges.
        stretch (optional) : float, default 1.1
            The stretch factor for the targeted graph spanner.

        Returns
        -------
        spanner : networkx.Graph
            A graph spanner where the distance between any two vertices is stretch by no more than a factor of stretch.

    """
    G = nx.Graph()
    n = len(x_list)
    G.add_nodes_from(range(n))
    weighted_edges = [(i, j, dQ(x_list[i], x_list[j])) for i in range(n) for j in range(n)]
    G.add_weighted_edges_from(weighted_edges)

    sorted_edges = sorted(weighted_edges, key=lambda x: x[2])
    spanner = nx.Graph()
    spanner.add_nodes_from(range(n))
    print('Start generating graph spanner. Please wait for a while. '
          'This process might take longer if the stretch factor is small.')
    for edge in sorted_edges:
        source, target, weight = edge
        if not nx.has_path(spanner, source, target):
            spanner.add_weighted_edges_from([edge])
        else:
            spanner_weight = shortest_path_length(spanner, source, target, weight='weight')
            if spanner_weight < weight * stretch:
                continue
            else:
                spanner.add_weighted_edges_from([edge])
    print(f'A graph spanner with stretch {stretch} has been generated.')
    return spanner


def optql_graph_spanner(x_list, pi_list, spanner, dQ, epsilon=0.5, delta=1.1):
    """
        This function implements a geo-indistinguishable mechanism of optimal utility with graph spanner. For more
        details, please check the paper: https://hal.inria.fr/hal-01114241/document.
        Privacy Level: epsilon / delta (this is no greater than epsilon since delta >= 1)

        Parameters
        ----------
        x_list : list
            The list of locations for generating the spanner. Eg. [(1, 3), (-2, 2), (4, 1)]
        pi_list : list
            The weight of each location in x_list. It should have the same number of elements as x_list. Normalization
            will be done during the mechanism. Eg. [3, 5, 2]
        spanner : networkx.Graph
            A graph spanner for the complete graph generated from x_list. Such spanner can be generated using
            generate_spanner method.
        dQ : function
            The distance metric used for assigning weights to edges. The function should take two locations and return
            a scaler distance.
            Eg. dQ = lambda (x, y): math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
        epsilon (optional) : float, default 0.5
            The parameter determining privacy level for this mechanism. The final privacy level would be epsilon/delta.
        stretch (optional) : float, default 1.1
            The stretch factor for the targeted graph spanner.

        Returns
        -------
        matrix : numpy.ndarray
            The generated stochastic transition matrix. Given n locations, the shape of this matrix would be (n, n).
        pre_prob : numpy.ndarray
            The normalized pre-process probability distribution
        post_prob : numpy.ndarray
            The post-process probability distribution
    """
    try:
        assert len(x_list) == len(pi_list)
    except AssertionError:
        print('x_list and pi_list should have the same length.')
    print(f'Start building a linear program for {len(x_list)} locations...')
    pre_prob = np.array(pi_list) / sum(pi_list)  # normalize probability distribution
    threshold = math.exp(epsilon / delta)
    n = len(x_list)

    # define a model
    model = gp.Model('OptQL')

    # add variables accessed as (0, 0), (0, 1), (1, 1), ...
    variables = model.addVars(n, n, lb=0.0, ub=1.0, name='k')

    # set objective function
    model.setObjective(gp.quicksum(pre_prob[i] * variables[i, j] * dQ(x_list[i], x_list[j]) \
                                   for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # add constraints (1)
    print('Adding differential privacy constraints...')
    model.addConstrs(variables[i, k] <= pow(threshold, dQ(x_list[i], x_list[j])) * variables[j, k] \
                     for (i, j) in spanner.edges for k in range(n))
    model.addConstrs(variables[i, k] <= pow(threshold, dQ(x_list[i], x_list[j])) * variables[j, k] \
                     for (j, i) in spanner.edges for k in range(n))

    # add constraints (2)
    print('Add probability sum constraints...')
    model.addConstrs(gp.quicksum(variables.select(i, '*')) == 1 for i in range(n))

    # constraints (3) are already satisfied

    # optimize the model
    print('Start solving the model...')
    model.optimize()

    # build a matrix to store the stochastic matrix
    variables = model.getAttr('x', variables)
    matrix = np.zeros((n, n))
    for key, value in variables.items():
        matrix[key] = value

    # get post-process probability distribution
    post_prob = pre_prob @ matrix

    return matrix, pre_prob, post_prob


def optql(x_list, pi_list, dQ, epsilon=0.5):
    """
        This function implements a geo-indistinguishable mechanism of optimal utility without graph spanner. For more
        details, please check the paper: https://hal.inria.fr/hal-01114241/document.
        Privacy Level: epsilon

        Parameters
        ----------
        x_list : list
            The list of locations for generating the spanner. Eg. [(1, 3), (-2, 2), (4, 1)]
        pi_list : list
            The weight of each location in x_list. It should have the same number of elements as x_list. Normalization
            will be done during the mechanism. Eg. [3, 5, 2]
        dQ : function
            The distance metric used for assigning weights to edges. The function should take two locations and return
            a scaler distance.
            Eg. dQ = lambda (x, y): math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
        epsilon (optional) : float, default 0.5
            The parameter determining privacy level for this mechanism. The final privacy level would be epsilon.

        Returns
        -------
        matrix : numpy.ndarray
            The generated stochastic transition matrix. Given n locations, the shape of this matrix would be (n, n).
        pre_prob : numpy.ndarray
            The normalized pre-process probability distribution
        post_prob : numpy.ndarray
            The post-process probability distribution
    """
    try:
        assert len(x_list) == len(pi_list)
    except AssertionError:
        print('x_list and pi_list should have the same length.')
    print(f'Start building a linear program for {len(x_list)} locations...')
    pre_prob = np.array(pi_list) / sum(pi_list)  # normalize probability distribution
    n = len(x_list)  # get number of elements
    threshold = math.exp(epsilon)

    # define a model
    model = gp.Model('OptQL')

    # add variables accessed as (0, 0), (0, 1), (1, 1), ...
    variables = model.addVars(n, n, lb=0.0, ub=1.0, name='k')

    # set objective function
    model.setObjective(gp.quicksum(pre_prob[i] * variables[i, j] * dQ(x_list[i], x_list[j]) \
                                   for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # add constraints (1)
    print('Adding differential privacy constraints...')
    model.addConstrs(variables[i, k] <= pow(threshold, dQ(x_list[i], x_list[j])) * variables[j, k] \
                     for i in range(n) for j in range(n) for k in range(n))

    # add constraints (2)
    print('Add probability sum constraints...')
    model.addConstrs(gp.quicksum(variables.select(i, '*')) == 1 for i in range(n))

    # constraints (3) are already satisfied

    # optimize the model
    print('Start solving the model...')
    model.optimize()

    # build a matrix to store the stochastic matrix
    variables = model.getAttr('x', variables)
    matrix = np.zeros((n, n))
    for key, value in variables.items():
        matrix[key] = value

    # get post-process probability distribution
    post_prob = pre_prob @ matrix

    return matrix, pre_prob, post_prob
