from GeoPrivacy.mechanism import *
from GeoPrivacy.metrics import *
import numpy as np


class DataProcessor:
    def __init__(self, x_list, pi_list):
        self.x_list = x_list
        self.pi_list = pi_list
        self.metric = geodesic
        self.n = len(x_list)

    def customize_metric(self, metric):
        self.metric = metric

    def choose_metric(self, mode, ):
        if mode == 'euclidean':
            self.metric = euclidean
        elif mode == 'mile':
            self.metric = geodesic_mile
        else:
            self.metric = geodesic_km

    def build_model(self, epsilon, delta, mode):
        if mode == 'spanner':
            spanner = generate_spanner(self.x_list, self.dist, delta)
            p_matrix, pre_prob, post_prob = optql_graph_spanner(self.x_list, self.pi_list, spanner, self.metric,
                                                                epsilon=epsilon, delta=delta)
        else:
            p_matrix, pre_prob, post_prob = optql(self.x_list, self.pi_list, self.metric, epsilon=0.5)
        self.p_matrix = p_matrix
        return p_matrix, pre_prob, post_prob

    def generate_point(self, orig_pt):
        return np.random.choice(range(self.n), p=self.p_matrix[orig_pt])
