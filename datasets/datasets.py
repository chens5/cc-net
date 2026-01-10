import sklearn.datasets as datasets
import scipy.sparse as sp
from scipy.sparse import coo_matrix, triu
import torch
from torch_geometric.data import Data
from . import dataset_utils as utils
import graphlearning as gl


# def create_two_moons(n_samples, noise=0.15, **kwargs):
#     return datasets.make_moons(n_samples, noise=noise)

def two_moons(n_samples, noise=0.15, **kwargs):
    return datasets.make_moons(n_samples, noise=noise)

def images(dataset, metric='raw', **kwargs):
    """https://jwcalder.github.io/GraphLearning/datasets.html"""
    return gl.datasets.load(dataset, metric)

# def create_image_dset(dataset, metric='raw', **kwargs):
#     """https://jwcalder.github.io/GraphLearning/datasets.html"""
#     return gl.datasets.load(dataset, metric)

def single_knn(cfg):
    params = cfg['params']
    generator = globals()[params['base']]
    X, labels = generator(**params)
    W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
    W.setdiag(0); W.eliminate_zeros()
    data = utils.graphlearning_to_pyg(X, W)
    return [data]

# def create_knn_dataset_from_base(cfg):
#     params = cfg['params']
#     X, labels = globals()[cfg['type']](**params)
#     W = gl.weightmatrix.knn(X, k=10, kernel='gaussian')
#     W.setdiag(0); W.eliminate_zeros()
#     data = utils.graphlearning_to_pyg(X, W)
#     return data


