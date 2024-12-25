import torch
import numpy as np
import random
import scipy.sparse as sparse
from sklearn.utils import check_symmetric
from sklearn.preprocessing import normalize
from sklearn import cluster

def same_seeds(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                    9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                    12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                    4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                    9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                    19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    np_labels = np.array(coarse_targets)[targets.numpy()]
    torch_labels = torch.from_numpy(np_labels)
    return torch_labels


def warmup_lr(optimizer,epoch,base_lr,warmup_epoch=10):
    if epoch<warmup_epoch:
        optimizer.param_groups[0]['lr'] = base_lr*min(1.,(epoch+1)/warmup_epoch)


def euclidean_distance(X):
    Y = X
    sum_dims = list(range(2,X.dim()+1))
    X = torch.unsqueeze(X, dim=1)
    distance  = torch.sum(torch.square(X - Y), dim=sum_dims)
    return distance


def full_affinity(X, sigma=1.0):
    distance = euclidean_distance(X)
    scaled_distance = distance / (2 * sigma**2)
    W = torch.exp(-scaled_distance)
    return W


def spectral_clustering(affinity_matrix_, n_clusters, k, seed=1, n_init=20):
    affinity_matrix_ = check_symmetric(affinity_matrix_)
    laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)

    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, 
                                 k=k, sigma=None, which='LA')
    embedding = normalize(vec)

    _, labels_, _ = cluster.k_means(embedding, n_clusters, 
                                         random_state=seed, n_init=n_init)
    return labels_


def get_sparse_coeff(C, non_zeros):
    N = C.shape[0]
    non_zeros = min(N, non_zeros)
    val = []
    indicies = []
    _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)
    
    val.append(C.gather(1, index).reshape([-1]).data.numpy())
    index = index.reshape([-1]).data.numpy()
    indicies.append(index)
    
    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]
    
    C = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C