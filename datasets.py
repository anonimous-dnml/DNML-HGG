import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate
from copy import deepcopy

np.random.seed(0)

INTEGRAL_DIV = 10000


def e_dist(u, v):
    return np.sqrt(np.sum((u - v)**2))


def h_dist(u_e, v_e):
    ret = 1
    ret += (2 * e_dist(u_e, v_e)**2) / \
        ((1 - e_dist(0, u_e)**2) * (1 - e_dist(0, v_e)**2))
    return np.arccosh(ret)


def connection_prob(d, R, beta):
    """
    接続確率
    """
    return 1 / (1 + beta * np.exp((d - R)))


def integral_sin(n, theta):
    if n == 0:
        return theta
    elif n == 1:
        return 1 - np.cos(theta)
    else:
        return -np.cos(theta) * (np.sin(theta)**(n - 1)) / n + ((n - 1) / n) * integral_sin(n - 2, theta)


def calc_dist_angle(n_dim, n, div=INTEGRAL_DIV):
    # nは1からn_dim-1となることが想定される。0次元目はr
    if n_dim - 1 == n:
        theta_array = 2 * np.pi * np.arange(0, div + 1) / div
        cum_dens = theta_array / (2 * np.pi)
    else:
        theta_array = np.pi * np.arange(0, div + 1) / div
        # numerator = lambda theta: np.sin(theta)**(n_dim - 1 - n)
        cum_dens = []
        for theta in theta_array:
            # cum_dens.append(integrate.quad(numerator, 0, theta)[0])
            cum_dens.append(integral_sin(n_dim - 1 - n, theta))
        cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return theta_array, cum_dens


def calc_dist_r(n_dim, sigma, R, div=INTEGRAL_DIV):
    # n_dimかRが大きくなると現状だと数値積分があまりうまくいかない。divを増やす必要がある。
    # 発散を防ぐために、exp(sigma*R*(n_dim-1))/(2**(n_dim-1))(分子の積分の支配項)で割ってある。

    def integral_sinh(n, r):  # (exp(sigma*R)/2)^(D-1)で割った結果
        if n == 0:
            return r * (2 * np.exp(-sigma * R))**(n_dim - 1)
        elif n == 1:
            return 1 / sigma * (np.exp(sigma * (r - R)) + np.exp(- sigma * (r + R)) - 2 * np.exp(-sigma * R)) * (2 * np.exp(-sigma * r))**(n_dim - 2)
        else:
            ret = 1 / (sigma * n)
            ret = ret * (np.exp(sigma * (r - R)) - np.exp(-sigma * (r + R))
                         )**(n - 1) * (np.exp(sigma * (r - R)) + np.exp(-sigma * (r + R)))
            ret = ret * (2 * np.exp(-sigma * R)
                         )**(n_dim - 1 - n)
            return ret - (n - 1) / n * integral_sinh(n - 2, r)

    numerator = lambda r: (
        (np.exp(sigma * (r - R)) - np.exp(-sigma * (r + R))))**(n_dim - 1)
    r_array = R * np.arange(0, div + 1) / div
    cum_dens = []
    for r in r_array:
        cum_dens.append(integral_sinh(n_dim - 1, r))
    cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return r_array, cum_dens


def hyperbolic_geometric_graph(n_nodes, n_dim, R, sigma, beta):
    # TODO: プログラム前半部分も実行時間を短くする。
    # 現状は次元の2乗オーダーの計算量
    # n_dimは2以上で
    x_polar = np.random.uniform(0, 1, (n_nodes, n_dim))
    # 逆関数法で点を双曲空間からサンプリング
    # 双曲空間の意味での極座標で表示
    for i in range(n_dim):
        if i == 0:
            val_array, cum_dens = calc_dist_r(n_dim, sigma, R)
        else:
            val_array, cum_dens = calc_dist_angle(n_dim, i)
        for j in range(n_nodes):
            idx = np.max(np.where(cum_dens <= x_polar[j, i])[0])
            x_polar[j, i] = val_array[idx]
    # 直交座標に変換(Euclid)
    print('sampling ended')

    x_e = convert_euclid(x_polar)

    print('convert euclid ended')

    adj_mat = np.zeros((n_nodes, n_nodes))
    # 対角
    sampling_mat = np.random.uniform(0, 1, adj_mat.shape)
    sampling_mat = np.triu(
        sampling_mat) + np.triu(sampling_mat).T - np.diag(sampling_mat.diagonal())

    # できる限り行列演算を駆使して計算時間を短くする。
    norm_x_e_2 = np.sum(x_e**2, axis=1).reshape((-1, 1))
    denominator_mat = (1 - norm_x_e_2) * (1 - norm_x_e_2.T)
    numerator_mat = norm_x_e_2 + norm_x_e_2.T
    numerator_mat -= 2 * x_e.dot(x_e.T)
    # arccoshのエラー対策
    for i in range(n_nodes):
        numerator_mat[i, i] = 0
    adj_mat = np.arccosh(1 + 2 * numerator_mat / denominator_mat)
    adj_mat = connection_prob(adj_mat, R, beta)
    # 対角成分は必要ない
    for i in range(n_nodes):
        adj_mat[i, i] = 0
    adj_mat = np.where(sampling_mat < adj_mat, 1, 0)

    print("adj mat generated")

    return adj_mat, x_e


def convert_euclid(x_polar):
    n_nodes = x_polar.shape[0]
    n_dim = x_polar.shape[1]
    x_euclid = np.zeros((n_nodes, n_dim))
    radius = np.sqrt(
        (np.cosh(x_polar[:, 0]) - 1) / (np.cosh(x_polar[:, 0]) + 1))
    for i in range(n_dim):
        x_euclid[:, i] = radius
        for j in range(0, i + 1):
            if j + 1 < n_dim:
                if j == i:
                    x_euclid[:, i] *= np.cos(x_polar[:, j + 1])
                else:
                    x_euclid[:, i] *= np.sin(x_polar[:, j + 1])
    return x_euclid


def create_dataset(
    adj_mat,
    n_max_positives=2,
    n_max_negatives=10,
    val_size=0.1
):
    # データセットを作成し、trainとvalidationに分ける
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i, i] = -1
    # -1はサンプリング済みの箇所か対角要素

    data = []

    for i in range(n_nodes):
        # positiveをサンプリング
        idx_positives = np.where(_adj_mat[i, :] == 1)[0]
        idx_negatives = np.where(_adj_mat[i, :] == 0)[0]
        idx_positives = np.random.permutation(idx_positives)
        idx_negatives = np.random.permutation(idx_negatives)
        n_positives = min(len(idx_positives), n_max_positives)
        n_negatives = min(len(idx_negatives), n_max_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample

        # 隣接行列から既にサンプリングしたものを取り除く
        _adj_mat[i, idx_positives[0:n_positives]] = -1
        _adj_mat[idx_positives[0:n_positives], i] = -1

        for j in idx_negatives[0:n_negatives]:
            data.append((i, j, 0))  # positive sample

        # 隣接行列から既にサンプリングしたものを取り除く
        _adj_mat[i, idx_negatives[0:n_negatives]] = -1
        _adj_mat[idx_negatives[0:n_negatives], i] = -1

    data = np.random.permutation(data)

    train = data[0:int(len(data) * (1 - val_size))]
    val = data[int(len(data) * (1 - val_size)):]
    return train, val


def create_dataset_for_basescore(
    adj_mat,
    n_max_samples,
):
    # データセットを作成し、trainとvalidationに分ける
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i, i] = -1
    # -1はサンプリング済みの箇所か対角要素

    data = []
    # print(np.sum(_adj_mat))

    for i in range(n_nodes):
        idx_samples = np.where(_adj_mat[i, :] != -1)[0]
        idx_samples = np.random.permutation(idx_samples)
        n_samples = min(len(idx_samples), n_max_samples)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_samples[0:n_samples]:
            data.append((i, j, _adj_mat[i, j]))

        # 隣接行列から既にサンプリングしたものを取り除く
        # _adj_mat[i, idx_samples[0:n_samples]] = -1
        # _adj_mat[idx_samples[0:n_samples], i] = -1

    # data = np.random.permutation(data)

    # train = data[0:int(len(data) * (1 - val_size))]
    # val = data[int(len(data) * (1 - val_size)):]
    # print(np.sum(_adj_mat))

    return data

if __name__ == '__main__':

    n_dim_true_list = [4, 8, 16]
    n_nodes_list = [400, 800, 1600, 3200, 6400]
    n_graphs = 10

    for n_dim_true in n_dim_true_list:
        for n_nodes in n_nodes_list:
            for n_graph in range(n_graphs):

                params_adj_mat = {
                    'n_nodes': n_nodes,
                    'n_dim': n_dim_true,
                    'R': np.log(n_nodes) - 0.5,
                    'sigma': 0.1,
                    'beta': 0.3
                }
                adj_mat, x_e = hyperbolic_geometric_graph(
                    n_nodes=params_adj_mat["n_nodes"],
                    n_dim=params_adj_mat["n_dim"],
                    R=params_adj_mat["R"],
                    sigma=params_adj_mat["sigma"],
                    beta=params_adj_mat["beta"]
                )

                graph_dict = {
                    "params_adj_mat": params_adj_mat,
                    "adj_mat": adj_mat
                }

                # 平均次数が少なくなるように手で調整する用
                print('average degree:', np.sum(adj_mat) / len(adj_mat))

                os.makedirs('dataset/dim_' +
                            str(params_adj_mat['n_dim']), exist_ok=True)
                np.save('dataset/dim_' + str(params_adj_mat['n_dim']) + '/graph_' + str(
                    params_adj_mat['n_nodes']) + '_' + str(n_graph) + '.npy', graph_dict)
