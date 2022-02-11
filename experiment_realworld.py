import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import torch
import numpy as np
import pandas as pd
import gc
import time
from copy import deepcopy
from torch.utils.data import DataLoader
# from datasets import hyperbolic_geometric_sgraph
# from embed import create_dataset, get_unobserved, Graph, SamplingGraph, RSGD, Poincare, calc_lik_pc_cpu, calc_lik_pc_gpu
from embed_lvm import CV_HGG, DNML_HGG, LinkPrediction
import torch.multiprocessing as multi
from functools import partial
from scipy.io import mmread


def calc_metrics_realworld(dataset_name, device_idx, model_n_dim):

    adj_mat = np.load('dataset/' + dataset_name + '/' + dataset_name + '.npy')
    print(adj_mat)
    print(len(adj_mat))
    print(np.sum(adj_mat))
    n_nodes = len(adj_mat)

    params_dataset = {
        'n_nodes': n_nodes,
        'R': np.log(n_nodes) - 0.5,
    }

    # パラメータ
    burn_epochs = 300
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    n_max_negatives = n_max_positives * 10
    learning_rate = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更
    sigma_max = 1.0
    sigma_min = 0.001
    beta_min = 0.1
    beta_max = 10.0
    # それ以外
    loader_workers = 8
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = "cuda:" + str(device_idx)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    result = pd.DataFrame()
    basescore_y_and_z_list = []
    basescore_y_given_z_list = []
    basescore_z_list = []
    DNML_codelength_list = []
    pc_first_list = []
    pc_second_list = []
    AIC_naive_list = []
    BIC_naive_list = []
    CV_score_list = []
    AUC_list = []

    basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive, AUC = LinkPrediction(
        adj_mat=adj_mat,
        params_dataset=params_dataset,
        model_n_dim=model_n_dim,
        burn_epochs=burn_epochs,
        burn_batch_size=burn_batch_size,
        n_max_positives=n_max_positives,
        n_max_negatives=n_max_negatives,
        learning_rate=learning_rate,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta_min=beta_min,
        beta_max=beta_max,
        device=device,
        loader_workers=16,
        shuffle=True,
        sparse=False
    )
    basescore_y_and_z_list.append(basescore_y_and_z)
    basescore_y_given_z_list.append(basescore_y_given_z)
    basescore_z_list.append(basescore_z)
    DNML_codelength_list.append(DNML_codelength)
    pc_first_list.append(pc_first)
    pc_second_list.append(pc_second)
    AIC_naive_list.append(AIC_naive)
    BIC_naive_list.append(BIC_naive)
    AUC_list.append(AUC)

    result["model_n_dims"] = [model_n_dim]
    result["DNML_codelength"] = DNML_codelength_list
    result["AIC_naive"] = AIC_naive_list
    result["BIC_naive"] = BIC_naive_list
    # result["CV_score"] = CV_score_list
    result["basescore_y_and_z"] = basescore_y_and_z_list
    result["basescore_y_given_z"] = basescore_y_given_z_list
    result["AUC"] = AUC_list

    result.to_csv("results/" + dataset_name + "/result_" +
                  str(model_n_dim) + ".csv", index=False)


def data_generation(dataset_name):
    # データセット生成
    edges_ids = np.loadtxt('dataset/' + dataset_name +
                           "/" + dataset_name + ".txt", dtype=int)

    ids_all = set(edges_ids[:, 0]) & set(edges_ids[:, 1])
    n_nodes = len(ids_all)
    adj_mat = np.zeros((n_nodes, n_nodes))
    ids_all = list(ids_all)

    for i in range(len(edges_ids)):
        print(i)
        u = np.where(ids_all == edges_ids[i, 0])[0]
        v = np.where(ids_all == edges_ids[i, 1])[0]
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1

    adj_mat = adj_mat.astype(np.int)
    print("n_nodes:", n_nodes)

    np.save('dataset/' + dataset_name + "/" + dataset_name + ".npy", adj_mat)


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('dataset', help='dataset')
    parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('device', help='device')
    args = parser.parse_args()
    print(args)

    if int(args.dataset) == 0:
        dataset_name = "ca-AstroPh"
    elif int(args.dataset) == 1:
        dataset_name = "ca-HepPh"
    elif int(args.dataset) == 2:
        dataset_name = "ca-CondMat"
    elif int(args.dataset) == 3:
        dataset_name = "ca-GrQc"

    data_generation(dataset_name)
    calc_metrics_realworld(dataset_name=dataset_name, device_idx=int(
        args.device), model_n_dim=int(args.n_dim))
