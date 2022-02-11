import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate
# from embed import create_dataset
from copy import deepcopy
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score
import matplotlib.pyplot as plt
import matplotlib


def artificial():

    D_true_list = [4, 8, 16]
    n_nodes_list = [400, 800, 1600, 3200, 6400]
    n_graphs = 10
    T_gap = 2

    for D_true in D_true_list:
        if D_true == 4:
            label = [0, 1, 0, 0, 0, 0]
        elif D_true == 8:
            label = [0, 0, 1, 0, 0, 0]
        elif D_true == 16:
            label = [0, 0, 0, 1, 0, 0]

        for n_nodes in n_nodes_list:
            bene_DNML = []
            bene_AIC = []
            bene_BIC = []
            bene_MinGE = []

            for n_graph in range(n_graphs):
                result = pd.read_csv("results/Dim" + str(D_true) + "/result_" +
                                     str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + ".csv")
                result = result.fillna(9999999999999)
                D_DNML = result["model_n_dims"].values[
                    np.argmin(result["DNML_codelength"].values)]
                D_AIC = result["model_n_dims"].values[
                    np.argmin(result["AIC_naive"].values)]
                D_BIC = result["model_n_dims"].values[
                    np.argmin(result["BIC_naive"].values)]

                result_MinGE = pd.read_csv("results/Dim" + str(D_true) + "/result_" +
                                           str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + "_MinGE.csv")
                D_MinGE = result_MinGE["model_n_dims"].values[
                    np.argmin(result_MinGE["MinGE"].values)]

                bene_DNML.append(
                    max(0, 1 - abs(np.log2(D_DNML) - np.log2(D_true)) / T_gap))
                bene_AIC.append(
                    max(0, 1 - abs(np.log2(D_AIC) - np.log2(D_true)) / T_gap))
                bene_BIC.append(
                    max(0, 1 - abs(np.log2(D_BIC) - np.log2(D_true)) / T_gap))
                bene_MinGE.append(
                    max(0, 1 - abs(np.log2(D_MinGE) - np.log2(D_true)) / T_gap))

                plt.clf()
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)

                def normalize(x):
                    return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

                result["DNML_codelength"] = normalize(
                    result["DNML_codelength"])
                result["AIC_naive"] = normalize(result["AIC_naive"])
                result["BIC_naive"] = normalize(result["BIC_naive"])
                result["MinGE"] = normalize(result_MinGE["MinGE"])

                ax.plot(result["model_n_dims"], result[
                        "DNML_codelength"], label="DNML-HGG", color="red")
                ax.plot(result["model_n_dims"], result["AIC_naive"],
                        label="AIC_naive", color="blue")
                ax.plot(result["model_n_dims"], result["BIC_naive"],
                        label="BIC_naive", color="green")
                ax.plot(result["model_n_dims"], result[
                        "MinGE"], label="MinGE", color="orange")
                plt.xscale('log')
                plt.xticks(result["model_n_dims"], fontsize=20)
                plt.yticks(fontsize=20)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
                           borderaxespad=0, fontsize=15)
                ax.set_xlabel("Dimensionality", fontsize=20)
                ax.set_ylabel("Normalized Criterion", fontsize=20)
                plt.tight_layout()

                plt.savefig("results/Dim" + str(D_true) + "/result_" +
                            str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + ".png")

            bene_DNML = np.array(bene_DNML)
            bene_AIC = np.array(bene_AIC)
            bene_BIC = np.array(bene_BIC)
            bene_MinGE = np.array(bene_MinGE)

            print("n_nodes:", n_nodes)
            print("dimensionality:", D_true)
            print("DNML:", np.mean(bene_DNML), "±", np.std(bene_DNML))
            print("AIC:", np.mean(bene_AIC), "±", np.std(bene_AIC))
            print("BIC:", np.mean(bene_BIC), "±", np.std(bene_BIC))
            print("MinGE:", np.mean(bene_MinGE), "±", np.std(bene_MinGE))


def realworld():
    dataset_name_list = ["ca-AstroPh", "ca-CondMat", "ca-GrQc", "ca-HepPh"]

    n_dim_list = [2, 4, 8, 16, 32, 64]

    for dataset_name in dataset_name_list:

        result = pd.DataFrame()

        for n_dim in n_dim_list:
            row = pd.read_csv("results/" + dataset_name +
                              "/result_" + str(n_dim) + ".csv")
            result = result.append(row)

        result_MinGE = pd.read_csv(
            "results/" + dataset_name + "/result_MinGE.csv")

        result = pd.merge(result, result_MinGE, on="model_n_dims", how="left")

        D_DNML = result["model_n_dims"].values[
            np.argmin(result["DNML_codelength"].values)]
        D_AIC = result["model_n_dims"].values[
            np.argmin(result["AIC_naive"].values)]
        D_BIC = result["model_n_dims"].values[
            np.argmin(result["BIC_naive"].values)]
        D_MinGE = result["model_n_dims"].values[
            np.argmin(result["MinGE"].values)]

        print(dataset_name)
        print("DNML:", D_DNML)
        print("AIC:", D_AIC)
        print("BIC:", D_BIC)
        print("MinGE:", D_MinGE)

        print(result["AUC"])

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        def normalize(x):
            return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

        result["DNML_codelength"] = normalize(result["DNML_codelength"])
        result["AIC_naive"] = normalize(result["AIC_naive"])
        result["BIC_naive"] = normalize(result["BIC_naive"])
        result["MinGE"] = normalize(result["MinGE"])

        ax.plot(result["model_n_dims"], result[
                "DNML_codelength"], label="DNML-HGG", color="red")
        ax.plot(result["model_n_dims"], result["AIC_naive"],
                label="AIC_naive", color="blue")
        ax.plot(result["model_n_dims"], result["BIC_naive"],
                label="BIC_naive", color="green")
        ax.plot(result["model_n_dims"], result[
                "MinGE"], label="MinGE", color="orange")
        plt.xscale('log')
        plt.xticks(result["model_n_dims"], fontsize=20)
        plt.yticks(fontsize=20)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
                   borderaxespad=0, fontsize=15)
        ax.set_xlabel("Dimensionality", fontsize=20)
        ax.set_ylabel("Normalized Criterion", fontsize=20)
        plt.tight_layout()

        plt.savefig("results/" + dataset_name +
                    "/result_" + dataset_name + ".png")


if __name__ == "__main__":

    artificial()
    realworld()
