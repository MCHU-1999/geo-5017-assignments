import numpy as np
from urban_obj import urban_object, FEATURES
from main import data_loading, feature_preparation
import matplotlib.pyplot as plt
import seaborn as sns


def in_class_scatter(X: np.ndarray, Y: np.ndarray, N=500):
    classes = np.unique(Y)
    n = X.shape[1]
    Sw = np.zeros(shape=(n, n), dtype=np.float32)

    for class_id in classes:
        y_filter = (Y == class_id)
        filtered_x = X[y_filter]
        covariance = np.cov(filtered_x.transpose())

        Sw += (len(filtered_x)/N) * covariance

    return Sw

def between_class_scatter(X: np.ndarray, Y: np.ndarray, N=500):
    classes = np.unique(Y)
    mean = np.mean(X, axis=0)
    n = X.shape[1]
    Sb = np.zeros(shape=(n, n), dtype=np.float32)
    
    for class_id in classes:
        y_filter = (Y == class_id)
        filtered_x = X[y_filter]
        class_mean = np.mean(filtered_x, axis=0)
        vec = class_mean - mean
        # n = len(vec)
        vec = vec.reshape(n, -1)
        vec_T = vec.reshape(-1, n)
        mat = np.matmul(vec, vec_T)

        Sb += (len(filtered_x)/N) * mat
    
    return Sb

def plot_heatmap(matrix: np.ndarray[np.ndarray[np.float32]], plot_title: str):
        feature_keys = list(FEATURES.keys())

        # Plot matrix
        plt.figure(figsize=(8, 8))
        sns.heatmap(
            matrix, annot=False, fmt="f", cmap="Purples",
            # vmin=0, vmax=10,
            xticklabels=feature_keys, yticklabels=feature_keys
        )

        plt.title(f"{plot_title}")
        plt.ylabel("Features")
        plt.xlabel("Features")
        plt.tight_layout()
        plt.savefig(f"./feature_cov/{plot_title}.png")

if __name__ == "__main__":
    path = './pointclouds-500'
    feature_keys = list(FEATURES.keys())

    print('Start preparing features (to test Var and Cov)')
    feature_preparation(path, feature_keys, 'data_11_feature.txt')
    ID, X, y, features = data_loading('data_11_feature.txt')

    Sw = in_class_scatter(X, y, 500)
    Sb = between_class_scatter(X, y, 500)
    # plot_heatmap(Sw, "Sw")
    # plot_heatmap(Sb, "Sb")
    # print(Sw)
    # print(Sb)

    J = np.diag(Sb / Sw)
    ranked_indices = np.argsort(J)[::-1]    # Sort in descending order
    
    print("J values")
    print("-"*40)
    for i in ranked_indices:
        print(f"{feature_keys[i]:20s} {J[i]:.4f}")





