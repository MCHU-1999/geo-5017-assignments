import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from os.path import exists, join
from os import listdir
from typing import Dict, Literal, Callable
import itertools
from sklearn.metrics import hinge_loss
from sklearn.metrics import log_loss
from urban_obj import urban_object


def feature_preparation(data_path: str, feature_names: list[str], o_filename='data.txt'):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    # data_file = 'data.txt'
    if exists(o_filename):
        print(f"{o_filename} already exist, will proceed with the file.")
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        if file_name.split('.')[-1] != 'xyz':
            continue

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features(feature_names)

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = 'ID,label,' + ','.join(feature_names)
    # data_header = 'ID,label,height,root_density,area,shape_index,linearity,sphericity'
    np.savetxt(o_filename, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)

def normalize_feature(X: np.ndarray):
    XT = X.transpose()
    for row in XT:
        max = np.amax(row)
        min = np.amin(row)
        diff = max - min
        row = (row - min) / diff
        assert np.amax(row) == 1
        assert np.amin(row) == 0

    return XT.transpose()


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')
    header = np.loadtxt(data_file, dtype=str, delimiter=',',max_rows=1, comments=None)

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)
    X = normalize_feature(X)
    
    feature_names = header.reshape((-1))[2:]
    # print(feature_names)

    return ID, X, y, feature_names

def feature_visualization_2d(X, reduce: Literal["PCA","TSNE"] | None, features):
    """
    Visualize the features using PCA or T-SNE
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title(f"{reduce if reduce else ''} feature visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    # Reduce dimensionality to 2D
    if reduce == 'PCA':
        pca = PCA(n_components=2)
        X_new = pca.fit_transform(X)
    elif reduce == 'TSNE':
        tsne = TSNE(n_components=2, perplexity=20, random_state=420)
        X_new = tsne.fit_transform(X)
    else:
        print("Didn't specify dimensionality reducing method, plotting the first 2 features.")
        X_new = X

    for i in range(5):
        ax.scatter(X_new[100*i:100*(i+1), 0], X_new[100*i:100*(i+1), 1], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    ax.set_xlabel(f'x1: {features[0] if not reduce else ''}')
    ax.set_ylabel(f'x2: {features[1] if not reduce else ''}')
    ax.legend()
    plt.show()


def manual_learning_curve(X, y, classifier_type, param_set, test_sizes=np.linspace(0.1, 0.9, 9)):
    """
    Generate a learning curve by manually varying train-test splits

    Parameters:
    - X: Feature matrix
    - y: Label vector
    - classifier_type: String ('svm' or 'random_forest')
    - param_set: Dictionary of hyperparameters
    - test_sizes: Array of test set proportions
    """
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []
    sizes = []

    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Initialize model
        if classifier_type == 'svm':
            model = svm.SVC(**param_set)
        elif classifier_type == 'random_forest':
            model = RandomForestClassifier(**param_set)
        else:
            raise ValueError("classifier_type must be 'svm' or 'random_forest'")

        model.fit(X_train, y_train)
        sizes.append(len(X_train))

        # Store metrics
        train_acc.append(accuracy_score(y_train, model.predict(X_train)))
        test_acc.append(accuracy_score(y_test, model.predict(X_test)))

        if classifier_type == 'svm':
            train_loss.append(hinge_loss(y_train, model.decision_function(X_train)))
            test_loss.append(hinge_loss(y_test, model.decision_function(X_test)))
        else:
            train_loss.append(log_loss(y_train, model.predict_proba(X_train)))
            test_loss.append(log_loss(y_test, model.predict_proba(X_test)))

        # --- CALCULATE FINAL METRICS (LAST ELEMENT) ---
    final_size = sizes[-1]
    final_train_acc = train_acc[-1]
    final_test_acc = test_acc[-1]
    final_train_loss = train_loss[-1]
    final_test_loss = test_loss[-1]
    final_gap = final_test_loss - final_train_loss

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Accuracy Plot ---
    ax1.plot(sizes, train_acc, 'r--o', markersize=8, label="Train Accuracy")
    ax1.plot(sizes, test_acc, 'g-s', markersize=8, label="Test Accuracy")
    ax1.set_title("Accuracy Learning Curve", fontsize=14, pad=20)
    ax1.set_xlabel("Number of Training Samples", fontsize=12)
    ax1.set_ylabel("Accuracy Score", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)

    # Annotate FINAL values only
    ax1.annotate(f'Final Train: {final_train_acc:.3f}',
                 xy=(final_size, final_train_acc),
                 xytext=(10, 10), textcoords='offset points',
                 ha='left', va='bottom', fontsize=10)
    ax1.annotate(f'Final Test: {final_test_acc:.3f}',
                 xy=(final_size, final_test_acc),
                 xytext=(10, -20), textcoords='offset points',
                 ha='left', va='top', fontsize=10)

    # --- Loss Plot ---
    ax2.plot(sizes, train_loss, 'b--o', markersize=8, label="Train Loss")
    ax2.plot(sizes, test_loss, 'm-s', markersize=8, label="Test Loss")
    ax2.set_title("Loss Learning Curve", fontsize=14, pad=20)
    ax2.set_xlabel("Number of Training Samples", fontsize=12)
    ax2.set_ylabel("Loss Value", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)

    # Highlight FINAL gap only
    ax2.fill_between(sizes, train_loss, test_loss, color='gray', alpha=0.2)
    ax2.annotate(f'Final Gap: {final_gap:.3f}',
                 xy=(final_size, (final_train_loss + final_test_loss) / 2),
                 xytext=(10, 0), textcoords='offset points',
                 ha='left', va='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

    # Annotate FINAL values only
    ax2.annotate(f'Final Train: {final_train_loss:.3f}',
                 xy=(final_size, final_train_loss),
                 xytext=(10, 10), textcoords='offset points',
                 ha='left', va='bottom', fontsize=10)
    ax2.annotate(f'Final Test: {final_test_loss:.3f}',
                 xy=(final_size, final_test_loss),
                 xytext=(10, -20), textcoords='offset points',
                 ha='left', va='top', fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print numerical summary (with clear final values)
    print("\nLearning Curve Summary:")
    print(f"{'Train Size':<15}{'Train Acc':<12}{'Test Acc':<12}{'Train Loss':<12}{'Test Loss':<12}")
    for i, size in enumerate(sizes):
        print(f"{size:<15}{train_acc[i]:<12.3f}{test_acc[i]:<12.3f}"
              f"{train_loss[i]:<12.3f}{test_loss[i]:<12.3f}")

    print("\n=== FINAL METRICS ===")
    print(f"{'Train Size':<15}{'Train Acc':<12}{'Test Acc':<12}{'Train Loss':<12}{'Test Loss':<12}{'Gap':<12}")
    print(f"{final_size:<15}{final_train_acc:<12.3f}{final_test_acc:<12.3f}"
          f"{final_train_loss:<12.3f}{final_test_loss:<12.3f}{final_gap:<12.3f}")


def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    """
        Conduct SVM classification with manual learning curve for best parameters
        """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define hyperparameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced']
    }

    # Find best parameters
    best_acc = 0
    best_params = None
    predictions = {}

    param_sets = [dict(zip(param_grid.keys(), values))
                  for values in itertools.product(*param_grid.values())]

    for params in tqdm(param_sets, total=len(param_sets)):
        model = svm.SVC(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        predictions[str(params)] = acc

        if acc > best_acc:
            best_acc = acc
            best_params = params

    # Print all results
    for params, acc in predictions.items():
        print(f"Parameters: {params}")
        print(f"Accuracy: {acc:.4f}")
        print("-" * 50)

    # Generate learning curve for best parameters
    print("\nGenerating learning curve for best parameters:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_acc:.4f}")

    # Generate learning curve for best parameters
    print("\nGenerating learning curve for best parameters:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_acc:.4f}")

    manual_learning_curve(X, y, 'svm', best_params)


def RF_classification(X, y):
    """
    Conduct RF classification with manual learning curve for best parameters
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"],
        'class_weight': [None, 'balanced']
    }

    # Find best parameters
    best_acc = 0
    best_params = None
    predictions = {}

    param_sets = [dict(zip(param_grid.keys(), values))
                  for values in itertools.product(*param_grid.values())]

    for params in tqdm(param_sets, total=len(param_sets)):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        predictions[str(params)] = acc

        if acc > best_acc:
            best_acc = acc
            best_params = params

    # Print all results
    for params, acc in predictions.items():
        print(f"Parameters: {params}")
        print(f"Accuracy: {acc:.4f}")
        print("-" * 50)

    # Generate learning curve for best parameters
    print("\nGenerating learning curve for best parameters:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_acc:.4f}")

    manual_learning_curve(X, y, 'random_forest', best_params)


# This is used in feature_preparation() function, please update it properly.
SELECTED_FEATURES = [
    "height", 
    "hw_ratio", 
    "2d_density", 
    "sphericity"
]

if __name__=='__main__':
    # specify the data folder
    path = './pointclouds-500'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(path, SELECTED_FEATURES)

    # load the data
    print('Start loading data from the local file')
    ID, X, y, features = data_loading()

    # visualize features
    print('Visualize the features')
    feature_visualization_2d(X=X, reduce=None, features=features)

    # # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # # RF classification
    # print('Start RF classification')
    # RF_classification(X, y)