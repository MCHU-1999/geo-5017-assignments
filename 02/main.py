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

def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Define hyperparameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced']
    }

    # Get all parameter combinations
    param_sets = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

    # Store predictions for each model
    predictions = {}
    for params in tqdm(param_sets, total=len(param_sets)):
        # Train model with specific parameters
        model = svm.SVC(**params)
        model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Store predictions
        predictions[str(params)] = y_pred

    # Print predictions from different models
    for params, pred in predictions.items():
        acc = accuracy_score(y_test, pred)
        print(f"Parameters: {params}")
        print(f"Accuracy: {acc}")
        print("-" * 50)

def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Define hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'max_features': ['sqrt', 'log2'],
        'criterion': ["gini", "entropy", "log_loss"],
        'class_weight': [None, 'balanced']
    }

    # Get all parameter combinations
    param_sets = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]

    # Store predictions for each model
    predictions = {}
    for params in tqdm(param_sets, total=len(param_sets)):
        # Train model with specific parameters
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Store predictions
        predictions[str(params)] = y_pred

    # Print predictions from different models
    for params, pred in predictions.items():
        acc = accuracy_score(y_test, pred)
        print(f"Parameters: {params}")
        print(f"Accuracy: {acc}")
        print("-" * 50)

 
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

    # SVM classification
    # print('Start SVM classification')
    # SVM_classification(X, y)

    # RF classification
    print('Start RF classification')
    RF_classification(X, y)