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
from plot import learning_curve


def feature_preparation(data_path: str, feature_names: list[str], o_filename='data.txt'):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
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

    # Reduce dimensionality to 2D (or do nothing)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define hyperparameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1],
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
        predictions[str(params)] = y_pred

        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_params = params

    # Print predictions from different models
    print_param_table(param_grid, param_sets, predictions, y_test)

    # Generate learning curve for best parameters
    print("\nGenerating learning curve for best parameters:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_acc:.4f}")

    # Generate learning curve for best parameters
    print("\nGenerating learning curve for best parameters:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_acc:.4f}")

    learning_curve(X, y, 'svm', best_params)


def RF_classification(X, y):
    """
    Conduct RF classification with manual learning curve for best parameters
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],
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
        model = svm.SVC(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[str(params)] = y_pred

        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_params = params

    # Print predictions from different models
    print_param_table(param_grid, param_sets, predictions, y_test)

    # Generate learning curve for best parameters
    print("\nGenerating learning curve for best parameters:")
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_acc:.4f}")

    learning_curve(X, y, 'random_forest', best_params)

def print_param_table(param_grid: dict, param_sets: list[dict], predictions: dict, gt: list):
    header_list = [f"{key:15s}" for key in param_grid.keys()]
    header = ''.join(header_list)
    print("\n"+header+"acc")
    print("-" * 90)
    for params, pred in zip(param_sets, predictions.values()):
        acc = accuracy_score(gt, pred)
        row_list = [f"{str(value):15}" for value in params.values()]
        row_list.append(f"{acc:.4f}")
        row = ''.join(row_list)
        print(row)

def generate_latex_table(param_grid: dict, param_sets: list[dict], predictions: dict, gt: list) -> str:
    # Extract column names
    headers = list(param_grid.keys()) + ["accuracy"]
    
    # LaTeX table header
    latex_str = "\\begin{table}[h]\n"
    latex_str += "    \\centering\n"
    latex_str += "    \\begin{tabularx}{\\textwidth}{" + "".join("X" for _ in headers) + "}\n"
    latex_str += "        \\toprule\n"
    latex_str += "        " + " & ".join(f"\\textbf{{{col}}}" for col in headers) + " \\\\\n"
    latex_str += "        \\midrule\n"
    
    # Generate table rows
    for params, pred in zip(param_sets, predictions.values()):
        acc = accuracy_score(gt, pred)
        row = " & ".join(str(params[key]) for key in param_grid.keys()) + f" & {acc:.4f}"
        latex_str += f"        {row} \\\\\n"
    
    # Close table
    latex_str += "        \\bottomrule\n"
    latex_str += "    \\end{tabularx}\n"
    latex_str += "    \\caption{[CLASSIFIER] Hyperparameter Results}\n"
    latex_str += "    \\label{tab:rf_results}\n"
    latex_str += "\\end{table}"
    
    return latex_str


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
    feature_visualization_2d(X=X, reduce='TSNE', features=features)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # RF classification
    # print('Start RF classification')
    # RF_classification(X, y)