
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir

DATASET_PATH = "./pointclouds-500/"


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """

        # 1. Height
        height = np.amax(self.points[:, 2]) - np.amin(self.points[:, 2])
        self.feature.append(height)

        # 2. Root density
        root = self.points[np.argmin(self.points[:, 2])].reshape(1, -1)
        radius = 0.5
        kd_tree = KDTree(self.points)
        count = kd_tree.query_radius(root, r=radius, count_only=True)
        root_density = 1.0 * count[0] / len(self.points)
        self.feature.append(root_density)

        # 3. 2D area
        hull_2d = ConvexHull(self.points[:, :2])
        area_2d = hull_2d.volume
        self.feature.append(area_2d)

        # 4. Linearity (from top cluster)
        top = self.points[np.argmax(self.points[:, 2])].reshape(1, -1)
        k_top = max(int(len(self.points) * 0.005), 100)
        idx = kd_tree.query(top, k=k_top, return_distance=False)
        neighbours = self.points[np.squeeze(idx, axis=0)]
        cov = np.cov(neighbours.T)
        w = np.sort(np.linalg.eigvals(cov))
        linearity = (w[2] - w[1]) / (w[2] + 1e-5)
        self.feature.append(linearity)




    #
    #     # calculate the height
    #     height = np.amax(self.points[:, 2])
    #     self.feature.append(height)
    #
    #     # get the root point and top point
    #     root = self.points[[np.argmin(self.points[:, 2])]]
    #     top = self.points[[np.argmax(self.points[:, 2])]]
    #
    #     # construct the 2D and 3D kd tree
    #     kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
    #     kd_tree_3d = KDTree(self.points, leaf_size=5)
    #
    #     # # compute the root point planar density
    #     radius_root = 0.4
    #     count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
    #     root_density = 1.0*count[0] / len(self.points)
    #     self.feature.append(root_density)
    #
    #     # compute the 2D footprint and calculate its area
    #     hull_2d = ConvexHull(self.points[:, :2])
    #     hull_area = hull_2d.volume
    #     self.feature.append(hull_area)
    #
    #     # global densitty
    #     global_density = len(self.points) / (hull_area * height)
    #     self.feature.append(global_density)
    #
    #     #Height Variance
    #     height_variance = np.var(self.points[:, 2])
    #     self.feature.append(height_variance)
    #
    #     # get the hull shape index
    #     hull_perimeter = hull_2d.area
    #     shape_index = 1.0 * hull_area / hull_perimeter
    #     self.feature.append(shape_index)
    #
    #     # obtain the point cluster near the top area
    #     k_top = max(int(len(self.points) * 0.005), 100)
    #     idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
    #     idx = np.squeeze(idx, axis=0)
    #     neighbours = self.points[idx, :]
    #
    #     # obtain the covariance matrix of the top points
    #     cov = np.cov(neighbours.T)
    #     w, _ = np.linalg.eig(cov)
    #     w.sort()
    #
    #     # calculate the linearity and sphericity
    #     linearity = (w[2]-w[1]) / (w[2] + 1e-5)
    #     sphericity = w[0] / (w[2] + 1e-5)
    #     self.feature += [linearity, sphericity]
    #
    #     # Measures how "vertical" the object is (poles score high)
    #     z_mean = np.mean(self.points[:, 2])
    #     verticality = np.sum(np.abs(self.points[:, 2] - z_mean)) / len(self.points)
    #     self.feature.append(verticality)
    #
    #     # Captures flat surfaces (buildings/cars)
    #     cov_all = np.cov(self.points.T)
    #     eigvals_all = np.sort(np.linalg.eigvals(cov_all))
    #     planarity = (eigvals_all[1] - eigvals_all[0]) / eigvals_all[2]
    #     self.feature.append(planarity)
    #
    #     # Trees are bottom-heavy; poles are uniform
    #     from scipy.stats import skew
    #     height_skew = skew(self.points[:, 2])
    #     self.feature.append(height_skew)




def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = 'ID,label,height,root_density,area,shape_index,linearity,sphericity'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    # plot the data with first two features
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 3], X[100*i:100*(i+1), 4], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    """
    Replace the axis labels with your own feature names
    """
    ax.set_xlabel('x1:root density')
    ax.set_ylabel('x2:area')
    ax.legend()
    plt.show()


def SVM_classification(X, y):
    # Split data (use random_state for reproducibility)
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y  # Preserve class distribution
    )

    # Check class distribution
    class_counts = np.bincount(y)
    print("Class distribution:", class_counts)

    # Define parameter grid with class weighting
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 1],
        'class_weight': ['balanced', None]  # Test both weighted and unweighted
    }

    # Use 5-fold stratified cross-validation
    grid_search = GridSearchCV(
        svm.SVC(),
        param_grid,
        cv=5,
        verbose=1,
        scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)

    # Get best model
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)

    # Evaluation
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Feature importance (for linear kernel)
    if grid_search.best_params_['kernel'] == 'linear':
        print("\nFeature coefficients:", best_svm.coef_)

def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y  # Maintain class distribution
    )

    # Enhanced parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'max_features': ['sqrt', 'log2', 0.3],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }

    # Use parallel processing (n_jobs=-1) for faster tuning
    clf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_preds = clf.predict(X_test)
    print("\nBest parameters:", clf.best_params_)
    print("Accuracy: %.2f" % accuracy_score(y_test, y_preds))
    print("Classification Report:\n", classification_report(y_test, y_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_preds))

    # Feature importance
    importances = clf.best_estimator_.feature_importances_
    print("\nTop Features:")
    for i, imp in sorted(enumerate(importances), key=lambda x: x[1], reverse=True):
        print(f"Feature {i}: {imp:.4f}")

def learning_curve(X, y, classifier, train_sizes=np.linspace(0.1, 1.0, 10, endpoint=False)):
    """
    Generate a learning curve for the classifier.
    """
    train_scores = []
    test_scores = []

    for size in train_sizes:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=None)  # Remove fixed random_state

        # Train the classifier on the current training subset
        classifier.fit(X_train, y_train)

        # Calculate training and testing scores
        train_score = classifier.score(X_train, y_train)  # Training score on the training subset
        test_score = classifier.score(X_test, y_test)     # Testing score on the testing subset

        # Append scores to the lists
        train_scores.append(train_score)
        test_scores.append(test_score)

    # Plot the learning curve
    plt.plot(train_sizes, train_scores, label="Training score")
    plt.plot(train_sizes, test_scores, label="Testing score")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

def error_analysis(y_true, y_pred):
    """
    Perform error analysis using confusion matrix and classification report.
    """
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


if __name__=='__main__':
    # specify the data folder
    path = DATASET_PATH

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # visualize features
    print('Visualize the features')
    feature_visualization(X=X)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # RF classification
    print('Start RF classification')
    RF_classification(X, y)

    # Generate learning curve for SVM
    print('Generating learning curve for SVM')
    learning_curve(X, y, svm.SVC(C=10, kernel='rbf', gamma='scale'))

    # Generate learning curve for RF
    print('Generating learning curve for RF')
    learning_curve(X, y, RandomForestClassifier(n_estimators=200, max_depth=None, max_features='sqrt'))
