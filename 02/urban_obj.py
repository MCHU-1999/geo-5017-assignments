import numpy as np
import math
from os import listdir
from tqdm import tqdm
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from typing import Dict, Literal, Callable

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

class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)
        self.n = len(self.points)
        self.kd_tree = KDTree(self.points, leaf_size=5)
        self.hull_2d = ConvexHull(self.points[:, :2])

        # initialize the feature vector
        self.feature = []

    def compute_features(self, feature_names: list[str]):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        self.feature = [ FEATURES[f_name](self) for f_name in feature_names ]

# height
def cal_height(obj: urban_object):
    height = np.amax(obj.points[:, 2]) - np.amin(obj.points[:, 2])
    return height

# height variance
def cal_height_var(obj: urban_object):
    height_var = np.var(obj.points[:, 2])
    return height_var

# height width ratio
def cal_hw_ratio(obj: urban_object):
    height = np.amax(obj.points[:, 2]) - np.amin(obj.points[:, 2])
    area = obj.hull_2d.volume
    return height / math.sqrt(area)

# root_density
def cal_root_density(obj: urban_object):
    height_threshold = 0.5
    ground_level = np.min(obj.points[:, 2])  # Find lowest z-value
    root_mask = obj.points[:, 2] <= (ground_level + height_threshold)  # Points near ground
    root_points = obj.points[root_mask]

    root_density = len(root_points) / obj.n  # Ratio of root points to total
    return root_density

# center of mass neighbor count
def cal_cmass_density(obj: urban_object):
    cmass = obj.points.mean(axis=0)
    cmass = np.reshape(cmass, (1, -1))
    kd_tree = obj.kd_tree

    # compute the neighors / total points
    radius = 0.5
    count = kd_tree.query_radius(cmass, r=radius, count_only=True)
    return 1.0 * count[0] / obj.n

# 2d density
def cal_2d_density(obj: urban_object):
    hull_2d = obj.hull_2d
    area =  hull_2d.volume
    return obj.n / area

# 2d_hull_area
def cal_2d_hull_area(obj: urban_object):
    hull_2d = obj.hull_2d
    return hull_2d.volume

# shape_index
def cal_shape_index(obj: urban_object):
    hull_2d = obj.hull_2d
    hull_perimeter = hull_2d.area
    return 1.0 * hull_2d.volume / hull_perimeter

# compactness
def cal_compactness(obj: urban_object):
    hull_2d = obj.hull_2d
    hull_3d = ConvexHull(obj.points)

    compactness = hull_3d.volume / (hull_2d.volume * cal_height(obj))
    return compactness

# linearity
def cal_linearity(obj: urban_object):
    # obtain the point cluster near the top area
    top = obj.points[[np.argmax(obj.points[:, 2])]]
    kd_tree = obj.kd_tree
    k_top = max(int(obj.n * 0.005), 100)
    idx = kd_tree.query(top, k=k_top, return_distance=False)
    idx = np.squeeze(idx, axis=0)
    neighbours = obj.points[idx, :]

    # obtain the covariance matrix of the top points
    cov = np.cov(neighbours.T)
    w, _ = np.linalg.eig(cov)
    w.sort()

    return (w[2]-w[1]) / (w[2] + 1e-5)

# sphericity
def cal_sphericity(obj: urban_object):
    # obtain the point cluster near the top area
    top = obj.points[[np.argmax(obj.points[:, 2])]]
    kd_tree = obj.kd_tree
    k_top = max(int(obj.n * 0.005), 100)
    idx = kd_tree.query(top, k=k_top, return_distance=False)
    idx = np.squeeze(idx, axis=0)
    neighbours = obj.points[idx, :]

    # obtain the covariance matrix of the top points
    cov = np.cov(neighbours.T)
    w, _ = np.linalg.eig(cov)
    w.sort()

    return w[0] / (w[2] + 1e-5)


FEATURES: Dict[Literal[
    "height", 
    "hw_ratio", 
    "height_var", 
    "root_density", 
    "cmass_density", 
    "2d_density", 
    "2d_hull_area", 
    "shape_index", 
    "compactness",
    "linearity", 
    "sphericity"
], Callable[..., float]] = {
    "height": cal_height,
    "hw_ratio": cal_hw_ratio,
    "height_var": cal_height_var,
    "root_density": cal_root_density,
    "cmass_density": cal_cmass_density,
    "2d_density": cal_2d_density,
    "2d_hull_area": cal_2d_hull_area,
    "shape_index": cal_shape_index,
    "compactness": cal_compactness,
    "linearity": cal_linearity,
    "sphericity": cal_sphericity
}