## Assignment 2: Classification
The work is done in a Python 3.12 environment by Group 9: Ming-Chieh Hu, Neelabh Singh & Frederick Auer. 
- Run `main.py` if you want to reproduce our results. 
- Run `feature_selection.py` to see how each feature is evaluated and ranked according to our selection criteria.

### Getting started
Install these packages before use:
```
contourpy==1.3.1
cycler==0.12.1
fonttools==4.56.0
joblib==1.4.2
kiwisolver==1.4.8
matplotlib==3.10.0
numpy==2.2.3
opencv-python==4.11.0.86
packaging==24.2
pandas==2.2.3
pillow==11.1.0
pyparsing==3.2.1
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-learn==1.6.1
scipy==1.15.2
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.6.0
tqdm==4.67.1
tzdata==2025.2
```
Or simply:
```
pip install -r requirements.txt
```

### How to run the code
Simply use `python main.py`  
or `python feature_selection.py`.
The order doesn't matter.

### File format
Each 'xyz' file contains the point cloud of a single object, in which each line has three floating point numbers denoting the x, y, and z coordinates of a 3D point.

#### Ground truth labels
```
000 - 099: building
100 - 199: car
200 - 299: fence
300 - 399: pole
400 - 499: tree
```
#### Credit and references
All point clouds are taken from the DALES Objects dataset. More details about the dataset can be found in the following paper:
Singer et al. DALES Objects: A Large Scale Benchmark Dataset for Instance Segmentation in Aerial Lidar, 2021.
