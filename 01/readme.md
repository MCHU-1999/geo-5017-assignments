## Assignment 1: Linear Regression
The work is done in a Python 3.12 environment by Group 9: Ming-Chieh Hu, Neelabh Singh & Frederick Auer. 
- Run `main.py` if you want to reproduce the results of the two regression models (see below on how to set parameters). 
- Run `cylindrical_model.py` to see our experiment on a cylindrical tracjectory model for this specific dataset.

### Getting started
Install these packages before use:
```
contourpy==1.3.1
cycler==0.12.1
fonttools==4.56.0
kiwisolver==1.4.8
matplotlib==3.10.0
numpy==2.2.3
packaging==24.2
pillow==11.1.0
pyparsing==3.2.1
python-dateutil==2.9.0.post0
six==1.17.0
```
Or simply:
```
pip install -r requirements.txt
```

### How to run the code
Simply use `python main.py -d 1 -it 100 -lr 0.01 -ct 0.001`  
or `python main.py -d 2 -it 100 -lr 0.0001 -ct 0.001` 
to reproduce the results in the report. 

#### Parameters
There are 4 paramaters to be set, descriptions as below:
- `-lr` learning rate: learning rate of the gradient descent algorithm
- `-it` iterations: the max iteration number
- `-d` degrees: the degree for the polynomial model, 1: constant speed model, 2: constant acceleration model
- `-ct` convergence threshold: determines when an iterative algorithm should stop, if changes in the objective function become negligible according to the threshold value.

Use `python main.py -h` for help.