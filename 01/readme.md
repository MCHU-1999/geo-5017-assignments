## Assignment 1: Linear Regression
The work is done in a Python 3.12 environment by Group 9: Ming-Chieh Hu, Neelabh Singh & Frederick Auer.

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
Or just:
```
pip install -r requirements.txt
```

### How to run the code
There are 4 paramaters to set, descriptions as below:
```
options:
  -h, --help            show this help message and exit
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Set the learning rate.
  -it ITERATIONS, --iterations ITERATIONS
                        Set the max iteration number.
  -d {1,2}, --degree {1,2}
                        Set the degree for the polynomial model.
  -ct CONVERGENCE_THRESHOLD, --convergence_threshold CONVERGENCE_THRESHOLD
                        Set the convergence threshold (optional). Convergence Threshold: determines when an iterative algorithm should stop, if changes in the
                        objective function become negligible according to the threshold value.
```