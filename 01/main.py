import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


# Data definition (constant)
DATA = np.array([
    [ 2.00, 1.08, -0.83, -1.97, -1.31, 0.57 ],
    [ 0.00, 1.68, 1.82, 0.28, -1.51, -1.91 ],
    [ 1.00, 2.38, 2.49, 2.15, 2.59, 4.32 ]
])
DATA_T = DATA.T
TIME = np.arange(1, len(DATA_T)+1)

class Model:
    def __init__(self, degree: int):
        """Constructor.
        """
        assert degree > 0
        self.degree = degree

class Degree1(Model):
    def __init__(self):
        """A child class inherited from class Model.
        This models the drone's movement using degree 1 polynomial regression (constant velocity).
        """
        super().__init__(degree=1)
        self.A = np.array([
            [1, 1, 1],  # [alpha_0, beta_0, gamma_0]
            [1, 1, 1]   # [alpha_1, beta_1, gamma_1]
        ], dtype=float)
        self.error_arr = []
        self.coef_arr = []
        self.convergence_iter = None

    def __str__(self):
        pass

    def cal_grad(self, x: np.ndarray, y: np.ndarray):
        diff = self.A[0] + self.A[1] * x.reshape(-1, 1) - y  # Compute the term (A_0 + A_1*t - y)
        gradient = 2 * np.array([
            np.sum(diff, axis=0),       # First row: sum of (A_0 + A_1*t - y)
            np.sum(x.reshape(-1, 1) * diff, axis=0)  # Second row: sum of t * (A_0 + A_1*t - y)
        ])

        return gradient
        # return np.array([
        #     [
        #         sum([ 2*(self.A[0, 0] + self.A[1, 0]*t - p[0]) for t, p in zip(x, y) ]),
        #         sum([ 2*(self.A[0, 1] + self.A[1, 1]*t - p[1]) for t, p in zip(x, y) ]),
        #         sum([ 2*(self.A[0, 2] + self.A[1, 2]*t - p[2]) for t, p in zip(x, y) ])
        #     ],
        #     [
        #         sum([ 2*t*(self.A[0, 0] + self.A[1, 0]*t - p[0]) for t, p in zip(x, y) ]),
        #         sum([ 2*t*(self.A[0, 1] + self.A[1, 1]*t - p[1]) for t, p in zip(x, y) ]),
        #         sum([ 2*t*(self.A[0, 2] + self.A[1, 2]*t - p[2]) for t, p in zip(x, y) ])
        #     ]
        # ])

    def predict(self, t: float) -> np.ndarray:
        "This function will predict the next position of the drone at t=7."
        return self.A[0] + self.A[1] * t

    def fit(self, x: np.ndarray, y: np.ndarray, lr: int, iter: int, ct: float|None):
        assert len(x) == len(y)

        n = len(x)
        self.D = np.array([np.full(n, 1), np.arange(1, n+1) ], dtype=float).transpose()
        self.error = sum((y - np.dot(self.D, self.A))**2)
        i = 0
        while i < iter:
            self.gradient = self.cal_grad(x, y)
            self.A = self.A - lr * self.gradient
            self.coef_arr.append(self.A)

            # Compute error (squared error)
            y_pred = np.dot(self.D, self.A)
            error = np.sum((y - y_pred) ** 2)
            print(f"Residual error at iteration {i}: {error}")
            self.error_arr.append(error)

            # Check for convergence
            if i > 0 and abs(self.error_arr[-1] - self.error_arr[-2]) < ct and self.convergence_iter is None:
                self.convergence_iter = i  # Store the convergence iteration
            i += 1

           
        # If no convergence detected based on threshold, mark last iteration as convergence
        if self.convergence_iter is None:
            self.convergence_iter = iter - 1
        
        

    def plot(self, convergence_threshold=None):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.error_arr) + 1), self.error_arr, marker='o', linestyle='-', color='b', label="Error")

        if hasattr(self, "convergence_iter") and self.convergence_iter is not None:
            plt.axvline(self.convergence_iter + 1, color='r', linestyle='--', label=f'Converged at {self.convergence_iter + 1}')

            # Extract parameter values
            param_values = self.coef_arr[self.convergence_iter]
            alpha0, beta0, gamma0 = param_values[0]
            alpha1, beta1, gamma1 = param_values[1]

            # Compute velocity (since degree 1 is constant velocity)
            velocity_x, velocity_y, velocity_z = alpha1, beta1, gamma1 

            # Get residual error at convergence iteration
            residual_error = self.error_arr[self.convergence_iter]

            # Prepare annotation text
            param_text = (f'Converged\nIter: {self.convergence_iter + 1}\n'
                        f'Residual Error: {residual_error:.6f}\n'
                        f'Params:\n'
                        f'$\\alpha_0$: {alpha0:.3f}, $\\alpha_1$: {alpha1:.3f}\n'
                        f'$\\beta_0$: {beta0:.3f}, $\\beta_1$: {beta1:.3f}\n'
                        f'$\\gamma_0$: {gamma0:.3f}, $\\gamma_1$: {gamma1:.3f}\n'
                        f'Velocity:\n'
                        f'$v_x$: {velocity_x:.3f}, $v_y$: {velocity_y:.3f}, $v_z$: {velocity_z:.3f}')

            # Set position for annotation text
            x_pos = max(self.convergence_iter - 5, 1)  
            y_pos = residual_error * 1.1  # Adjust vertical position of the text

            # Annotate with arrow
            plt.annotate(param_text, 
                        xy=(self.convergence_iter + 1, residual_error), 
                        xytext=(x_pos, y_pos), 
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
                        fontsize=10, color='red', 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.85),
                        ha='right')

        plt.xlabel('Iteration')
        plt.ylabel('Error (Squared Error)')
        plt.title('Error vs. Iterations')

        # Show convergence threshold if provided
        if convergence_threshold:
            plt.axhline(convergence_threshold, color='g', linestyle=':', label=f'Threshold {convergence_threshold}')
        
        plt.legend()
        plt.grid(True)
        plt.show()

class Degree2(Model):
    def __init__(self):
        """A child class inherited from class Model.
        This models the drone's movement using degree 2 polynomial regression (constant acceleration).
        """
        super().__init__(degree=2)
        self.A = np.array([
            [1, 1, 1],  # [alpha_0, beta_0, gamma_0]
            [1, 1, 1],  # [alpha_1, beta_1, gamma_1]
            [1, 1, 1]   # [alpha_2, beta_2, gamma_2]
        ], dtype=float)
        self.error_arr = []
        self.coef_arr = []
        self.convergence_iter = None

    def __str__(self):
        pass

    def cal_grad(self, x: np.ndarray, y: np.ndarray):
        diff = self.A[0] + self.A[1] * x.reshape(-1, 1) + self.A[2] * (x.reshape(-1, 1)**2) - y  # Compute the term (A_0 + A_1*t + A_2*t^2 - y)
        gradient = 2 * np.array([
            np.sum(diff, axis=0),                           # 1st row: sum of (A_0 + A_1*t + A_2*t^2 - y)
            np.sum(x.reshape(-1, 1) * diff, axis=0),        # 2nd row: sum of t * (A_0 + A_1*t + A_2*t^2 - y)
            np.sum((x.reshape(-1, 1)**2) * diff, axis=0)    # 3rd row: sum of t*t * (A_0 + A_1*t + A_2*t^2 - y)
        ])

        return gradient

    def predict(self, t: float) -> np.ndarray:
        "Part c: This function will predict the next position of the drone at t=7."
        return self.A[0] + self.A[1] * t + self.A[2] * (t**2)

    def fit(self, x: np.ndarray, y: np.ndarray, lr: int, iter: int, ct: float):
        assert len(x) == len(y)

        n = len(x)
        self.D = np.array([ np.full(n, 1), np.arange(1, n+1), (np.arange(1, n+1)**2)], dtype=float).transpose()
        self.error = sum((y - np.dot(self.D, self.A))**2)
        i = 0
        while i < iter:
            self.gradient = self.cal_grad(x, y)
            self.A = self.A - lr * self.gradient
            self.coef_arr.append(self.A)

            # Compute error (squared error)
            y_pred = np.dot(self.D, self.A)
            error = np.sum((y - y_pred) ** 2)
            print(f"Residual error at iteration {i}: {error}")
            self.error_arr.append(error)

            # Check for convergence
            if i > 0 and abs(self.error_arr[-1] - self.error_arr[-2]) < ct and self.convergence_iter is None:
                self.convergence_iter = i  # Store the convergence iteration
            i += 1

        print(sum(self.error_arr))
           
        # If no convergence detected based on threshold, mark last iteration as convergence
        if self.convergence_iter is None:
            self.convergence_iter = iter - 1

    def plot(self, convergence_threshold=None):
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.error_arr) + 1), self.error_arr, marker='o', linestyle='-', color='b', label="Error")

        if hasattr(self, "convergence_iter") and self.convergence_iter is not None:
            plt.axvline(self.convergence_iter + 1, color='r', linestyle='--', label=f'Converged at {self.convergence_iter+1}')

            # Extract parameter values
            param_values = self.coef_arr[self.convergence_iter]
            alpha0, beta0, gamma0 = param_values[0]
            alpha1, beta1, gamma1 = param_values[1]
            alpha2, beta2, gamma2 = param_values[2]

            # Compute acceleration
            acc_x, acc_y, acc_z = alpha2 * 2, beta2 * 2, gamma2 * 2

            # Get residual error at convergence iteration
            residual_error = self.error_arr[self.convergence_iter]

            # Prepare annotation text
            param_text = (f'Converged\nIter: {self.convergence_iter+1}\n'
                        f'Residual Error: {residual_error:.6f}\n'
                        f'Params:\n'
                        f'$\\alpha_0$: {alpha0:.3f}, $\\alpha_1$: {alpha1:.3f}, $\\alpha_2$: {alpha2:.3f}\n'
                        f'$\\beta_0$: {beta0:.3f}, $\\beta_1$: {beta1:.3f}, $\\beta_2$: {beta2:.3f}\n'
                        f'$\\gamma_0$: {gamma0:.3f}, $\\gamma_1$: {gamma1:.3f}, $\\gamma_2$: {gamma2:.3f}\n'
                        f'Acceleration:\n'
                        f'$a_x$: {acc_x:.3f}, $a_y$: {acc_y:.3f}, $a_z$: {acc_z:.3f}')

            x_pos = max(self.convergence_iter - 5, 1)  # Shift left but keep within bounds
            y_pos = residual_error * 1.15  # Position slightly above the point

            plt.annotate(param_text, 
                        xy=(self.convergence_iter + 1, residual_error),  # Point to red line
                        xytext=(x_pos, y_pos),  # Keep text left
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
                        fontsize=10, color='red', 
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.85),
                        ha='right')

        plt.xlabel('Iteration')
        plt.ylabel('Error (Squared Error)')
        plt.title('Error vs. Iterations')

        if convergence_threshold:
            plt.axhline(convergence_threshold, color='g', linestyle=':', label=f'Threshold {convergence_threshold}')
        
        plt.legend()
        plt.grid(True)
        plt.show()

def plot_trajectory_3d(time: np.ndarray, points: np.ndarray):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(x, y, z, marker='o', linestyle='-', label='Tracked Trajectory')

    # Annotate time points
    for i, t in enumerate(time.flatten()):  # Flatten time array to avoid shape mismatch
        ax.text(x[i], y[i], z[i], f'T={t}', fontsize=10, color='red')

    # Labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Tracked Trajectory over Time')
    ax.legend()

    # plt.show()
    # print(os.getcwd())
    plt.savefig(f"{os.getcwd()}/trajectory_3d_plot.png")
    print("file has been save to: trajectory_3d_plot.png")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Parse parameters in a POSIX-style.")

    # Add arguments
    parser.add_argument("-lr", "--learning_rate", type=float, help="Set the learning rate", required=True, default=0.01)
    parser.add_argument("-it", "--iterations", type=int, help="Set the max iteration number", required=True, default=20)
    parser.add_argument("-d", "--degree", type=int, help="Set the degree for the polynomial model", choices=[1, 2], required=True, default=1)
    parser.add_argument("-ct", "--convergence_threshold", type=float, help="Set the convergence threshold", default=1e-5)
    
    # Parse arguments
    args = parser.parse_args()

    # Use parsed arguments
    lr = args.learning_rate
    iter = args.iterations
    deg = args.degree
    ct = args.convergence_threshold
    print(f"Learning Rate: {lr}")
    print(f"Iterations: {iter}")
    print(f"Degree: {deg}")

    if deg == 1:
        model = Degree1()
    elif deg == 2:
        model = Degree2()


    plot_trajectory_3d(TIME, DATA_T)
    model.fit(TIME, DATA_T, lr, iter, ct)
    predicted_position = model.predict(7)
    print(f"Predicted position at t=7: {predicted_position}")
    model.plot()