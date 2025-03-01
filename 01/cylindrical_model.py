import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data definition (constant)
DATA = np.array([
    [2.00, 1.08, -0.83, -1.97, -1.31, 0.57],
    [0.00, 1.68, 1.82, 0.28, -1.51, -1.91],
    [1.00, 2.38, 2.49, 2.15, 2.59, 4.32]
])
DATA_T = DATA.T
TIME = np.arange(1, len(DATA_T) + 1)


# Function to compute predictions based on parameters
def predict(t, r, omega, c):
    x = r * np.cos(omega * t)
    y = r * np.sin(omega * t)
    z = c * t
    return np.stack([x, y, z], axis=1)


# Gradient Descent Optimization
def gradient_descent(t, data, lr=0.01, epochs=1000):
    r, omega, c = 1.0, 1.0, 1.0  # Initial guesses
    n = len(t)

    for _ in range(epochs):
        pred = predict(t, r, omega, c)
        error = pred - data
        loss = np.sum(error ** 2)

        # Compute gradients
        dr = np.mean(2 * (pred[:, 0] - data[:, 0]) * np.cos(omega * t) +
                     2 * (pred[:, 1] - data[:, 1]) * np.sin(omega * t))
        domega = np.mean(2 * r * (-t * np.sin(omega * t) * (pred[:, 0] - data[:, 0]) +
                                  t * np.cos(omega * t) * (pred[:, 1] - data[:, 1])))
        dc = np.mean(2 * (pred[:, 2] - data[:, 2]) * t)

        # Update parameters
        r -= lr * dr
        omega -= lr * domega
        c -= lr * dc

        if _ % 100 == 0:
            print(f"Epoch {_}: Loss = {loss:.6f}, r = {r:.3f}, omega = {omega:.3f}, c = {c:.3f}")

    return r, omega, c


# Function to plot actual vs predicted data
def plot_trajectory(t, data, r, omega, c):
    predicted = predict(t, r, omega, c)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(data[:, 0], data[:, 1], data[:, 2], 'o-', label="Actual Data")
    ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], 'o-', label="Fitted Trajectory")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()
    plt.savefig("cylindrical.png")


# Function to plot actual vs predicted data and a estimate position a t=7;
def plot_trajectory_witht7(t, data, r, omega, c):
    predicted = predict(t, r, omega, c)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(data[:, 0], data[:, 1], data[:, 2], c='royalblue', marker='o', linestyle='-', label="Drone Trajectory")

    ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], c='darkorange', linestyle='-', marker='o',
            label="Estimated Trajectory (t=1 to 6)")

    # Predict position at t=7
    t_future = np.array([7])
    predicted_future = predict(t_future, r, omega, c)[0]

    ax.plot([predicted[-1, 0], predicted_future[0]],
            [predicted[-1, 1], predicted_future[1]],
            [predicted[-1, 2], predicted_future[2]],
            c='red', linestyle='--', label="Prediction Line")

    # Plot t=7 position
    ax.scatter(predicted_future[0], predicted_future[1], predicted_future[2], c='red', marker='x', s=100,
               label="Predicted Position at t=7")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Drone Trajectory with Prediction at t=7")
    plt.show()
    plt.savefig("cylindrical.png")

    #predicted t=7
    print(f"The position (x, y, z) at t=7 is: ({predicted_future[0]:.3f}, {predicted_future[1]:.3f}, {predicted_future[2]:.3f})")


# Example usage
if __name__ == "__main__":
    r_opt, omega_opt, c_opt = gradient_descent(TIME, DATA_T)
    plot_trajectory(TIME, DATA_T, r_opt, omega_opt, c_opt)
    plot_trajectory_witht7(TIME, DATA_T, r_opt, omega_opt, c_opt)

