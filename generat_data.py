import numpy as np
import os

def generate_trajectory(x0, A, len_time):
    """Generate a single trajectory given an initial condition and system matrix A."""
    trajectory = [x0]
    for _ in range(1, len_time):
        x_next = A @ trajectory[-1]
        trajectory.append(x_next)
    return np.array(trajectory)

def generate_data(num_trajectories, len_time, n):
    """
    Generate synthetic data for a discrete spectrum example.
    
    Arguments:
        num_trajectories: Number of trajectories to generate.
        len_time: Number of time steps per trajectory.
        n: Dimension of the state.
    
    Returns:
        data: A 2D numpy array where trajectories are stacked vertically.
    """
    # For demonstration, use a fixed stable system.
    # You can replace this with a more appropriate model.
    A = np.array([[0.95, 0.05],
                  [-0.05, 0.95]])
    
    data = []
    for _ in range(num_trajectories):
        # Initialize the state randomly
        x0 = np.random.randn(n)
        traj = generate_trajectory(x0, A, len_time)
        data.append(traj)
    # Stack trajectories along the time axis (each trajectory is a block of len_time rows)
    return np.vstack(data)

if __name__ == '__main__':
    # Parameters
    len_time = 51       # Each trajectory has 51 time steps.
    n = 2               # Dimension of the state.
    num_trajectories_train = 1000  # Number of training trajectories
    num_trajectories_val = 200     # Number of validation trajectories

    # Create the data directory if it doesn't exist.
    os.makedirs('data', exist_ok=True)

    # Generate and save training data
    train_data = generate_data(num_trajectories_train, len_time, n)
    np.savetxt('./data/DiscreteSpectrumExample_train_x.csv', train_data, delimiter=',')

    # Generate and save validation data
    val_data = generate_data(num_trajectories_val, len_time, n)
    np.savetxt('./data/DiscreteSpectrumExample_val_x.csv', val_data, delimiter=',')

    print("Data files generated:")
    print("  ./data/DiscreteSpectrumExample_train_x.csv")
    print("  ./data/DiscreteSpectrumExample_val_x.csv")
