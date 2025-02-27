import copy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

import training
import helperfns
import networkarch as net

# ------------------------------
# Set up parameters (as before)
# ------------------------------
params = {}

# settings related to dataset
params['data_name'] = 'DiscreteSpectrumExample'
params['data_train_len'] = 1
params['len_time'] = 51
n = 2  # dimension of system (and input layer)
num_initial_conditions = 5000  # per training file
params['delta_t'] = 0.02

# settings related to saving results
params['folder_name'] = 'exp1_best'

# settings related to network architecture
params['num_real'] = 2
params['num_complex_pairs'] = 0
params['num_evals'] = 2
k = params['num_evals']  # dimension of y-coordinates
w = 30
params['widths'] = [2, w, w, k, k, w, w, 2]
wo = 10
params['hidden_widths_omega'] = [wo, wo, wo]

# settings related to loss function
params['num_shifts'] = 30
params['num_shifts_middle'] = params['len_time'] - 1
max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
params['recon_lam'] = .1
params['Linf_lam'] = 10 ** (-7)
params['L1_lam'] = 0.0
params['L2_lam'] = 10 ** (-15)
params['auto_first'] = 0

# settings related to the training
params['num_passes_per_file'] = 15 * 6 * 10
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)
params['batch_size'] = 256
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']

# settings related to the timing
params['max_time'] = 4 * 60 * 60  # 4 hours
params['min_5min'] = .5
params['min_20min'] = .0004
params['min_40min'] = .00008
params['min_1hr'] = .00003
params['min_2hr'] = .00001
params['min_3hr'] = .000006
params['min_halfway'] = .000006

# ------------------------------
# Run experiments (training phase)
# ------------------------------
for count in range(200):  # Each experiment saves the best model if validation error improves.
    training.main_exp(copy.deepcopy(params))

# ---------------------------------------------------
# Inference: Plot sample trajectories of the best model
# ---------------------------------------------------
# Here we rebuild the inference graph, restore the best model, and run a single feed-forward pass
# on stacked validation data. The stacked input is built so that:
#   - The first time step is the true (or autoencoded) initial condition.
#   - Time steps 1 to 30 are the model’s predictions.
# We then plot the trajectories for a few randomly chosen samples,
# annotating the initial condition (time 0) as "Input" and the remaining points as "Predicted".

tf.reset_default_graph()
x, y, g_list, weights, biases = net.create_koopman_net(params)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, params['model_path'])

# We want to plot num_shifts+1 time steps (here, 31 = 1 initial + 30 predictions)
num_time_steps = params['num_shifts'] + 1

# Load validation data and stack it as during training.
data_val = np.loadtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',', dtype=np.float64)
data_val_tensor = helperfns.stack_data(data_val, num_time_steps - 1, params['len_time'])
# data_val_tensor has shape [num_time_steps, num_examples, n]

# Run the network to obtain outputs.
predicted_outputs = sess.run(y, feed_dict={x: data_val_tensor})
# Combine predictions into a single array of shape [num_time_steps, num_examples, n]
predicted_traj = np.stack([predicted_outputs[i] for i in range(num_time_steps)], axis=0)

# For comparison, use the stacked validation data as the “true” trajectory.
true_traj = data_val_tensor[:num_time_steps, :, :]

# Randomly select a few samples to plot.
num_samples = 3
sample_indices = random.sample(range(true_traj.shape[1]), num_samples)

# Create subplots: one row per state variable, one column per sample.
fig, axs = plt.subplots(2, num_samples, figsize=(18, 8), sharex=True, sharey=True)
time_steps = range(num_time_steps)

for i, idx in enumerate(sample_indices):
    # Extract trajectories for sample idx.
    true_sample = true_traj[:, idx, :]
    pred_sample = predicted_traj[:, idx, :]
    
    # Plot for state variable 1 (x1)
    axs[0, i].plot(time_steps, true_sample[:, 0], 'k--', label='True')
    axs[0, i].plot(time_steps, pred_sample[:, 0], 'bo-', label='Predicted')
    axs[0, i].set_title(f"Sample {idx} - x1")
    axs[0, i].set_xlabel("Time step")
    axs[0, i].set_ylabel("x1")
    axs[0, i].legend()
    # Annotate that time 0 is the input.
    axs[0, i].annotate('Input', xy=(0, true_sample[0, 0]), xytext=(1, true_sample[0, 0]),
                       arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Plot for state variable 2 (x2)
    axs[1, i].plot(time_steps, true_sample[:, 1], 'k--', label='True')
    axs[1, i].plot(time_steps, pred_sample[:, 1], 'ro-', label='Predicted')
    axs[1, i].set_title(f"Sample {idx} - x2")
    axs[1, i].set_xlabel("Time step")
    axs[1, i].set_ylabel("x2")
    axs[1, i].legend()
    axs[1, i].annotate('Input', xy=(0, true_sample[0, 1]), xytext=(1, true_sample[0, 1]),
                       arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()

sess.close()
