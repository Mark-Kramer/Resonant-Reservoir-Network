import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def plot_example_simulated_traces(X_train, y_train):

    y_train = y_train.ravel()

    # Define the vertical offset for stacking
    offset = 5
    
    # Define colors for each column: first red, second orange, third blue, fourth green
    colors = ['blue', 'orange', 'red', 'green']
    
    # Define the column names corresponding to the four label types
    column_names = ["SR", "Sp", "rp", "bk"]
    
    # Create a figure with 4 subplots arranged in one row
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    # Loop over each column name and plot 3 randomly chosen samples per label
    for i, col in enumerate(column_names):
        # Find indices of samples with the current label (assumes y_train contains these string labels)
        indices = np.where(y_train == col)[0]
    
        # Randomly choose 3 indices without replacement
        chosen_indices = np.random.choice(indices, size=3, replace=False)
        
        # Select the corresponding subplot
        ax = axes[i]
        x_axis = np.arange(X_train.shape[1])
        
        # Plot each chosen sample with a vertical offset using the specified color
        for j, idx in enumerate(chosen_indices):
            trace = X_train[idx, :]
            ax.plot(x_axis, trace + j * offset, color=colors[i])
        
        # Remove all axes: disable ticks and hide spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Set the y-axis limits
        ax.set_ylim([-4, 14])
        
        # Set subplot title as the column name
        #ax.set_title(col)
        
        # In the first subplot, add a horizontal thick black line at y = -2 from x = 100 to 200
        if i == 0:
            ax.hlines(y=-2, xmin=0, xmax=100, colors='black', linewidth=3)
    
    plt.tight_layout()
    plt.show()

    return fig

def plot_analytic_spectrum(w_t_minus_1, w_t_minus_2, node_step=25):
    """
    Plots a subset of analytic spectra starting from approximately 5 Hz with a user-defined step size.

    Parameters:
        w_t_minus_1 (array-like): Coefficients for w[t-1].
        w_t_minus_2 (array-like): Coefficients for w[t-2].
        node_step (int): Step between indices to select curves to plot.

    Fixed parameters:
        sigma = 1
        N     = 1000
        dt    = 0.001
        T     = N * dt

    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Fixed parameters
    sigma = 1
    N = 1000
    dt = 0.001
    j = np.arange(0, N//2 + 1)
    T = N * dt

    # Create figure and axes
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})

    # Frequency vector for selecting start frequency ~5 Hz
    freq_vector = j / T
    start_idx = np.where(freq_vector >= 5)[0][0]

    # Select indices based on node_step
    indices = np.arange(start_idx, len(w_t_minus_1), node_step)

    # Define a list of colors excluding 'red', 'orange', 'blue', 'green'
    colors = ['purple', 'brown', 'magenta', 'cyan', 'olive', 'pink', 'gray', 'navy', 'lime', 'teal']

    # Loop over the selected subset and plot
    for idx, k in enumerate(indices):
        denom = (
            1 
            + w_t_minus_1[k]**2 
            + w_t_minus_2[k]**2 
            - 2 * w_t_minus_1[k] * np.cos(2 * np.pi * j / N)
            - 2 * w_t_minus_2[k] * np.cos(2 * np.pi * 2 * j / N)
            + 2 * w_t_minus_1[k] * w_t_minus_2[k] * np.cos(2 * np.pi * j / N)
        )
        S = sigma**2 * dt / denom
        color = colors[idx % len(colors)]
        ax.semilogy(freq_vector, S, linewidth=2, color=color)

    # Set plot limits and labels
    ax.set_xlim([5, 300])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Analytic Spectrum [a.u.]')
    ax.grid(True)

    return fig

def plot_state_dynamics(states, node_step, plot_spacing, frange):
    """
    Plots the state dynamics by plotting each column (sampled with step node_step) 
    of the bottom half of 'states'. Each trace is shifted vertically by plot_spacing.

    Uses a predefined color scheme excluding 'red', 'orange', 'blue', and 'green'.

    Parameters:
      states      : 2D numpy array containing the state dynamics.
      node_step   : Sampling step for the columns of states.
      plot_spacing: Vertical spacing multiplier for each trace.
      frange      : Array-like values used to label the y-ticks.

    Returns:
      fig : matplotlib.figure.Figure object containing the plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define colors consistent with analytic spectrum plot
    colors = ['purple', 'brown', 'magenta', 'cyan', 'olive', 'pink', 'gray', 'navy', 'lime', 'teal']

    # Create a new figure and axes
    fig, ax = plt.subplots()

    ytick_positions = np.zeros(10)
    ytick_labels = np.zeros(10)

    for k in np.arange(10):
        # Extract the bottom half of the column (from row 500 onward)
        this_state = states[500:, k * node_step]
        # Center the trace and shift vertically by k * plot_spacing
        this_state = this_state - np.mean(this_state) + k * plot_spacing
        color = colors[k % len(colors)]
        ax.plot(this_state, color=color)
        ytick_positions[k] = k * plot_spacing
        ytick_labels[k] = frange[k * node_step]

    # Set y-ticks at positions and label them with integer versions of ytick_labels
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([int(label) for label in ytick_labels])

    # Draw a gray horizontal dashed line at each y-tick
    for y in ytick_positions:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.5)

    # Plot a time scale bar: a thick black horizontal line at y = -0.2 from x=0 to x=100
    ax.hlines(y=-0.2, xmin=0, xmax=100, colors='black', linewidth=3)

    # Remove x-tick labels and the bounding box on the top, bottom, and right axes
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Node resonant frequency [Hz]')

    return fig



def plot_scaled_reservoir_responses_by_label(X_test_features, y_test, reservoir, scaler):

    # Scale the features if a scaler is provided; otherwise, use the original features.
    if scaler != []:
        X_test_features_scaled = scaler.transform(X_test_features)
    else:
        X_test_features_scaled = np.log10(X_test_features)

    # Create figure for plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Identify all unique labels in y_test
    unique_labels = np.unique(y_test)
    
    for label in unique_labels:

        # Get features at the label
        label_features = X_test_features_scaled[y_test == label,:]
        
        # Compute mean and SEM
        mean_vals = np.mean(label_features, axis=0)
        std_vals  = np.std(label_features, axis=0, ddof=1)
        n_samples = label_features.shape[0]
        sem_vals  = std_vals / np.sqrt(n_samples)  # standard error of the mean
        # Compute the 5th and 95th percentiles for each reservoir frequency.
        #lower = np.percentile(label_features, 25, axis=0)
        #upper = np.percentile(label_features, 75, axis=0)
        
        # Plot the mean
        ax.plot(reservoir.frange, mean_vals, label=f'{label}')
        # Fill ± 2*SEM
        ax.fill_between(
            reservoir.frange,
            mean_vals - 2 * sem_vals,
            mean_vals + 2 * sem_vals,
            alpha=0.2
        )
    
    # Finalize the plot
    ax.set_xlabel("Reservoir frequency [Hz]", fontsize=16)
    ax.set_ylabel("Standardized Reservoir Response", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    return fig, ax