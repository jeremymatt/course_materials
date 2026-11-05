"""
A bare-bones neural network, trained with backpropagation, written from scratch
with nothing but NumPy.

The network learns to approximate a noisy sine wave. It is a toy for learning:
every step of the forward pass, the backward pass (backpropagation) and the
gradient-descent update is written out by hand so you can read it, change it and
see what happens. Real projects use frameworks such as PyTorch or
TensorFlow/Keras, which compute gradients automatically.

How to use:
    1. Edit the settings in the "USER SETTINGS" block below.
    2. Run:  python main.py
    3. Look at the plots saved in the output folder.

Conventions used throughout:
    * Data are stored as column arrays with one row per sample, shape (N, 1).
    * A layer computes  z = a_prev @ W + b,  then  a = activation(z).
      W has shape (inputs to the layer, neurons in the layer). This is the
      transpose of the "W x" notation on the slides; it lets us push a whole
      batch of samples through a layer with one matrix multiply.
    * The output layer is linear (no activation) because sin(x) can be
      negative, and sigmoid or ReLU outputs cannot reach negative values.
"""

import os
import numpy as np

import plotting_functions as pf

# =============================================================================
# USER SETTINGS -- change these and re-run
# =============================================================================
HIDDEN_LAYERS = [16,16]   # neurons in each hidden layer; list length = number of hidden layers
ACTIVATION = "relu"     # hidden-layer activation: "step", "sigmoid", or "relu"
LEARNING_RATE = 0.05       # gradient-descent step size (eta)
NUM_EPOCHS = 30000         # passes through the training data
BATCH_SIZE = None          # samples per gradient step; None = use all training data (full batch)

RANDOM_SEED = 42           # same seed -> same data, same starting weights, same results
NUM_POINTS = 60            # total number of noisy data points to generate
NOISE_STD = 0.15           # standard deviation of the noise added to sin(x); 0 = no noise
TRAIN_FRACTION = 0.8       # fraction of points used for training; the rest are for validation
NUM_PERIODS = 1.0          # data cover x from 0 to 2*pi*NUM_PERIODS

EXTRAPOLATION_PERIODS = 0.5  # also predict this many sine periods beyond each end of the
                             # training range, to show how the network behaves outside its data
NUM_PREDICTION_POINTS = 200  # number of evenly spaced x values used to draw the network's output
PRINT_EVERY = 2000           # print the losses every this many epochs
OUTPUT_DIR = "output"        # folder where the plots are saved


# =============================================================================
# 1. DATA
# =============================================================================
def generate_data(rng):
    """Sample x values at random, compute y = sin(x) + noise, and split the
    points at random into a training set and a validation set."""
    x_max = 2 * np.pi * NUM_PERIODS
    x = rng.uniform(0, x_max, size=(NUM_POINTS, 1))
    y = np.sin(x) + rng.normal(0, NOISE_STD, size=(NUM_POINTS, 1))

    # Shuffle the indices, then use the first part for training, the rest for validation.
    order = rng.permutation(NUM_POINTS)
    n_train = int(round(TRAIN_FRACTION * NUM_POINTS))
    train_idx, val_idx = order[:n_train], order[n_train:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_scaler(x_train):
    """Return a function that rescales x so the TRAINING data span [0, 1].

    Inputs on a small, consistent scale keep the weighted sums in a sensible
    range, so gradient descent behaves well. The same scaling (computed from
    the training data only) is applied to validation and prediction inputs.
    Inputs outside the training range land outside [0, 1].
    """
    x_min, x_max = x_train.min(), x_train.max()
    return lambda x: (x - x_min) / (x_max - x_min)


# =============================================================================
# 2. ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
# =============================================================================
# Each entry: (function, derivative). Backprop needs the derivative.
def step(z):
    return (z >= 0).astype(float)

def step_derivative(z):
    # The step is flat everywhere except at the jump, so its derivative is 0.
    # Backprop therefore sends back all-zero gradients: the hidden layers never learn.
    return np.zeros_like(z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

ACTIVATIONS = {
    "step": (step, step_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
}


# =============================================================================
# 3. NETWORK SET-UP
# =============================================================================
def init_network(layer_sizes, activation, rng):
    """Create random weights and zero biases for every layer.

    layer_sizes looks like [1, 16, 16, 1]: one input, the hidden layers, one output.
    The spread of the random weights is chosen so signals neither shrink nor
    blow up as they pass through the layers:
        ReLU:            He initialization,     std = sqrt(2 / n_in)
        sigmoid / step:  Xavier initialization,  std = sqrt(2 / (n_in + n_out))
    """
    weights, biases = [], []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        if activation == "relu":
            std = np.sqrt(2.0 / n_in)
        else:
            std = np.sqrt(2.0 / (n_in + n_out))
        weights.append(rng.normal(0, std, size=(n_in, n_out)))
        biases.append(np.zeros((1, n_out)))
    return weights, biases


# =============================================================================
# 4. FORWARD PASS
# =============================================================================
def forward(x, weights, biases, act):
    """Push inputs x (shape (N, 1)) through the network.

    Returns the prediction plus the lists of pre-activations (zs) and
    activations (activations) for every layer. Backprop reuses these, which is
    why we keep them instead of throwing them away.
    """
    activations = [x]  # activations[0] is the input itself
    zs = []
    n_layers = len(weights)
    for layer in range(n_layers):
        z = activations[-1] @ weights[layer] + biases[layer]   # weighted sum + bias
        zs.append(z)
        if layer < n_layers - 1:
            a = act(z)        # hidden layer: apply the activation function
        else:
            a = z             # output layer: linear, so outputs can be any real number
        activations.append(a)
    return activations[-1], zs, activations


def mse_loss(y_pred, y_true):
    """Mean squared error: the average of (prediction - target)^2."""
    return np.mean((y_pred - y_true) ** 2)


# =============================================================================
# 5. BACKWARD PASS (BACKPROPAGATION)
# =============================================================================
def backward(y_pred, y_true, zs, activations, weights, act_derivative):
    """Compute dLoss/dW and dLoss/db for every layer with the chain rule.

    delta = dLoss/dz for the current layer (one value per sample per neuron).
    We start at the output and walk backward, layer by layer:
        output layer:  delta = dLoss/dy_pred            (output is linear)
        weight grad:   dW = a_prev^T @ delta             (slides: eq. 3)
        bias grad:     db = sum of delta over samples    (slides: eq. 4)
        previous layer: delta = (delta @ W^T) * act'(z)  (slides: eq. 2)
    """
    n_samples = y_true.shape[0]
    n_layers = len(weights)
    grad_w = [None] * n_layers
    grad_b = [None] * n_layers

    # Derivative of the mean squared error with respect to the prediction.
    delta = 2.0 * (y_pred - y_true) / n_samples

    for layer in reversed(range(n_layers)):
        a_prev = activations[layer]                       # input that this layer received
        grad_w[layer] = a_prev.T @ delta
        grad_b[layer] = delta.sum(axis=0, keepdims=True)
        if layer > 0:
            # Send the error back through this layer's weights, then through
            # the previous layer's activation function.
            delta = (delta @ weights[layer].T) * act_derivative(zs[layer - 1])
    return grad_w, grad_b


# =============================================================================
# 6. TRAINING LOOP
# =============================================================================
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if ACTIVATION not in ACTIVATIONS:
        raise ValueError(f"ACTIVATION must be one of {list(ACTIVATIONS)}")
    act, act_derivative = ACTIVATIONS[ACTIVATION]

    warning = None
    if ACTIVATION == "step":
        warning = ("WARNING: the step function has zero gradient, so backprop cannot "
                   "train the hidden layers.\nOnly the output layer's weights change.")
        print(warning)

    # --- data ---
    x_train, y_train, x_val, y_val = generate_data(rng)
    scale = make_scaler(x_train)
    xs_train, xs_val = scale(x_train), scale(x_val)

    # Evenly spaced x values (inside AND beyond the training range) used to draw
    # the network's output between and outside the data points.
    period = 2 * np.pi
    x_plot = np.linspace(-EXTRAPOLATION_PERIODS * period,
                         (NUM_PERIODS + EXTRAPOLATION_PERIODS) * period,
                         NUM_PREDICTION_POINTS).reshape(-1, 1)
    xs_plot = scale(x_plot)

    # --- network ---
    layer_sizes = [1] + list(HIDDEN_LAYERS) + [1]
    weights, biases = init_network(layer_sizes, ACTIVATION, rng)
    n_params = sum(w.size + b.size for w, b in zip(weights, biases))
    print(f"Network layers {layer_sizes}: {n_params} trainable parameters, "
          f"{len(x_train)} training points")

    # Save the network's output at the start, 1/3, 2/3 and the end of training.
    snapshot_epochs = [0, NUM_EPOCHS // 3, 2 * NUM_EPOCHS // 3, NUM_EPOCHS]
    snapshots = {}
    train_losses, val_losses = [], []
    batch = BATCH_SIZE or len(x_train)

    for epoch in range(NUM_EPOCHS + 1):
        if epoch in snapshot_epochs:
            snapshots[epoch] = forward(xs_plot, weights, biases, act)[0]

        # Record the losses for this epoch (before this epoch's updates).
        train_losses.append(mse_loss(forward(xs_train, weights, biases, act)[0], y_train))
        val_losses.append(mse_loss(forward(xs_val, weights, biases, act)[0], y_val))
        if epoch % PRINT_EVERY == 0:
            print(f"epoch {epoch:6d}   train loss {train_losses[-1]:.4f}   val loss {val_losses[-1]:.4f}")
        if epoch == NUM_EPOCHS:
            break

        # Visit the training points in a new random order each epoch, a batch at a time.
        order = rng.permutation(len(x_train))
        for start in range(0, len(x_train), batch):
            idx = order[start:start + batch]
            y_pred, zs, activations = forward(xs_train[idx], weights, biases, act)   # forward pass
            grad_w, grad_b = backward(y_pred, y_train[idx], zs, activations,        # backward pass
                                      weights, act_derivative)
            for layer in range(len(weights)):                                       # gradient descent
                weights[layer] -= LEARNING_RATE * grad_w[layer]
                biases[layer] -= LEARNING_RATE * grad_b[layer]

    # --- plots ---
    pf.plot_losses(train_losses, val_losses, os.path.join(OUTPUT_DIR, "loss_curves.png"), warning)
    pf.plot_snapshots(snapshots, x_plot, x_train, y_train, x_val, y_val,
                      os.path.join(OUTPUT_DIR, "predictions.png"), warning,
                      title=f"Hidden layers {list(HIDDEN_LAYERS)}, {ACTIVATION}, lr={LEARNING_RATE}")
    print(f"Plots saved in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
