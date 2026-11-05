# ann_backprop

A bare-bones neural network trained with backpropagation, written from scratch in NumPy.
It accompanies the *AI Fundamentals: ANNs & Backpropagation* talk.

The network learns a noisy sine wave. Every step (forward pass, backward pass and
gradient-descent update) is written out by hand so you can read it and experiment with it.
This is a learning toy, not a template for real projects. Frameworks such as PyTorch and
TensorFlow/Keras handle gradients for you.

## Requirements

- Python 3.8+
- numpy
- matplotlib

## Files

| File | Purpose |
|---|---|
| `main.py` | Settings, data generation, network, forward/backward pass, training loop |
| `plotting_functions.py` | Saves the plots |

## Usage

1. Edit the `USER SETTINGS` block at the top of `main.py` (hidden layers, activation,
   learning rate, epochs, noise, number of points, seed, and so on).
2. Run `python main.py`.
3. Open the plots in the `output/` folder:
   - `loss_curves.png`: training and validation loss per epoch
   - `predictions.png`: the network's output at the start, 1/3, 2/3 and end of training;
     shaded regions are outside the training data

## Things to try

- Change `HIDDEN_LAYERS`, e.g. `[2]`, `[16, 16]`, `[128, 128]`
- Compare `"sigmoid"`, `"relu"` and `"step"` (step cannot learn; see the warning)
- Raise `LEARNING_RATE` until training diverges
- Fewer points plus more noise plus a big network gives overfitting
- Look at the shaded regions: what does the network predict outside its training data?