"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # Return (X, y) under one shared seeded row permutation
    
    n = X.shape[0]

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    return X[indices], y[indices]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # Slice already-shuffled data into contiguous train/val/test partitions...
    
    n = X.shape[0]
    lim1 = int(n * train_frac)
    lim2 = lim1 + int(n * val_frac)

    return X[:lim1], y[:lim1], X[lim1:lim2], y[lim1:lim2], X[lim2:], y[lim2:]

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # Compute per-feature mean and std; replace std of 0 with 1
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1

    return mean, std

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # Apply z-score normalization using precomputed training mean and std.
    return (X - mean[None, :]) / std[None, :]

# Step 5 - add_bias_column
def add_bias_column(X):
    # Prepend a column of ones to feature matrix X
    return np.column_stack((np.ones(X.shape[0]), X))

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # Standardize features then add the bias column to form the design matrix.
    return add_bias_column(standardize_features(X, mean, std))

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # Return the predicted target vector from X and weights
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # Return the average of squared residuals as a scalar float.
    return np.mean((y_true - y_pred) ** 2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    return (X.T @ (y_pred - y_true)) / (0.5 * X.shape[0])

# Step 10 - normal_equation
def normal_equation(X, y):
    # Solve for the closed-form least-squares weights via the normal equation.
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    # Return (n_features,) weights sampled from N(0, 0.01)
    if seed is not None:
        np.random.seed(seed)

    return np.random.normal(loc=0, scale=0.01, size=n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # return the updated weight vector after one MSE gradient step
    return weights - lr * mse_gradient(X, y, predict_linear(X, weights))

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    # return the pair (train_loss, val_loss) as MSE floats
    return mse_loss(y_train, predict_linear(X_train, weights)), mse_loss(y_val, predict_linear(X_val, weights))

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    # Update best weights and patience counter; signal stop when val loss stalls...
    stop = False
    wait += 1

    if val_loss < best_val_loss:
        wait = 0
        best_val_loss = val_loss
        best_weights = weights
    elif wait >= patience:
        stop = True
    
    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    # Build the initial training-state dictionary for the GD epoch loop.
    weights = initialize_weights(n_features, seed)

    return {
        'best_val_loss': np.inf,
        'wait': 0,
        'train_losses': [],
        'val_losses': [],
        'stopped': False,
        'weights': weights,
        'best_weights': weights.copy()
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    # Take one GD step
    state['weights'] = gd_step(X_train, y_train, state['weights'], lr)

    # log train/val losses
    train_loss, val_loss = epoch_train_val_losses(X_train, y_train, X_val, y_val, state['weights'])
    state['train_losses'].append(train_loss)
    state['val_losses'].append(val_loss)

    # refresh early-stopping fields...
    state['best_val_loss'], state['wait'], state['best_weights'], state['stopped'] = update_early_stop_state(
        val_loss, 
        state['best_val_loss'], 
        state['wait'], 
        state['weights'], 
        state['best_weights'], 
        patience)

    return state

# Step 17 - train_batch_gd
def train_batch_gd(X_train, y_train, X_val, y_val, lr, epochs, patience, seed=None):
    # Train weights with full-batch GD for up to epochs, with early stopping.
    training_state = init_training_state(X_train.shape[1], seed)

    for _ in range(epochs):
        training_state = run_one_epoch(training_state, X_train, y_train, X_val, y_val, lr, patience)
        if training_state['stopped']:
            break

    return training_state['best_weights'], training_state['train_losses'], training_state['val_losses']

# Step 18 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # Compute the mean absolute error between true targets and predictions
    return np.mean(np.abs(y_true - y_pred))

# Step 19 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    # Return the root mean squared error between y_true and y_pred.
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

