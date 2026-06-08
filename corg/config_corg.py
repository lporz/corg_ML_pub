"""
Configuration file for the Organic Carbon Prediction model.
Adjust these parameters to experiment with different model configurations.
"""

# Data Parameters
TRAIN_FILE = '../data/Corg_BS.csv'  # Path to training data
# TEST_FILE = '../data/Corg_BS_test_points_FPS.dat'  # Path to test data
TEST_FILE = '../data/Corg_BS_test_points_FPS_constr.dat'  # Path to test data
PREDICTION_FILE = '../data/BS_out_pnts_500m.csv'  # Path to predict data
RANDOM_SEED = 42  # Random seed for reproducibility
N_TEST_POINTS = 100  # Number of test points to sample

# Spatial Weighting Parameters (for addressing data imbalance)
USE_SPATIAL_WEIGHTING = True  # Whether to use spatial weighting to address data imbalance
SPATIAL_WEIGHT_METHOD = 'density'  # Method for calculating spatial weights: 'density', 'distance', or 'kde'
SPATIAL_WEIGHT_RADIUS = 0.25  # Radius (in degrees) for density calculation
SPATIAL_WEIGHT_MIN_WEIGHT = 0.1  # Minimum weight to prevent over-weighting sparse regions
SPATIAL_WEIGHT_MAX_WEIGHT = 3.0  # Maximum weight to prevent under-weighting dense regions
SPATIAL_WEIGHT_POWER = 1.0  # Power to apply to weights (1.0 = linear, 0.5 = square root, 2.0 = squared)

# Model Architecture Parameters
HIDDEN_LAYERS = [128, 64, 32]  # Number of neurons in each hidden layer
DROPOUT_RATE = 0.1  # Probability of dropping neurons during training (0.0 to 1.0)
                    # Higher values (e.g., 0.3-0.5) can help prevent overfitting
                    # Lower values (e.g., 0.1-0.2) are better for smaller datasets

# Training Parameters
BATCH_SIZE = 32  # Number of samples per training batch
                 # Smaller values (16-32) are good for smaller datasets
                 # Larger values (64-128) can speed up training for larger datasets

LEARNING_RATE = 0.002  # Initial learning rate
                       # Typical values: 0.1, 0.01, 0.001, 0.0001
                       # If training is unstable, try a smaller value
                       # If training is too slow, try a larger value

MAX_EPOCHS = 220  # Maximum number of training epochs
                  # The model will stop earlier if early stopping is triggered

EARLY_STOPPING_PATIENCE = 20  # Number of epochs to wait before early stopping
                             # If validation loss doesn't improve for this many epochs,
                             # training will stop
                             # Increase this if the model is stopping too early
                             # Decrease this if the model is overfitting

MIN_LEARNING_RATE = 1e-6  # Minimum learning rate for the scheduler
                          # The learning rate won't go below this value

# Learning Rate Scheduler Parameters
LR_SCHEDULER_FACTOR = 0.5  # Factor to reduce learning rate by when plateauing
                           # e.g., 0.5 means halve the learning rate
LR_SCHEDULER_PATIENCE = 10  # Number of epochs to wait before reducing learning rate

# Device Configuration
USE_CUDA = True  # Set to True to use GPU if available, False to use CPU

# Output Configuration
SAVE_MODEL = True  # Whether to save the trained model
MODEL_SAVE_PATH = 'models/corg_model.pth'  # Path to save the trained model
SAVE_PLOTS = True  # Whether to save the performance plots
PLOTS_SAVE_PATH = 'plots/corg_model_performance.png'  # Path to save the plots 

SCALER_PATH = 'misc/corg_scaler.pkl'
OUTPUT_FILE = 'outputs/corg_BS_out_pnts_500m_predictions_paramopt.csv'
OUTPUT_PLOT = 'plots/corg_prediction.png'
CONFIDENCE_PLOT = 'plots/corg_confidence_map.png'
