import pickle
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from test_prepare import farthest_point_sampling
import hydra
from omegaconf import DictConfig
import logging
import os
from pathlib import Path

# Set up logging
def setup_logging(log_dir):
    """Set up logging to write to files in the Hydra sweep directory."""
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()  # Keep console output for debugging
        ]
    )
    return logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed, use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_spatial_weights(lon, lat, method='density', radius=0.25, min_weight=0.1, max_weight=5.0, power=1.0, logger=None):
    """
    Calculate spatial weights to address data imbalance.
    
    Args:
        lon, lat: Longitude and latitude arrays
        method: 'density', 'distance', or 'kde'
        radius: Radius for density calculation (in degrees)
        min_weight, max_weight: Bounds for the weights
        power: Power to apply to weights
        logger: Logger instance for output
    
    Returns:
        weights: Array of weights for each sample
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Calculating spatial weights using method: {method}")
    
    if method == 'density':
        # Calculate local density around each point
        coords = np.column_stack([lon, lat])
        weights = np.zeros(len(coords))
        
        for i, point in enumerate(coords):
            # Calculate distances to all other points
            distances = cdist([point], coords)[0]
            # Count points within radius
            density = np.sum(distances <= radius)
            weights[i] = density
        
        # Invert weights (higher density = lower weight)
        weights = 1.0 / (weights + 1)  # Add 1 to avoid division by zero
        
    elif method == 'distance':
        # Weight based on distance to nearest neighbors
        coords = np.column_stack([lon, lat])
        weights = np.zeros(len(coords))
        
        for i, point in enumerate(coords):
            # Calculate distances to all other points
            distances = cdist([point], coords)[0]
            # Remove self-distance
            distances = distances[distances > 0]
            # Use distance to k-th nearest neighbor
            k = min(5, len(distances))  # Use 5th nearest neighbor or all if less than 5
            kth_distance = np.partition(distances, k-1)[k-1]
            weights[i] = kth_distance
        
    elif method == 'kde':
        # Use Kernel Density Estimation
        coords = np.column_stack([lon, lat])
        kde = gaussian_kde(coords.T)
        densities = kde(coords.T)
        weights = 1.0 / (densities + 1e-10)  # Invert densities
    
    else:
        raise ValueError(f"Unknown spatial weight method: {method}")
    
    # Normalize weights to [min_weight, max_weight] range
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    weights = weights * (max_weight - min_weight) + min_weight
    
    # Apply power transformation
    weights = weights ** power
    
    logger.info(f"Weight statistics:")
    logger.info(f"  Min weight: {weights.min():.3f}")
    logger.info(f"  Max weight: {weights.max():.3f}")
    logger.info(f"  Mean weight: {weights.mean():.3f}")
    logger.info(f"  Std weight: {weights.std():.3f}")
    
    return weights

class CarbonPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_layers=None, dropout_rate=0.1):
        super(CarbonPredictor, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
            
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def prepare_data(cfg, logger=None):
    """
    Load and prepare the data for training and testing.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load the training data
    train_df = pd.read_csv(cfg.data.train_file)
    
    # Load the test points data
    test_df = pd.read_csv(cfg.data.test_file)
    

   
    # test_indices = farthest_point_sampling(test_df, cfg.data.n_test_points)
    # test_df = test_df.iloc[test_indices]
    
    # Rename columns in test data to match training data
    test_df = test_df.rename(columns={
        'X': 'lon',
        'Y': 'lat',
        'Z': 'Corg (%)'
    })
    
    # Log information about NaN values
    logger.info("Missing values in training data:")
    logger.info(train_df.isnull().sum().to_string())
    
    # First, look up the depth and distance values from the original training data
    test_df['depth (m)'] = train_df.iloc[test_df['ID'] - 2]['depth (m)'].values
    test_df['dist_coast (km)'] = train_df.iloc[test_df['ID'] - 2]['dist_coast (km)'].values
    
    # remove test data from train data
   # train_df = train_df.drop(index=[test_df['ID'] - 2])
   # train_df = train_df.drop(index=(test_df['ID'] - 2))    
    
    # Extract features and target
    X_train = train_df[['lon', 'lat', 'depth (m)', 'dist_coast (km)']].values
    y_train = train_df['Corg (%)'].values
    
    X_test = test_df[['lon', 'lat', 'depth (m)', 'dist_coast (km)']].values
    y_test = test_df['Corg (%)'].values
    
    # Calculate spatial weights for training data
    if cfg.spatial_weighting.use_spatial_weighting:
        spatial_weights = calculate_spatial_weights(
            train_df['lon'].values, 
            train_df['lat'].values,
            method=cfg.spatial_weighting.method,
            radius=cfg.spatial_weighting.radius,
            min_weight=cfg.spatial_weighting.min_weight,
            max_weight=cfg.spatial_weighting.max_weight,
            power=cfg.spatial_weighting.power,
            logger=logger
        )
    else:
        spatial_weights = np.ones(len(train_df))
    
    # Standardize the features using only training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Save scaler
    with open(cfg.output.scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return X_train, y_train, X_test, y_test, scaler, test_df, spatial_weights

def train_model(model, train_loader, criterion, optimizer, scheduler, device, model_name, cfg, logger=None):
    """
    Train a PyTorch model with early stopping.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Training Neural Network Model {model_name}...")
    
    # Initialize variables for early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(cfg.training.max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Learning rate scheduling
        scheduler.step(train_loss)
        
        # Early stopping check
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{cfg.training.max_epochs}, Train Loss: {train_loss:.4f}")
    
    # Load the best model
    model.load_state_dict(best_model_state)
    return model

def train_model_full(X_train, y_train, spatial_weights, device, cfg, logger=None):
    """
    Train the neural network model using all available data with spatial weighting.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    # Create weighted sampler if spatial weighting is enabled
    if cfg.spatial_weighting.use_spatial_weighting:
        # Convert weights to sampling weights (higher weight = higher chance of being sampled)
        sampling_weights = torch.FloatTensor(spatial_weights)
        sampler = WeightedRandomSampler(
            weights=sampling_weights,
            num_samples=len(X_train),
            replacement=True
        )
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), 
            batch_size=cfg.training.batch_size, 
            sampler=sampler
        )
        logger.info(f"Using spatial weighting with {len(spatial_weights)} samples")
    else:
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor), 
            batch_size=cfg.training.batch_size, 
            shuffle=True
        )
        logger.info("Using standard random sampling (no spatial weighting)")
    
    # Initialize model
    model = CarbonPredictor(
        input_dim=X_train.shape[1], 
        hidden_layers=cfg.model.hidden_layers,
        dropout_rate=cfg.model.dropout_rate
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=cfg.training.lr_scheduler_factor, 
        patience=cfg.training.lr_scheduler_patience, 
        min_lr=cfg.training.min_learning_rate
    )
    
    # Train model
    model = train_model(
        model, train_loader, 
        criterion, optimizer, scheduler, device, "Carbon Predictor", cfg, logger
    )
    
    return model

def evaluate_model(model, X_test, y_test, device, logger=None):
    """
    Evaluate the model on the test set.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    # Convert test data to PyTorch tensor
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - y_test)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))
    r2 = r2_score(y_test, predictions)
    
    logger.info("Model Evaluation:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R-squared: {r2:.4f}")
    
    return {
        'predictions': predictions,
        'metrics': {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R-squared': r2
        }
    }



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    import torch
    
    # Set up logging - use Hydra's output directory
    logger = setup_logging(hydra.utils.get_original_cwd())
    
    # Set random seed for reproducibility
    set_seed(cfg.seed, cfg.device.use_cuda)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device.use_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(cfg.output.model_save_path).parent
    misc_dir = Path(cfg.output.scaler_path).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    misc_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    X_train, y_train, X_test, y_test, scaler, test_df, spatial_weights = prepare_data(cfg, logger)
    
    # Load training data for visualization
    train_df = pd.read_csv(cfg.data.train_file)
    test_df = pd.read_csv(cfg.data.test_file)
    
    # remove test data from train data
    #train_df = train_df.drop(index=test_df['ID'] - 2)
    #train_df = train_df.drop(index=[1,2,3])
   
    # Train model
    model = train_model_full(X_train, y_train, spatial_weights, device, cfg, logger)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test, device, logger)
    
   
    # Save model if configured
    if cfg.output.save_model:
        torch.save(model.state_dict(), cfg.output.model_save_path)
        logger.info(f"Model saved to {cfg.output.model_save_path}")
    
    # Log final metrics
    logger.info("Final Model Metrics:")
    for metric, value in results['metrics'].items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 