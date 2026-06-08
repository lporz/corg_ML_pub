import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from corg_train import CarbonPredictor
import hydra
from omegaconf import DictConfig

def predict_corg(input_file, model_path, scaler_path, output_file, output_plot='predicted_corg_map.png', confidence_plot='corg_confidence_map.png', cfg=None):
    """
    Make predictions using the trained neural network model and create visualization.
    
    Args:
        input_file (str): Path to the input CSV file containing features
        model_path (str): Path to the saved neural network model
        scaler_path (str): Path to the saved StandardScaler
        output_file (str): Path to save the predictions
        output_plot (str): Path to save the visualization
        confidence_plot (str): Path to save the confidence visualization
        cfg: Configuration object
    """
    # Check if GPU is available
    use_cuda = cfg.device.use_cuda if cfg else True
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load the prediction data
    df = pd.read_csv(input_file)
    
    # Load the scaler from training
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load the trained model
    hidden_layers = cfg.model.hidden_layers if cfg else [128, 64, 32]
    dropout_rate = cfg.model.dropout_rate if cfg else 0.2
    model = CarbonPredictor(input_dim=4, hidden_layers=hidden_layers, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Extract features
    X = df[['lon', 'lat', 'depth (m)', 'dist_coast (km)']].values
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Convert to PyTorch tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    # Make predictions with Monte Carlo Dropout
    n_samples = 100
    predictions_list = []
    model.train()  # Enable dropout for uncertainty estimation
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(X_tensor).cpu().numpy().flatten()
            # Ensure predictions are non-negative
            pred = np.maximum(pred, 0)
            predictions_list.append(pred)
    
    # Calculate mean, standard deviation, and max of predictions
    predictions_array = np.array(predictions_list)
    mean_predictions = np.mean(predictions_array, axis=0)
    std_predictions = np.std(predictions_array, axis=0)
    max_predictions = np.max(predictions_array, axis=0)
    
    # Add predictions to the dataframe (all values are already non-negative due to clipping)
    df['predicted_Corg (%)'] = mean_predictions
    df['prediction_std'] = std_predictions
    df['prediction_cv'] = std_predictions / mean_predictions  # Coefficient of variation
    df['prediction_max'] = max_predictions
    
    # Save results to CSV
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    # Create and save the plots
    plot_predictions(df, output_plot)
    plot_confidence(df, confidence_plot)
    
    # Print some statistics
    print("\nPrediction Statistics:")
    print(f"Mean predicted Corg: {mean_predictions.mean():.4f}%")
    print(f"Mean prediction std: {std_predictions.mean():.4f}%")
    print(f"Mean coefficient of variation: {(std_predictions / mean_predictions).mean():.4f}")

def plot_predictions(df, output_plot='predicted_corg_map.png'):
    """
    Create a spatial plot of predicted Corg values.
    
    Args:
        df: DataFrame containing lon, lat, and predicted_Corg columns
        output_plot: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Create a regular grid to interpolate the data
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    
    # Create grid points
    n_points = 100
    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:n_points*1j, lat_min:lat_max:n_points*1j]
    
    # Prepare points and values
    points = np.column_stack((df['lon'], df['lat']))
    values = df['predicted_Corg (%)']
    
    # Clip values to remove outliers
    vmin, vmax = np.percentile(values, [1, 99])
    values_clipped = np.clip(values, vmin, vmax)
    
    # Plot predictions
    plt.scatter(df['lon'], df['lat'], c=values_clipped, cmap='viridis', s=10)
    plt.colorbar(label='Predicted Corg (%)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Carbon Content Predictions')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Map saved as {output_plot}")
    
    # Print the value range used for visualization
    print(f"\nVisualization value range:")
    print(f"Min value (1st percentile): {vmin:.4f}%")
    print(f"Max value (99th percentile): {vmax:.4f}%")

def plot_confidence(df, output_plot='corg_confidence_map.png'):
    """
    Create visualizations of prediction confidence.
    
    Args:
        df: DataFrame containing predictions and uncertainty measures
        output_plot: Path to save the plot
    """
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Standard deviation map
    plt.subplot(2, 2, 1)
    plt.scatter(df['lon'], df['lat'], c=df['prediction_std'], cmap='viridis', s=10)
    plt.colorbar(label='Standard Deviation (%)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Prediction Uncertainty (Standard Deviation)')
    
    # Plot 2: Prediction certainty map
    plt.subplot(2, 2, 2)
    # Calculate certainty as exp(-std/mean) which gives values between 0 and 1
    certainty = np.exp(-df['prediction_std'] / df['predicted_Corg (%)'])
    plt.scatter(df['lon'], df['lat'], c=certainty, cmap='viridis', s=10)
    plt.colorbar(label='Prediction Certainty')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Prediction Certainty (exp(-std/mean))')
    
    # Plot 3: Mean vs Uncertainty relationship
    plt.subplot(2, 2, 3)
    plt.scatter(df['predicted_Corg (%)'], certainty, alpha=0.5, s=10)
    plt.xlabel('Mean Predicted Corg (%)')
    plt.ylabel('Prediction Certainty')
    plt.title('Prediction Value vs. Certainty')
    
    # Plot 4: Certainty distribution
    plt.subplot(2, 2, 4)
    plt.hist(certainty, bins=50, alpha=0.75)
    plt.xlabel('Prediction Certainty')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Certainty')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence visualization saved as {output_plot}")

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    input_file = cfg.data.prediction_file
    model_path = cfg.output.model_save_path
    scaler_path = cfg.output.scaler_path
    output_file = cfg.output.output_file
    output_plot = cfg.output.output_plot
    confidence_plot = cfg.output.confidence_plot
    
    predict_corg(input_file, model_path, scaler_path, output_file, output_plot, confidence_plot, cfg)

if __name__ == "__main__":
    main()