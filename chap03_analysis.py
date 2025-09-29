"""
Deep Learning-based Policy Time Series Prediction - Practical Analysis Module
Module for data generation, preprocessing, model training and evaluation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime, timedelta
import joblib

warnings.filterwarnings('ignore')

# Font settings
import platform

# Set default font for all platforms
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Fix minus sign display
plt.rcParams['axes.unicode_minus'] = False

# Visualization style settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create necessary directories
os.makedirs(os.path.join('output'), exist_ok=True)
os.makedirs(os.path.join('data'), exist_ok=True)
os.makedirs(os.path.join('models'), exist_ok=True)


def load_and_prepare_data(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join('data')
    """Load saved data and prepare for modeling"""

    # Check files
    required_files = ['energy_demand.csv', 'renewable_policy.csv', 'electricity_market.csv']
    files_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)

    if not files_exist:
        raise FileNotFoundError("Required data files not found. Please check the data/ folder.")

    # Load data
    print("\n📊 Loading data...")
    demand_df = pd.read_csv(os.path.join(data_dir, 'energy_demand.csv'), parse_dates=['timestamp'])
    policy_df = pd.read_csv(os.path.join(data_dir, 'renewable_policy.csv'), parse_dates=['timestamp'])
    market_df = pd.read_csv(os.path.join(data_dir, 'electricity_market.csv'), parse_dates=['timestamp'])

    # Merge data
    merged_df = demand_df.merge(market_df, on='timestamp', how='left')
    merged_df = pd.merge_asof(merged_df.sort_values('timestamp'),
                              policy_df.sort_values('timestamp'),
                              on='timestamp',
                              direction='backward')
    merged_df = merged_df.ffill().fillna(0)

    print(f"✅ Data loaded successfully: {len(merged_df)} records")

    return merged_df

def create_sequences(data, sequence_length=24, target_col='demand_mw'):
    """Create sequences for time series prediction"""

    # Exclude timestamp column if exists
    cols_to_drop = [col for col in ['timestamp', target_col] if col in data.columns]
    features = data.drop(columns=cols_to_drop).values
    targets = data[target_col].values

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(targets[i+sequence_length])

    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def build_gru_model(input_shape):
    """Build GRU model"""
    model = keras.Sequential([
        keras.layers.GRU(64, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.GRU(32, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def build_simple_rnn_model(input_shape):
    """Build simple RNN model (for comparison)"""
    model = keras.Sequential([
        keras.layers.SimpleRNN(32, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.SimpleRNN(16),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['mae'])
    return model

def save_models(models, scaler, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join('models')
    """Save trained models and scaler"""
    print("\n💾 Saving models...")

    for name, model in models.items():
        model_path = os.path.join(save_dir, f'{name}_model.keras')
        model.save(model_path)
        print(f"✅ {name} model saved: {model_path}")

    # Save scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    print(f"\nAll models have been saved to {save_dir}/ folder.")

def load_models(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join('models')
    """Load saved models and scaler"""
    print("\n📂 Loading saved models...")

    models = {}
    model_files = ['LSTM_model.keras', 'GRU_model.keras', 'RNN_model.keras']

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            model_name = model_file.replace('_model.keras', '')
            models[model_name] = keras.models.load_model(model_path)
            print(f"✅ {model_name} model loaded: {model_path}")
        else:
            print(f"⚠️ {model_file} file not found.")

    # Load scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded: {scaler_path}")
    else:
        print("⚠️ Scaler file not found.")

    if not models:
        raise FileNotFoundError("No models available to load. Please train and save models first.")

    return models, scaler

def train_and_evaluate_models(data, epochs=10, save=True):
    """Model training and evaluation"""
    print("\n" + "="*50)
    print("Model Training and Evaluation")
    print("="*50)

    # Data preprocessing
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.fit_transform(data[numeric_columns])

    # Create sequences
    X, y = create_sequences(data_scaled, sequence_length=24)

    # Train/Validation/Test split
    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")

    models = {}
    histories = {}

    # Train LSTM model
    print("\n🔄 Training LSTM model...")
    lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    models['LSTM'] = lstm_model
    histories['LSTM'] = lstm_history

    # Train GRU model
    print("\n🔄 Training GRU model...")
    gru_model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    models['GRU'] = gru_model
    histories['GRU'] = gru_history

    # Train simple RNN model (optional)
    print("\n🔄 Training Simple RNN model...")
    rnn_model = build_simple_rnn_model((X_train.shape[1], X_train.shape[2]))
    rnn_history = rnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    models['RNN'] = rnn_model
    histories['RNN'] = rnn_history

    # Model evaluation
    print("\n📊 Model Performance Evaluation")
    print("-" * 50)

    results = {}
    for name, model in models.items():
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {'loss': test_loss, 'mae': test_mae}
        print(f"{name:10s} - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Save models
    if save:
        save_models(models, scaler)

    # Visualize predictions
    visualize_predictions(models, X_test, y_test, scaler)
    visualize_training_history(histories)

    return models, histories, results, scaler

def visualize_predictions(models, X_test, y_test, scaler):
    """Visualize prediction results"""

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(15, 5*n_models))

    if n_models == 1:
        axes = [axes]

    # Select sample range (first 200)
    n_show = min(200, len(y_test))

    for idx, (name, model) in enumerate(models.items()):
        # Prediction
        pred = model.predict(X_test)

        # Visualization
        ax = axes[idx]
        ax.plot(y_test[:n_show], label='Actual', alpha=0.7, linewidth=1.5)
        ax.plot(pred[:n_show], label=f'{name} Prediction', alpha=0.7, linewidth=1.5)
        ax.set_title(f'{name} Model Prediction Results')
        ax.set_xlabel('Time')
        ax.set_ylabel('Power Demand (Normalized)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add caption
        mae = np.mean(np.abs(pred[:n_show].flatten() - y_test[:n_show]))
        ax.text(0.5, -0.15, f'{name} model power demand prediction: Predicts next hour using 24 hours of prior data\nMAE: {mae:.4f} (closer to 0 is more accurate)',
                ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'model_predictions.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ Prediction visualization complete ({os.path.join('output', 'model_predictions.png')})")

def visualize_training_history(histories):
    """Visualize training history"""

    n_models = len(histories)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))

    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for idx, (name, history) in enumerate(histories.items()):
        # Loss graph
        ax = axes[0, idx]
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Val Loss')
        ax.set_title(f'{name} - Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.2, 'Training Loss: Error reduction trend for training and validation data',
                ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

        # MAE graph
        ax = axes[1, idx]
        ax.plot(history.history['mae'], label='Train MAE')
        ax.plot(history.history['val_mae'], label='Val MAE')
        ax.set_title(f'{name} - MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.5, -0.2, 'Mean Absolute Error: Average difference between predicted and actual values',
                ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ Training history visualization complete ({os.path.join('output', 'training_history.png')})")

def analyze_policy_impact(data, model):
    """Policy impact analysis"""
    print("\n" + "="*50)
    print("Policy Impact Analysis")
    print("="*50)

    # Find policy change points
    policy_cols = ['renewable_target', 'subsidy_rate', 'carbon_tax']

    fig, axes = plt.subplots(len(policy_cols), 1, figsize=(15, 10))

    for idx, col in enumerate(policy_cols):
        if col in data.columns:
            ax = axes[idx]

            # Relationship between policy variables and power demand
            ax2 = ax.twinx()

            ax.plot(data.index[:1000], data[col].iloc[:1000],
                   color='blue', alpha=0.7, label=col)
            ax2.plot(data.index[:1000], data['demand_mw'].iloc[:1000],
                    color='red', alpha=0.5, label='Power Demand')

            ax.set_xlabel('Time')
            ax.set_ylabel(col, color='blue')
            ax2.set_ylabel('Power Demand', color='red')
            ax.tick_params(axis='y', labelcolor='blue')
            ax2.tick_params(axis='y', labelcolor='red')

            ax.set_title(f'Relationship between {col} and Power Demand')
            ax.grid(True, alpha=0.3)

            # Add caption
            caption_text = {
                'renewable_target': 'Analysis of renewable energy target increase impact on power demand',
                'subsidy_rate': 'Correlation between subsidy rate changes and power demand',
                'carbon_tax': 'Effect of carbon tax policy on power consumption patterns'
            }.get(col, '')

            if caption_text:
                ax.text(0.5, -0.15, caption_text,
                        ha='center', fontsize=9, style='italic', wrap=True, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'policy_impact_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ Policy impact analysis complete ({os.path.join('output', 'policy_impact_analysis.png')})")

def perform_statistical_analysis(data):
    """Perform statistical analysis"""
    print("\n" + "="*50)
    print("Statistical Analysis")
    print("="*50)

    # Basic statistics
    print("\n📊 Basic Statistics:")
    print("-" * 50)
    print(data.describe())

    # Correlation analysis
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation between Variables')

    # Heatmap caption
    plt.figtext(0.5, -0.02, 'Correlation coefficient: -1 (perfect negative) ~ 0 (no correlation) ~ +1 (perfect positive)\nVariables with strong correlation are important for prediction',
                ha='center', fontsize=9, style='italic', wrap=True)

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ Correlation heatmap generated ({os.path.join('output', 'correlation_heatmap.png')})")

    # Print main correlations
    print("\n📈 Main correlations with power demand:")
    print("-" * 50)
    demand_corr = correlation_matrix['demand_mw'].sort_values(ascending=False)
    for var, corr in demand_corr.items():
        if var != 'demand_mw' and abs(corr) > 0.3:
            print(f"{var:20s}: {corr:+.3f}")

def evaluate_loaded_models(models, data, scaler):
    """Perform predictions with saved models"""
    print("\n" + "="*50)
    print("Predictions with Saved Models")
    print("="*50)

    # Data preprocessing
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data_scaled = data.copy()
    data_scaled[numeric_columns] = scaler.transform(data[numeric_columns])

    # Create sequences
    X, y = create_sequences(data_scaled, sequence_length=24)

    # Split test data
    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"Test data: {X_test.shape}")

    # Model evaluation
    print("\n📊 Model Performance Evaluation")
    print("-" * 50)

    results = {}
    for name, model in models.items():
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        results[name] = {'loss': test_loss, 'mae': test_mae}
        print(f"{name:10s} - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

    # Visualize predictions
    visualize_predictions(models, X_test, y_test, scaler)

    return results

def main():
    """Main execution function"""
    import sys

    print("\n" + "="*60)
    print("Deep Learning-based Policy Time Series Prediction - Practical Analysis")
    print("="*60)

    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            choice = int(sys.argv[1])
            if choice not in [1, 2, 3, 4]:
                print(f"Error: {choice} is not a valid choice. Please enter a number between 1-4.")
                sys.exit(1)
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid number.")
            sys.exit(1)
    else:
        print("\nSelect task to execute:")
        print("1. Model training and saving")
        print("2. Load and evaluate saved models")
        print("3. Policy impact analysis (using saved models)")
        print("4. Statistical analysis (data correlation, basic stats, heatmap generation)")

        while True:
            try:
                choice = input("\nChoice (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    choice = int(choice)
                    break
                else:
                    print("Please enter a valid choice (1-4)")
            except KeyboardInterrupt:
                print("\nExiting program.")
                sys.exit(0)
            except:
                print("Invalid input. Please try again.")

    # Load data
    data = load_and_prepare_data()

    if choice == 1:
        # Model training and saving
        models, histories, results, scaler = train_and_evaluate_models(data, epochs=3, save=True)
        print("\n✅ Model training and saving complete!")

    elif choice == 2:
        # Load and evaluate saved models
        try:
            models, scaler = load_models()
            results = evaluate_loaded_models(models, data, scaler)
            print("\n✅ Saved model evaluation complete!")
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("Please select option 1 first to train and save models.")

    elif choice == 3:
        # Policy impact analysis (using saved models)
        try:
            models, scaler = load_models()
            analyze_policy_impact(data, models['LSTM'])
            print("\n✅ Policy impact analysis complete!")
        except FileNotFoundError as e:
            print(f"\n❌ Error: {e}")
            print("Please select option 1 first to train and save models.")

    elif choice == 4:
        # Statistical analysis
        perform_statistical_analysis(data)
        print("\n✅ Statistical analysis complete!")

    print("\n" + "="*60)
    print("Analysis Program Finished")
    print("="*60)

if __name__ == "__main__":
    main()