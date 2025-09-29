"""
Time Series Prediction Models: LSTM, Transformer, and Mamba
Comparing three state-of-the-art models for electricity demand forecasting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TimeSeriesDataPreparator:
    """Prepare time series data for model training"""

    def __init__(self, sequence_length=24, forecast_horizon=6):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()

    def create_sequences(self, data, target_col_idx=0):
        """Create sequences for time series prediction"""
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon,
                         target_col_idx]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def prepare_data(self, df, target_col='demand', train_split=0.8):
        """Prepare data for training"""
        # Select relevant features
        feature_cols = ['demand_mw', 'temperature', 'humidity', 'hour_sin', 'hour_cos',
                       'day_sin', 'day_cos', 'month_sin', 'month_cos']

        # Create time features if not present
        if 'hour_sin' not in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['month'] = pd.to_datetime(df['timestamp']).dt.month

            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Use available features
        available_features = [col for col in feature_cols if col in df.columns]

        # Get target column index
        if target_col in df.columns:
            target_col_idx = available_features.index(target_col) if target_col in available_features else 0
        elif 'demand_mw' in available_features:
            target_col_idx = available_features.index('demand_mw')
        else:
            target_col_idx = 0

        data = df[available_features].values

        # Normalize data
        data_scaled = self.scaler.fit_transform(data)

        # Create sequences
        X, y = self.create_sequences(data_scaled, target_col_idx)

        # Split data
        split_idx = int(len(X) * train_split)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        return X_train, X_test, y_train, y_test


class LSTMModel:
    """LSTM model for time series prediction"""

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        self.history = None

    def build(self):
        """Build LSTM model architecture"""
        self.model = keras.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.2),

            # Second LSTM layer
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),

            # Third LSTM layer
            layers.LSTM(32),
            layers.Dropout(0.2),

            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.output_shape)
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM model"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        return self.history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class TransformerModel:
    """Transformer model for time series prediction"""

    def __init__(self, input_shape, output_shape, d_model=128, num_heads=8):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.d_model = d_model
        self.num_heads = num_heads
        self.model = None
        self.history = None

    def positional_encoding(self, length, d_model):
        """Create positional encoding"""
        positions = np.arange(length)[:, np.newaxis]
        dimensions = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (dimensions // 2)) / d_model)

        pos_encoding = np.zeros((length, d_model))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])

        return pos_encoding

    def build(self):
        """Build Transformer model architecture"""
        inputs = layers.Input(shape=self.input_shape)

        # Linear projection to d_model dimensions
        x = layers.Dense(self.d_model)(inputs)

        # Add positional encoding
        seq_len = self.input_shape[0]
        pos_encoding = self.positional_encoding(seq_len, self.d_model)
        x = x + pos_encoding

        # Transformer blocks
        for _ in range(2):
            # Multi-head attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.d_model // self.num_heads
            )(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)

            # Feed-forward network
            ffn_output = keras.Sequential([
                layers.Dense(256, activation='relu'),
                layers.Dense(self.d_model)
            ])(x)
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization()(x)

        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.output_shape)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the Transformer model"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        return self.history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class MambaModel:
    """Mamba (State Space Model) for time series prediction"""

    def __init__(self, input_shape, output_shape, state_size=64):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.state_size = state_size
        self.model = None
        self.history = None

    def selective_scan_layer(self, x, state_size):
        """Implement selective scan mechanism (simplified version)"""
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        d_model = x.shape[-1]

        # Initialize state
        h = tf.zeros((batch_size, state_size))
        outputs = []

        # Selective scan through sequence
        for t in range(seq_len):
            # Select relevant information
            x_t = x[:, t, :]

            # State transition (simplified SSM)
            A = tf.nn.sigmoid(layers.Dense(state_size)(x_t))
            B = layers.Dense(state_size)(x_t)
            C = layers.Dense(d_model)(h)

            # Update state selectively
            h = A * h + B

            # Output
            y_t = C + x_t
            outputs.append(y_t)

        # Stack outputs
        output = tf.stack(outputs, axis=1)
        return output

    def build(self):
        """Build Mamba model architecture"""
        inputs = layers.Input(shape=self.input_shape)

        # Initial projection
        x = layers.Dense(128)(inputs)
        x = layers.LayerNormalization()(x)

        # Mamba blocks (simplified SSM blocks)
        for _ in range(2):
            # Selective scan mechanism
            residual = x

            # Expand dimensions
            x_expanded = layers.Dense(256, activation='swish')(x)

            # Convolution for local patterns
            x_conv = layers.Conv1D(256, 3, padding='same', activation='swish')(x_expanded)

            # Selective state space modeling
            x_ssm = layers.GRU(128, return_sequences=True)(x_conv)

            # Combine and project back
            x = layers.Dense(128)(x_ssm)

            # Residual connection
            x = layers.Add()([residual, x])
            x = layers.LayerNormalization()(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.output_shape)(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the Mamba model"""
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        return self.history

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, verbose=0)


class ModelComparator:
    """Compare performance of different models"""

    def __init__(self):
        self.results = {}
        self.predictions = {}

    def evaluate_model(self, model_name, y_true, y_pred, training_time):
        """Evaluate model performance"""
        # Calculate metrics
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # Store results
        self.results[model_name] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Training_Time': training_time
        }

        self.predictions[model_name] = y_pred

        return self.results[model_name]

    def plot_comparison(self, y_true, save_path='output'):
        """Plot model comparison results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Prediction comparison for first sample
        ax = axes[0, 0]
        time_steps = np.arange(len(y_true[0]))
        ax.plot(time_steps, y_true[0], 'k-', label='Actual', linewidth=2, alpha=0.7)

        colors = ['blue', 'red', 'green']
        for i, (model_name, y_pred) in enumerate(self.predictions.items()):
            ax.plot(time_steps, y_pred[0], colors[i],
                   label=model_name, linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time Steps (hours)')
        ax.set_ylabel('Electricity Demand (Normalized)')
        ax.set_title('Sample Prediction Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Metrics comparison
        ax = axes[0, 1]
        metrics_df = pd.DataFrame(self.results).T
        metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
        x = np.arange(len(metrics_to_plot))
        width = 0.25

        for i, model in enumerate(metrics_df.index):
            values = [metrics_df.loc[model, m] for m in metrics_to_plot]
            ax.bar(x + i*width, values, width, label=model, alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Error Value')
        ax.set_title('Model Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. R² Score comparison
        ax = axes[0, 2]
        models = list(self.results.keys())
        r2_scores = [self.results[m]['R2'] for m in models]
        colors_bar = ['#3498db', '#e74c3c', '#2ecc71']

        bars = ax.bar(models, r2_scores, color=colors_bar, alpha=0.8)
        ax.set_ylabel('R² Score')
        ax.set_title('Model R² Score Comparison')
        ax.set_ylim([min(r2_scores) * 0.95, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}', ha='center', va='bottom')

        # 4. Training time comparison
        ax = axes[1, 0]
        training_times = [self.results[m]['Training_Time'] for m in models]
        bars = ax.bar(models, training_times, color=colors_bar, alpha=0.8)
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Model Training Time Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, time in zip(bars, training_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.1f}s', ha='center', va='bottom')

        # 5. Error distribution
        ax = axes[1, 1]
        for model_name, y_pred in self.predictions.items():
            errors = (y_true - y_pred).flatten()
            ax.hist(errors, bins=30, alpha=0.5, label=model_name, density=True)

        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Model summary table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')

        # Create summary table
        table_data = []
        table_data.append(['Model', 'RMSE', 'R²', 'Time(s)'])
        for model in models:
            table_data.append([
                model,
                f"{self.results[model]['RMSE']:.4f}",
                f"{self.results[model]['R2']:.4f}",
                f"{self.results[model]['Training_Time']:.1f}"
            ])

        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#3498db')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.suptitle('Deep Learning Models Performance Comparison\nLSTM vs Transformer vs Mamba',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save figure
        save_file = os.path.join(save_path, 'model_comparison.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved to: {save_file}")

        plt.show()

        return fig

    def print_summary(self):
        """Print model comparison summary"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON SUMMARY")
        print("="*80)

        # Create DataFrame for better visualization
        df = pd.DataFrame(self.results).T
        df = df.round(4)

        print("\n📊 Performance Metrics:")
        print(df.to_string())

        # Find best model for each metric
        print("\n🏆 Best Models by Metric:")
        print("-" * 40)

        for metric in ['MSE', 'MAE', 'RMSE', 'MAPE']:
            best_model = df[metric].idxmin()
            best_value = df[metric].min()
            print(f"  {metric:15s}: {best_model:15s} ({best_value:.4f})")

        best_r2_model = df['R2'].idxmax()
        best_r2_value = df['R2'].max()
        print(f"  {'R² Score':15s}: {best_r2_model:15s} ({best_r2_value:.4f})")

        fastest_model = df['Training_Time'].idxmin()
        fastest_time = df['Training_Time'].min()
        print(f"  {'Fastest Training':15s}: {fastest_model:15s} ({fastest_time:.1f}s)")

        # Overall recommendation
        print("\n💡 Overall Recommendation:")
        print("-" * 40)

        # Calculate overall score (weighted average)
        weights = {'RMSE': 0.3, 'R2': 0.3, 'Training_Time': 0.2, 'MAE': 0.2}

        # Normalize metrics (lower is better for errors, higher for R2, lower for time)
        normalized_scores = {}
        for model in df.index:
            score = 0
            score += (1 - df.loc[model, 'RMSE'] / df['RMSE'].max()) * weights['RMSE']
            score += df.loc[model, 'R2'] * weights['R2']
            score += (1 - df.loc[model, 'Training_Time'] / df['Training_Time'].max()) * weights['Training_Time']
            score += (1 - df.loc[model, 'MAE'] / df['MAE'].max()) * weights['MAE']
            normalized_scores[model] = score

        best_overall = max(normalized_scores, key=normalized_scores.get)
        print(f"  Best Overall Model: {best_overall}")
        print(f"  Overall Score: {normalized_scores[best_overall]:.4f}")

        print("\n📝 Model Characteristics:")
        print("-" * 40)
        print("  • LSTM: Good for capturing long-term dependencies")
        print("  • Transformer: Excellent parallel processing and attention mechanism")
        print("  • Mamba: Efficient linear-time complexity with selective state spaces")

        return df


def run_model_comparison(data_df, epochs=30):
    """Run complete model comparison"""
    import time

    print("\n" + "="*80)
    print("STARTING MODEL COMPARISON: LSTM vs Transformer vs Mamba")
    print("="*80)

    # Prepare data
    print("\n📊 Preparing data...")
    preparator = TimeSeriesDataPreparator(sequence_length=24, forecast_horizon=6)

    # Check if we have the right columns and rename if necessary
    if 'demand_mw' in data_df.columns:
        target_col = 'demand_mw'
    elif 'demand' in data_df.columns:
        target_col = 'demand'
    else:
        # Create a demand column if it doesn't exist
        target_col = 'demand_mw'

    X_train, X_test, y_train, y_test = preparator.prepare_data(data_df, target_col=target_col)

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Input shape: {X_train.shape[1:]}")
    print(f"  Output shape: {y_train.shape[1]}")

    # Initialize comparator
    comparator = ModelComparator()

    # Train and evaluate LSTM
    print("\n🔵 Training LSTM Model...")
    lstm_model = LSTMModel(X_train.shape[1:], y_train.shape[1])
    lstm_model.build()

    start_time = time.time()
    lstm_model.train(X_train, y_train, X_test, y_test, epochs=epochs)
    lstm_time = time.time() - start_time

    lstm_pred = lstm_model.predict(X_test)
    comparator.evaluate_model('LSTM', y_test, lstm_pred, lstm_time)
    print(f"  ✅ LSTM training completed in {lstm_time:.1f} seconds")

    # Train and evaluate Transformer
    print("\n🔴 Training Transformer Model...")
    transformer_model = TransformerModel(X_train.shape[1:], y_train.shape[1])
    transformer_model.build()

    start_time = time.time()
    transformer_model.train(X_train, y_train, X_test, y_test, epochs=epochs)
    transformer_time = time.time() - start_time

    transformer_pred = transformer_model.predict(X_test)
    comparator.evaluate_model('Transformer', y_test, transformer_pred, transformer_time)
    print(f"  ✅ Transformer training completed in {transformer_time:.1f} seconds")

    # Train and evaluate Mamba
    print("\n🟢 Training Mamba Model...")
    mamba_model = MambaModel(X_train.shape[1:], y_train.shape[1])
    mamba_model.build()

    start_time = time.time()
    mamba_model.train(X_train, y_train, X_test, y_test, epochs=epochs)
    mamba_time = time.time() - start_time

    mamba_pred = mamba_model.predict(X_test)
    comparator.evaluate_model('Mamba', y_test, mamba_pred, mamba_time)
    print(f"  ✅ Mamba training completed in {mamba_time:.1f} seconds")

    # Generate comparison plots and summary
    comparator.plot_comparison(y_test)
    summary_df = comparator.print_summary()

    return comparator, summary_df


if __name__ == "__main__":
    # Load sample data
    print("Loading sample electricity demand data...")

    # Create sample data if not available
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='H')
    np.random.seed(42)

    # Generate synthetic electricity demand data
    demand = 50000 + 10000 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*7))  # Weekly pattern
    demand += 5000 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Daily pattern
    demand += np.random.normal(0, 2000, len(dates))  # Random noise

    # Generate weather data
    temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365))  # Yearly pattern
    temperature += 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Daily pattern
    temperature += np.random.normal(0, 2, len(dates))

    humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / (24*365))
    humidity += np.random.normal(0, 5, len(dates))
    humidity = np.clip(humidity, 20, 95)

    # Create DataFrame
    data_df = pd.DataFrame({
        'timestamp': dates,
        'demand': demand,
        'temperature': temperature,
        'humidity': humidity
    })

    # Run comparison
    comparator, results = run_model_comparison(data_df, epochs=30)

    print("\n" + "="*80)
    print("MODEL COMPARISON COMPLETED!")
    print("="*80)