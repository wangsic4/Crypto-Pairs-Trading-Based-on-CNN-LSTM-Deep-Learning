# Example usage and demonstration
def demonstrate_strategy():
    """
    Demonstrate the Enhanced CNN-LSTM pairs trading strategy for 2020-2025 period
    """
    print("Enhanced CNN-LSTM Pairs Trading Strategy for 2020-2025")
    print("=" * 60)
    
    # Generate realistic synthetic data for 2020-2025 period
    np.random.seed(42)
    start_date = '2020-01-01'
    end_date = '2025-06-01'
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)
    
    print(f"Generating synthetic data for {n_days} days ({start_date} to {end_date})")
    
    # Create more realistic cryptocurrency price movements
    # Base trend with multiple cycles
    time_trend = np.linspace(0, 4*np.pi, n_days)
    
    # BTC price simulation with realistic volatility and trends
    btc_trend = 0.0003 *import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, concatenate
import warnings
warnings.filterwarnings('ignore')

class CNNLSTMPairsTradingStrategy:
    """
    Pairs Trading Strategy combining CNN-LSTM forecasting with cointegration analysis
    Based on the MICDL model architecture from the paper
    Optimized for 2020-2025 period with enhanced hyperparameters
    """
    
    def __init__(self, lookback_window=21, entry_threshold=1.8, exit_threshold=0.3, 
                 lstm_units=100, conv_filters=32, dense_units=512, dropout_rates=[0.4, 0.3]):
        self.lookback_window = lookback_window  # Increased for longer-term patterns
        self.entry_threshold = entry_threshold  # Lowered for more frequent trading
        self.exit_threshold = exit_threshold    # Tighter exit for better risk management
        self.lstm_units = lstm_units           # Increased capacity for complex patterns
        self.conv_filters = conv_filters       # More filters for feature extraction
        self.dense_units = dense_units         # Larger dense layers
        self.dropout_rates = dropout_rates     # Adaptive dropout rates
        
        self.model = None
        self.scaler_btc = MinMaxScaler()
        self.scaler_eth = MinMaxScaler()
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        
    def create_micdl_model(self, input_shape):
        """
        Create the Enhanced Multiple-Input Cryptocurrency Deep Learning (MICDL) model
        Optimized for 2020-2025 cryptocurrency market conditions
        """
        # Input layers for BTC and ETH
        btc_input = Input(shape=input_shape, name='btc_input')
        eth_input = Input(shape=input_shape, name='eth_input')
        
        # BTC processing branch - Enhanced architecture
        btc_conv1 = Conv1D(filters=self.conv_filters, kernel_size=3, activation='relu', padding='same')(btc_input)
        btc_conv2 = Conv1D(filters=self.conv_filters//2, kernel_size=2, activation='relu', padding='same')(btc_conv1)
        btc_pool = MaxPooling1D(pool_size=2)(btc_conv2)
        btc_dropout1 = Dropout(0.2)(btc_pool)
        btc_lstm1 = LSTM(self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(btc_dropout1)
        btc_lstm2 = LSTM(self.lstm_units//2, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(btc_lstm1)
        
        # ETH processing branch - Enhanced architecture
        eth_conv1 = Conv1D(filters=self.conv_filters, kernel_size=3, activation='relu', padding='same')(eth_input)
        eth_conv2 = Conv1D(filters=self.conv_filters//2, kernel_size=2, activation='relu', padding='same')(eth_conv1)
        eth_pool = MaxPooling1D(pool_size=2)(eth_conv2)
        eth_dropout1 = Dropout(0.2)(eth_pool)
        eth_lstm1 = LSTM(self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(eth_dropout1)
        eth_lstm2 = LSTM(self.lstm_units//2, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(eth_lstm1)
        
        # Concatenate the outputs
        concatenated = concatenate([btc_lstm2, eth_lstm2])
        
        # Enhanced dense layers for final processing
        dense1 = Dense(self.dense_units, activation='relu')(concatenated)
        batch_norm1 = BatchNormalization()(dense1)
        dropout1 = Dropout(self.dropout_rates[0])(batch_norm1)
        
        dense2 = Dense(self.dense_units//2, activation='relu')(dropout1)
        batch_norm2 = BatchNormalization()(dense2)
        dropout2 = Dropout(self.dropout_rates[1])(batch_norm2)
        
        dense3 = Dense(128, activation='relu')(dropout2)
        batch_norm3 = BatchNormalization()(dense3)
        dropout3 = Dropout(0.2)(batch_norm3)
        
        # Output layers for BTC and ETH predictions
        btc_output = Dense(1, activation='linear', name='btc_output')(dropout3)
        eth_output = Dense(1, activation='linear', name='eth_output')(dropout3)
        
        model = Model(inputs=[btc_input, eth_input], 
                     outputs=[btc_output, eth_output])
        
        # Enhanced optimizer with learning rate scheduling
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        
        model.compile(optimizer=optimizer, 
                     loss={'btc_output': 'huber', 'eth_output': 'huber'},  # More robust to outliers
                     metrics=['mae', 'mse'])
        
        return model
    
    def prepare_data(self, btc_prices, eth_prices):
        """
        Prepare data for training including returns transformation for stationarity
        """
        # Calculate returns for stationarity (as mentioned in the paper)
        btc_returns = btc_prices.pct_change().dropna()
        eth_returns = eth_prices.pct_change().dropna()
        
        # Align the data
        min_length = min(len(btc_returns), len(eth_returns))
        btc_returns = btc_returns[-min_length:]
        eth_returns = eth_returns[-min_length:]
        
        # Scale the returns
        btc_scaled = self.scaler_btc.fit_transform(btc_returns.values.reshape(-1, 1))
        eth_scaled = self.scaler_eth.fit_transform(eth_returns.values.reshape(-1, 1))
        
        return btc_scaled.flatten(), eth_scaled.flatten(), btc_returns.index
    
    def create_sequences(self, btc_data, eth_data, lookback):
        """
        Create sequences for CNN-LSTM training
        """
        X_btc, X_eth, y_btc, y_eth = [], [], [], []
        
        for i in range(lookback, len(btc_data)):
            X_btc.append(btc_data[i-lookback:i])
            X_eth.append(eth_data[i-lookback:i])
            y_btc.append(btc_data[i])
            y_eth.append(eth_data[i])
        
        return (np.array(X_btc).reshape(-1, lookback, 1),
                np.array(X_eth).reshape(-1, lookback, 1),
                np.array(y_btc),
                np.array(y_eth))
    
    def train_model(self, btc_prices, eth_prices, epochs=150, validation_split=0.15, 
                   early_stopping_patience=20, reduce_lr_patience=10):
        """
        Train the Enhanced MICDL model with advanced training techniques
        """
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        
        # Prepare data
        btc_scaled, eth_scaled, dates = self.prepare_data(btc_prices, eth_prices)
        
        # Create sequences
        X_btc, X_eth, y_btc, y_eth = self.create_sequences(
            btc_scaled, eth_scaled, self.lookback_window)
        
        print(f"Training data shape: X_btc: {X_btc.shape}, X_eth: {X_eth.shape}")
        print(f"Target data shape: y_btc: {y_btc.shape}, y_eth: {y_eth.shape}")
        
        # Create model
        input_shape = (self.lookback_window, 1)
        self.model = self.create_micdl_model(input_shape)
        
        # Enhanced callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_micdl_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train model with enhanced parameters
        history = self.model.fit(
            [X_btc, X_eth],
            [y_btc, y_eth],
            epochs=epochs,
            validation_split=validation_split,
            batch_size=64,  # Increased batch size for better gradient estimates
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1,
            shuffle=True
        )
        
        return history
    
    def predict_returns(self, btc_recent, eth_recent):
        """
        Predict next period returns using trained CNN-LSTM model
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Scale recent data
        btc_scaled = self.scaler_btc.transform(btc_recent.reshape(-1, 1)).flatten()
        eth_scaled = self.scaler_eth.transform(eth_recent.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X_btc = btc_scaled[-self.lookback_window:].reshape(1, self.lookback_window, 1)
        X_eth = eth_scaled[-self.lookback_window:].reshape(1, self.lookback_window, 1)
        
        # Predict
        btc_pred, eth_pred = self.model.predict([X_btc, X_eth])
        
        # Inverse transform
        btc_pred_return = self.scaler_btc.inverse_transform(btc_pred.reshape(-1, 1))[0, 0]
        eth_pred_return = self.scaler_eth.inverse_transform(eth_pred.reshape(-1, 1))[0, 0]
        
        return btc_pred_return, eth_pred_return
    
    def test_cointegration(self, btc_prices, eth_prices, significance_level=0.05):
        """
        Test for cointegration between BTC and ETH using Engle-Granger test
        """
        # Ensure same length
        min_length = min(len(btc_prices), len(eth_prices))
        btc_aligned = btc_prices[-min_length:]
        eth_aligned = eth_prices[-min_length:]
        
        # Perform cointegration test
        coint_stat, p_value, critical_values = coint(btc_aligned, eth_aligned)
        
        is_cointegrated = p_value < significance_level
        
        return {
            'is_cointegrated': is_cointegrated,
            'p_value': p_value,
            'coint_stat': coint_stat,
            'critical_values': critical_values
        }
    
    def estimate_hedge_ratio(self, btc_prices, eth_prices):
        """
        Estimate hedge ratio using OLS regression
        """
        # Ensure same length
        min_length = min(len(btc_prices), len(eth_prices))
        btc_aligned = btc_prices[-min_length:]
        eth_aligned = eth_prices[-min_length:]
        
        # Run regression: BTC = alpha + beta * ETH + error
        model = OLS(btc_aligned, np.column_stack([np.ones(len(eth_aligned)), eth_aligned]))
        results = model.fit()
        
        self.hedge_ratio = results.params[1]  # Beta coefficient
        return self.hedge_ratio, results
    
    def calculate_spread(self, btc_prices, eth_prices):
        """
        Calculate the spread: BTC - hedge_ratio * ETH
        """
        if self.hedge_ratio is None:
            raise ValueError("Hedge ratio must be estimated first")
        
        spread = btc_prices - self.hedge_ratio * eth_prices
        return spread
    
    def calculate_zscore(self, spread, window=60):
        """
        Calculate rolling z-score of the spread with longer window for 2020-2025 period
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def generate_enhanced_signals(self, btc_prices, eth_prices, btc_predictions, eth_predictions):
        """
        Generate enhanced trading signals combining cointegration, CNN-LSTM predictions, 
        and momentum indicators for 2020-2025 market conditions
        """
        # Calculate current spread and z-score
        spread = self.calculate_spread(btc_prices, eth_prices)
        zscore = self.calculate_zscore(spread)
        
        # Calculate momentum indicators
        btc_momentum = btc_prices.pct_change(5).rolling(10).mean()  # 5-day momentum
        eth_momentum = eth_prices.pct_change(5).rolling(10).mean()
        
        # Predict future prices using CNN-LSTM predictions
        current_btc = btc_prices.iloc[-1]
        current_eth = eth_prices.iloc[-1]
        
        predicted_btc = current_btc * (1 + btc_predictions)
        predicted_eth = current_eth * (1 + eth_predictions)
        
        # Calculate predicted spread
        predicted_spread = predicted_btc - self.hedge_ratio * predicted_eth
        current_spread = spread.iloc[-1]
        
        # Generate signals with enhanced logic
        signals = pd.DataFrame(index=btc_prices.index)
        signals['btc_position'] = 0
        signals['eth_position'] = 0
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['btc_momentum'] = btc_momentum
        signals['eth_momentum'] = eth_momentum
        signals['confidence'] = 0.0
        
        # Current values
        current_zscore = zscore.iloc[-1]
        current_btc_momentum = btc_momentum.iloc[-1] if not pd.isna(btc_momentum.iloc[-1]) else 0
        current_eth_momentum = eth_momentum.iloc[-1] if not pd.isna(eth_momentum.iloc[-1]) else 0
        
        # Enhanced trading logic with confidence scoring
        confidence = 0.0
        
        if current_zscore > self.entry_threshold:
            # Spread is too high, expect mean reversion
            if predicted_spread < current_spread:  # CNN-LSTM confirms mean reversion
                confidence += 0.5
                if current_btc_momentum > current_eth_momentum:  # BTC momentum higher
                    confidence += 0.3
                
                if confidence > 0.6:  # High confidence threshold
                    signals['btc_position'].iloc[-1] = -1
                    signals['eth_position'].iloc[-1] = self.hedge_ratio
                    
        elif current_zscore < -self.entry_threshold:
            # Spread is too low, expect mean reversion
            if predicted_spread > current_spread:  # CNN-LSTM confirms mean reversion
                confidence += 0.5
                if current_eth_momentum > current_btc_momentum:  # ETH momentum higher
                    confidence += 0.3
                
                if confidence > 0.6:  # High confidence threshold
                    signals['btc_position'].iloc[-1] = 1
                    signals['eth_position'].iloc[-1] = -self.hedge_ratio
        
        # Exit signals with momentum consideration
        if abs(current_zscore) < self.exit_threshold:
            signals['btc_position'].iloc[-1] = 0
            signals['eth_position'].iloc[-1] = 0
        
        signals['confidence'].iloc[-1] = confidence
        return signals
    
    def backtest_strategy(self, btc_prices, eth_prices, initial_capital=100000, 
                         position_size_pct=0.15, max_positions=3):
        """
        Enhanced backtest for the pairs trading strategy with improved risk management
        """
        # Test cointegration
        coint_result = self.test_cointegration(btc_prices, eth_prices)
        print(f"Cointegration test p-value: {coint_result['p_value']:.4f}")
        
        if not coint_result['is_cointegrated']:
            print("Warning: BTC and ETH may not be cointegrated!")
        
        # Estimate hedge ratio
        hedge_ratio, ols_results = self.estimate_hedge_ratio(btc_prices, eth_prices)
        print(f"Estimated hedge ratio: {hedge_ratio:.4f}")
        print(f"R-squared: {ols_results.rsquared:.4f}")
        
        # Calculate spread and z-score
        spread = self.calculate_spread(btc_prices, eth_prices)
        zscore = self.calculate_zscore(spread)
        
        # Initialize enhanced portfolio tracking
        portfolio = pd.DataFrame(index=btc_prices.index)
        portfolio['btc_holdings'] = 0.0
        portfolio['eth_holdings'] = 0.0
        portfolio['cash'] = initial_capital
        portfolio['total_value'] = initial_capital
        portfolio['returns'] = 0.0
        portfolio['active_positions'] = 0
        portfolio['trade_pnl'] = 0.0
        
        # Enhanced trading logic
        last_trade_day = 0
        min_trade_interval = 5  # Minimum days between trades
        
        for i in range(60, len(portfolio)):  # Start after z-score calculation window
            current_zscore = zscore.iloc[i]
            prev_zscore = zscore.iloc[i-1]
            
            btc_price = btc_prices.iloc[i]
            eth_price = eth_prices.iloc[i]
            
            # Copy previous positions
            portfolio['btc_holdings'].iloc[i] = portfolio['btc_holdings'].iloc[i-1]
            portfolio['eth_holdings'].iloc[i] = portfolio['eth_holdings'].iloc[i-1]
            portfolio['cash'].iloc[i] = portfolio['cash'].iloc[i-1]
            portfolio['active_positions'].iloc[i] = portfolio['active_positions'].iloc[i-1]
            
            # Check if enough time has passed since last trade
            can_trade = (i - last_trade_day) >= min_trade_interval
            
            # Enhanced entry conditions
            if (can_trade and portfolio['active_positions'].iloc[i] < max_positions and 
                abs(prev_zscore) < self.entry_threshold):
                
                if current_zscore > self.entry_threshold:
                    # Enter short spread position (BTC overpriced relative to ETH)
                    trade_size = initial_capital * position_size_pct
                    btc_shares = -trade_size / btc_price
                    eth_shares = trade_size * hedge_ratio / eth_price
                    
                    # Check if we have enough cash
                    required_cash = abs(btc_shares * btc_price) + abs(eth_shares * eth_price)
                    if portfolio['cash'].iloc[i] >= required_cash * 0.5:  # 50% margin requirement
                        portfolio['btc_holdings'].iloc[i] += btc_shares
                        portfolio['eth_holdings'].iloc[i] += eth_shares
                        portfolio['cash'].iloc[i] -= required_cash * 0.1  # Transaction costs
                        portfolio['active_positions'].iloc[i] += 1
                        last_trade_day = i
                    
                elif current_zscore < -self.entry_threshold:
                    # Enter long spread position (ETH overpriced relative to BTC)
                    trade_size = initial_capital * position_size_pct
                    btc_shares = trade_size / btc_price
                    eth_shares = -trade_size * hedge_ratio / eth_price
                    
                    # Check if we have enough cash
                    required_cash = abs(btc_shares * btc_price) + abs(eth_shares * eth_price)
                    if portfolio['cash'].iloc[i] >= required_cash * 0.5:  # 50% margin requirement
                        portfolio['btc_holdings'].iloc[i] += btc_shares
                        portfolio['eth_holdings'].iloc[i] += eth_shares
                        portfolio['cash'].iloc[i] -= required_cash * 0.1  # Transaction costs
                        portfolio['active_positions'].iloc[i] += 1
                        last_trade_day = i
            
            # Enhanced exit conditions
            elif (portfolio['active_positions'].iloc[i] > 0 and 
                  (abs(current_zscore) < self.exit_threshold or 
                   (current_zscore * prev_zscore < 0))):  # Z-score crosses zero
                
                # Close positions gradually
                close_ratio = 0.5 if abs(current_zscore) < self.exit_threshold else 1.0
                
                btc_to_close = portfolio['btc_holdings'].iloc[i] * close_ratio
                eth_to_close = portfolio['eth_holdings'].iloc[i] * close_ratio
                
                portfolio['cash'].iloc[i] += (btc_to_close * btc_price + eth_to_close * eth_price)
                portfolio['btc_holdings'].iloc[i] -= btc_to_close
                portfolio['eth_holdings'].iloc[i] -= eth_to_close
                portfolio['active_positions'].iloc[i] = max(0, portfolio['active_positions'].iloc[i] - 1)
                
            # Calculate total portfolio value
            portfolio['total_value'].iloc[i] = (portfolio['cash'].iloc[i] + 
                                              portfolio['btc_holdings'].iloc[i] * btc_price + 
                                              portfolio['eth_holdings'].iloc[i] * eth_price)
            
            # Calculate returns
            if portfolio['total_value'].iloc[i-1] > 0:
                portfolio['returns'].iloc[i] = (portfolio['total_value'].iloc[i] / 
                                              portfolio['total_value'].iloc[i-1] - 1)
        
        return portfolio, spread, zscore
    
    def plot_results(self, btc_prices, eth_prices, portfolio, spread, zscore):
        """
        Plot backtest results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Price series
        axes[0, 0].plot(btc_prices.index, btc_prices, label='BTC', alpha=0.7)
        axes[0, 0].plot(eth_prices.index, eth_prices, label='ETH', alpha=0.7)
        axes[0, 0].set_title('BTC and ETH Prices')
        axes[0, 0].set_ylabel('Price (USD)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Spread and Z-score
        ax2 = axes[0, 1]
        ax2.plot(spread.index, spread, label='Spread', color='blue', alpha=0.7)
        ax2.set_ylabel('Spread', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(zscore.index, zscore, label='Z-score', color='red', alpha=0.7)
        ax2_twin.axhline(self.entry_threshold, color='red', linestyle='--', alpha=0.5)
        ax2_twin.axhline(-self.entry_threshold, color='red', linestyle='--', alpha=0.5)
        ax2_twin.axhline(self.exit_threshold, color='green', linestyle='--', alpha=0.5)
        ax2_twin.axhline(-self.exit_threshold, color='green', linestyle='--', alpha=0.5)
        ax2_twin.set_ylabel('Z-score', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2.set_title('Spread and Z-score')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Portfolio value
        axes[1, 0].plot(portfolio.index, portfolio['total_value'])
        axes[1, 0].set_title('Portfolio Value')
        axes[1, 0].set_ylabel('Value (USD)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative returns
        cumulative_returns = (1 + portfolio['returns']).cumprod()
        axes[1, 1].plot(portfolio.index, cumulative_returns)
        axes[1, 1].set_title('Cumulative Returns')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_performance_metrics(self, portfolio):
        """
        Calculate strategy performance metrics
        """
        returns = portfolio['returns'].dropna()
        
        # Basic metrics
        total_return = (portfolio['total_value'].iloc[-1] / portfolio['total_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (returns > 0).sum()
        total_trades = (returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Trades': total_trades
        }
        
        return metrics

# Example usage and demonstration
def demonstrate_strategy():
    """
    Demonstrate the CNN-LSTM pairs trading strategy
    """
    print("CNN-LSTM Pairs Trading Strategy Demonstration")
    print("=" * 50)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Synthetic cointegrated price series
    epsilon = np.random.normal(0, 0.02, 500)  # Error term
    btc_base = np.cumsum(np.random.normal(0.001, 0.05, 500))
    eth_base = 0.6 * btc_base + np.cumsum(epsilon)  # Cointegrated with BTC
    
    # Convert to price levels
    btc_prices = pd.Series(10000 * np.exp(btc_base), index=dates)
    eth_prices = pd.Series(500 * np.exp(eth_base), index=dates)
    
    # Initialize strategy
    strategy = CNNLSTMPairsTradingStrategy(
        lookback_window=14,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    # Split data for training and testing
    train_size = int(0.7 * len(btc_prices))
    btc_train = btc_prices[:train_size]
    eth_train = eth_prices[:train_size]
    btc_test = btc_prices[train_size:]
    eth_test = eth_prices[train_size:]
    
    print(f"Training model on {len(btc_train)} data points...")
    
    # Train the CNN-LSTM model
    try:
        history = strategy.train_model(btc_train, eth_train, epochs=50, validation_split=0.2)
        print("Model training completed!")
    except Exception as e:
        print(f"Model training failed: {e}")
        print("Proceeding with traditional pairs trading strategy...")
    
    # Backtest the strategy on test data
    print("Running backtest...")
    portfolio, spread, zscore = strategy.backtest_strategy(btc_test, eth_test, initial_capital=100000)
    
    # Calculate performance metrics
    metrics = strategy.calculate_performance_metrics(portfolio)
    
    print("\nStrategy Performance Metrics:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Plot results
    strategy.plot_results(btc_test, eth_test, portfolio, spread, zscore)
    
    return strategy, portfolio, metrics

if __name__ == "__main__":
    demonstrate_strategy()
