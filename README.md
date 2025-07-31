# CNN-LSTM Pairs Trading Strategy for Cryptocurrency Markets

## Overview

This project implements an advanced pairs trading strategy that combines **Deep Learning forecasting** with **Statistical Arbitrage** techniques for cryptocurrency markets. The strategy is based on the research paper "An Advanced CNN-LSTM Model for Cryptocurrency Forecasting" by Livieris et al. (2021) and extends their Multiple-Input Cryptocurrency Deep Learning (MICDL) model for practical trading applications.

## 🚀 Key Features

### 1. **Advanced Deep Learning Architecture**
- **Multi-Input CNN-LSTM Model**: Processes BTC and ETH data through separate convolutional and LSTM branches
- **Enhanced Architecture**: Optimized for 2020-2025 market conditions with deeper networks and improved regularization
- **Stationarity Handling**: Automatic returns transformation to ensure time series stationarity (as recommended in the original paper)

### 2. **Statistical Arbitrage Foundation**
- **Cointegration Testing**: Engle-Granger test to verify long-term equilibrium relationship between BTC and ETH
- **Hedge Ratio Estimation**: OLS regression to determine optimal position sizing
- **Mean Reversion Signals**: Z-score based entry/exit signals with dynamic thresholds

### 3. **Enhanced Trading Logic**
- **Hybrid Signal Generation**: Combines traditional mean reversion with CNN-LSTM momentum predictions
- **Confidence Scoring**: Multi-factor signal validation before trade execution
- **Risk Management**: Position sizing limits, maximum concurrent positions, and transaction cost modeling

## 📊 Strategy Methodology

### Phase 1: Data Preparation
```python
# Transform prices to returns for stationarity
btc_returns = btc_prices.pct_change().dropna()
eth_returns = eth_prices.pct_change().dropna()

# Scale data for neural network training
btc_scaled = scaler.fit_transform(btc_returns.values.reshape(-1, 1))
```

### Phase 2: CNN-LSTM Model Training
```python
# Multi-input architecture with enhanced parameters
- Lookback Window: 21 days (increased from 14 for longer patterns)
- LSTM Units: 100 (doubled for complex pattern recognition)
- Conv Filters: 32 (increased feature extraction capability)
- Dense Units: 512 (enhanced representation learning)
- Dropout: Adaptive rates [0.4, 0.3] for better regularization
```

### Phase 3: Cointegration Analysis
```python
# Test for long-term equilibrium relationship
coint_stat, p_value, critical_values = coint(btc_prices, eth_prices)

# Estimate hedge ratio via OLS regression
# BTC = α + β × ETH + ε
hedge_ratio = ols_results.params[1]
```

### Phase 4: Signal Generation
```python
# Calculate spread and z-score
spread = btc_prices - hedge_ratio * eth_prices
zscore = (spread - rolling_mean) / rolling_std

# Combine with CNN-LSTM predictions
if zscore > entry_threshold and cnn_lstm_confirms_reversion:
    signal = "SHORT_SPREAD"  # Short BTC, Long ETH
```

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install numpy pandas matplotlib scikit-learn statsmodels tensorflow keras
```

### Quick Start
```python
from cnn_lstm_pairs_trading import CNNLSTMPairsTradingStrategy

# Initialize strategy with optimized parameters
strategy = CNNLSTMPairsTradingStrategy(
    lookback_window=21,
    entry_threshold=1.8,
    exit_threshold=0.3,
    lstm_units=100,
    conv_filters=32
)

# Train the model (replace with real data)
history = strategy.train_model(btc_prices, eth_prices, epochs=150)

# Run backtest
portfolio, spread, zscore = strategy.backtest_strategy(
    btc_prices, eth_prices, 
    initial_capital=100000
)

# Analyze performance
metrics = strategy.calculate_performance_metrics(portfolio)
strategy.plot_results(btc_prices, eth_prices, portfolio, spread, zscore)
```

## 📈 Strategy Performance Metrics

The strategy tracks comprehensive performance indicators:

| Metric | Description |
|--------|-------------|
| **Total Return** | Cumulative portfolio return over backtest period |
| **Annualized Return** | Compound annual growth rate (CAGR) |
| **Sharpe Ratio** | Risk-adjusted return measure |
| **Maximum Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of profitable trades |
| **Volatility** | Annualized standard deviation of returns |

## 🏗️ Architecture Details

### Enhanced CNN-LSTM Model Structure

```
Input Layer (BTC/ETH) → Conv1D (32 filters) → Conv1D (16 filters) → MaxPooling1D
                      ↓
                   Dropout (0.2) → LSTM (100 units) → LSTM (50 units)
                      ↓
              Concatenate → Dense (512) → BatchNorm → Dropout (0.4)
                      ↓
                Dense (256) → BatchNorm → Dropout (0.3) → Dense (128)
                      ↓
                Output Layer (BTC/ETH Predictions)
```

### Key Optimizations for 2020-2025 Period

1. **Increased Model Capacity**: 
   - LSTM units: 50 → 100
   - Conv filters: 16 → 32
   - Dense units: 256 → 512

2. **Enhanced Regularization**:
   - Adaptive dropout rates
   - Batch normalization layers
   - Early stopping with patience

3. **Improved Training**:
   - Huber loss (robust to outliers)
   - Adam optimizer with learning rate scheduling
   - Extended validation patience

4. **Advanced Risk Management**:
   - Position size limits (15% vs 10%)
   - Maximum concurrent positions (3)
   - Transaction cost modeling
   - Minimum trade intervals

## 📊 Backtesting Framework

### Risk Management Features
- **Position Sizing**: Configurable percentage of capital per trade
- **Maximum Positions**: Limit concurrent open positions
- **Transaction Costs**: Realistic cost modeling (0.1% per trade)
- **Margin Requirements**: 50% margin for leveraged positions

### Signal Validation
- **Cointegration Verification**: Ensures statistical relationship exists
- **Confidence Scoring**: Multi-factor signal strength assessment
- **Momentum Confirmation**: Additional technical indicators
- **Mean Reversion Timing**: CNN-LSTM prediction validation

## 🔬 Research Foundation

This implementation is based on:

**"An Advanced CNN-LSTM Model for Cryptocurrency Forecasting"**  
*Livieris, I.E., Kiriakidou, N., Stavroyiannis, S., Pintelas, P.*  
Electronics 2021, 10, 287

### Key Research Findings Applied:
1. **Stationarity Importance**: Returns transformation is essential for reliable deep learning predictions
2. **Multi-Input Architecture**: Separate processing of different cryptocurrencies improves performance
3. **Feature Extraction**: CNN layers effectively filter noise and extract valuable patterns
4. **Temporal Dependencies**: LSTM layers capture both short and long-term market relationships

## 📁 Project Structure

```
├── cnn_lstm_pairs_trading.py    # Main strategy implementation
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── examples/
│   ├── basic_example.py         # Simple usage example
│   ├── advanced_backtesting.py  # Comprehensive backtesting
│   └── hyperparameter_tuning.py # Parameter optimization
├── data/
│   ├── btc_prices.csv          # BTC historical data
│   └── eth_prices.csv          # ETH historical data
└── results/
    ├── backtest_results.png    # Performance visualization
    ├── model_architecture.png  # Network structure diagram
    └── performance_metrics.json # Detailed statistics
```

## ⚠️ Risk Disclaimers

### Important Considerations

1. **Market Risk**: Cryptocurrency markets are highly volatile and unpredictable
2. **Model Risk**: Deep learning models may not generalize to future market conditions
3. **Execution Risk**: Real-world trading involves slippage, fees, and liquidity constraints
4. **Cointegration Risk**: Statistical relationships can break down during market stress
5. **Overfitting Risk**: Backtesting may not reflect future performance

### Recommended Usage
- **Paper Trading First**: Test strategy with simulated capital before real implementation
- **Risk Management**: Never risk more than you can afford to lose
- **Continuous Monitoring**: Regularly validate cointegration relationships
- **Parameter Updates**: Retrain models periodically with new data

## 🔄 Future Enhancements

### Planned Features
- [ ] **Multi-Asset Extension**: Support for additional cryptocurrency pairs
- [ ] **Real-Time Trading**: Integration with cryptocurrency exchanges
- [ ] **Advanced Features**: Technical indicators, sentiment analysis, options Greeks
- [ ] **Risk Analytics**: VaR calculation, stress testing, scenario analysis
- [ ] **Hyperparameter Optimization**: Automated parameter tuning with Optuna
- [ ] **Model Ensemble**: Combination of multiple prediction models

### Research Extensions
- [ ] **Transformer Architecture**: Attention-based models for sequence prediction
- [ ] **Reinforcement Learning**: Q-learning for adaptive trading strategies
- [ ] **Alternative Data**: Social media sentiment, on-chain metrics
- [ ] **Cross-Asset Analysis**: Include traditional asset correlations

## 📖 References

1. Livieris, I.E., et al. "An Advanced CNN-LSTM Model for Cryptocurrency Forecasting." Electronics 2021, 10, 287.
2. Engle, R.F., Granger, C.W.J. "Co-integration and Error Correction: Representation, Estimation, and Testing." Econometrica 1987.
3. Gatev, E., Goetzmann, W.N., Rouwenhorst, K.G. "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." Review of Financial Studies 2006.

## 📞 Contact & Support

For questions, suggestions, or collaboration opportunities:
- **Issues**: Please open GitHub issues for bug reports or feature requests
- **Discussions**: Use GitHub Discussions for strategy ideas and implementation questions
- **Contributions**: Pull requests welcome! Please read CONTRIBUTING.md first

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⚠️ Trading Disclaimer**: This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consider seeking advice from qualified financial professionals before making investment decisions.
