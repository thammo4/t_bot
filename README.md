# Reinforcement Learning for Algorithmic Trading

Testing applications of RL to algo trading. Thus far, work includes:
* Simulating stock prices as sinusoids with added noise having a specified distribution
* Fetching historical stock data using the `yfinance` API
* Implementing Q-learning algorithms using TD(0) update rules
  * Single stock: Q-table
  * Multiple stocks: Approximate Q-function with artificial neural network containing single hidden layer
* Applying Supervised Learning to enrich/augment/improve the agent's environment for decision making
  * Pull data from `FRED` API to construct macroeconomic environment
  * Applying SVMs with various kernels to predict improvement in economic conditions for trading
  
