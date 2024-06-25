# Stock-Portfolio-creator
Portfolio Optimization Using LINPROG
Overview
This project focuses on optimizing a stock portfolio using linear programming (LINPROG). The goal is to maximize the Sharpe ratio of the portfolio, which measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. The project involves collecting historical stock prices, calculating returns, optimizing portfolio weights using linear programming, and comparing the optimized portfolio to a naive equally weighted portfolio.

Project Structure
historical_prices.csv: Combined historical adjusted close prices for selected stocks from August 21, 2018, to August 21, 2021.
Testing_data.csv: Combined historical adjusted close prices for selected stocks from August 21, 2021, to August 21, 2023.
Returns1.csv: Calculated returns based on the historical prices.
portfolio_optimization.py: Python script containing the entire code for the project.
README.md: Detailed description of the project (this file).

Dependencies
pandas: For data manipulation and analysis.
numpy: For numerical operations.
scipy.optimize: For linear programming optimization using linprog.

Acknowledgements
Data sourced from Yahoo Finance.
The project is inspired by various online tutorials and articles on portfolio optimization and linear programming.

Conclusion
This project demonstrates the application of linear programming for optimizing a stock portfolio. By maximizing the Sharpe ratio, the portfolio's performance is enhanced compared to a naive equally weighted portfolio. The approach can be extended to include more constraints and incorporate additional factors for more sophisticated portfolio management.
