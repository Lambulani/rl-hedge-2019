import os
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from envs import TradingEnv

def cost(delta_h, multiplier):
    TickSize = 0.1
    return multiplier * TickSize * (np.abs(delta_h) + 0.01 * delta_h**2)


bs_env = TradingEnv(continuous_action_flag=True, sabr_flag=False, 
                    dg_random_seed= 1, spread=0.01, num_contract=1, 
                    init_ttm=10, trade_freq=0.2, num_sim= 10000,
                        mu =0, vol = 0.01, S = 100, K = 100, r = 0, q = 0)

delta_path = bs_env.delta_path*100
delta_path = np.insert(delta_path, 0, 0, axis=1)
delta_h = np.diff(delta_path, axis=1)
transaction_costs = cost(delta_h, multiplier=5)

# total_costs = np.sum(transaction_costs, axis =1)


# df = pd.DataFrame({"Sample Path ": np.arange(1, 11), "Total Cost" : total_costs})
# with open("hedging_costs_table.tex", "w") as f:
#     f.write(df.to_latex(index=False))

# sns.set_style("whitegrid")

# # Plot histogram for the distribution of total hedging costs
# plt.figure(figsize=(10, 5))
# sns.histplot(total_costs, bins=10, kde=True, color="orange")
# plt.xlabel("Total Hedging Cost")
# plt.ylabel("Frequency")
# plt.title("Distribution of Total Hedging Costs")
# plt.savefig("hedging_costs_hist.png")
# plt.show()

# # Box plot to identify outliers
# plt.figure(figsize=(8, 4))
# sns.boxplot(x=total_costs, color="orange")
# plt.xlabel("Total Hedging Cost")
# plt.title("Box Plot of Total Hedging Costs (Outliers Detection)")
# plt.savefig("hedging_costs_boxplot.png")
# plt.show()

# rebalance_frequencies = [1, 2, 5, 10]  
# colors = ["blue", "green", "orange", "red"]  # Different colors for each distribution
# num_simulations = 10000

# plt.figure(figsize=(10, 5))
# sns.set_style("whitegrid")

# for i, freq in enumerate(rebalance_frequencies):
#     # Create environment with given rebalancing frequency
#     bs_env = TradingEnv(continuous_action_flag=True, sabr_flag=False, 
#                         dg_random_seed=1, spread=0.01, num_contract=1, 
#                         init_ttm=10, trade_freq=1/freq, num_sim=num_simulations,
#                         mu=0, vol=0.01, S=100, K=100, r=0, q=0)

#     # Compute total hedging cost
#     delta_path = bs_env.delta_path * 100
#     delta_path = np.insert(delta_path, 0, 0, axis=1)
#     delta_h = np.diff(delta_path, axis=1)
#     transaction_costs = cost(delta_h, multiplier=5)
#     total_costs = np.sum(transaction_costs, axis=1)

#     # Plot KDE for the current rebalancing frequency
#     sns.kdeplot(total_costs, label=f"{freq} trades/day", color=colors[i], linewidth=2)

# # Graph formatting
# plt.xlabel("Total Hedging Cost")
# plt.ylabel("Density")
# plt.title("Effect of Rebalancing Frequency on Hedging Costs")
# plt.legend()
# plt.savefig("hedging_costs_Rebalancing.png")
# plt.show()


# from scipy.stats import ks_2samp, kruskal, mannwhitneyu

# # Store total hedging costs for different frequencies
# hedging_costs = {}

# for freq in [1, 2, 5, 10]:
#     bs_env = TradingEnv(continuous_action_flag=True, sabr_flag=False, 
#                         dg_random_seed=1, spread=0.01, num_contract=1, 
#                         init_ttm=10, trade_freq=1/freq, num_sim=10000,
#                         mu=0, vol=0.01, S=100, K=100, r=0, q=0)

#     delta_path = bs_env.delta_path * 100
#     delta_path = np.insert(delta_path, 0, 0, axis=1)
#     delta_h = np.diff(delta_path, axis=1)
#     transaction_costs = cost(delta_h, multiplier=5)
#     total_costs = np.sum(transaction_costs, axis=1)

#     hedging_costs[freq] = total_costs

# # Compute KS test results
# ks_results = {
#     "1 vs 2 trades/day": ks_2samp(hedging_costs[1], hedging_costs[2]),
#     "1 vs 5 trades/day": ks_2samp(hedging_costs[1], hedging_costs[5]),
#     "2 vs 5 trades/day": ks_2samp(hedging_costs[2], hedging_costs[5]), 
#     "5 vs 10 trades/day": ks_2samp(hedging_costs[5], hedging_costs[10])
# }

# # Convert results to a DataFrame
# ks_df = pd.DataFrame([
#     [key, round(value.statistic, 4), round(value.pvalue, 4)]
#     for key, value in ks_results.items()
# ], columns=["Comparison", "KS Statistic", "p-value"])

# # Save LaTeX table
# with open("ks_test_results.tex", "w") as f:
#     f.write(ks_df.to_latex(index=False, caption="Kolmogorov-Smirnov Test Results for Different Rebalancing Frequencies", 
#                            label="tab:ks_test", float_format="%.4f"))


# price_path = bs_env.path
# bs_delta = bs_env.delta_path
# bs_gamma = bs_env.gamma_path
# bs_theta = bs_env.theta_path



# plt.figure(figsize=(10,5))
# plt.plot(bs_delta[1], color = "orange", label = "Delta")
# plt.plot(bs_gamma[1], color = "blue", label = "Gamma")
# plt.plot(bs_theta[1], color = "green", label = "Theta")
# plt.xlabel("Time Steps")
# plt.ylabel("Values")
# plt.title("Out of Sample Simulation ")
# plt.legend()
# plt.savefig("Sensitivity OSS")
# plt.show



# plt.figure(figsize=(10,5))
# sns.kdeplot(bs_delta.flatten(), shade = True, color = "orange")
# plt.legend()
# plt.savefig("Delta_Distribution.png")
# plt.show()

# plt.figure(figsize=(10,5))
# sns.kdeplot(bs_gamma.flatten(), shade = True, color = "blue")
# plt.legend()
# plt.savefig("Gamma_Distribution.png")
# plt.show()

# plt.figure(figsize=(10,5))
# sns.kdeplot(bs_theta.flatten(), shade = True, color = "green")
# plt.legend()
# plt.savefig("Theta_Distribution.png")
# plt.show()

# strikes = [100, 50, 150]
# bs_deltas = {}  # Dictionary to store delta values for different strikes

# for i, strike in enumerate(strikes):
#     bs_env = TradingEnv(continuous_action_flag=True, sabr_flag=False, 
#                         dg_random_seed= None, spread=0.01, num_contract=1, 
#                         init_ttm=10, trade_freq=0.2, num_sim=10000,
#                         mu=0, vol=0.5, S=100, K=strike, r=0, q=0)
    
#     bs_deltas[f"bs_delta_{i}"] = bs_env.delta_path  # Store deltas in dictionary

# # Access stored deltas
# atm_delta = np.mean(bs_deltas["bs_delta_0"], axis = 0)
# itm_delta = np.mean(bs_deltas["bs_delta_1"], axis =0)
# otm_delta = np.mean(bs_deltas["bs_delta_2"], axis =0)

# plt.figure(figsize= (10,5))
# plt.plot(atm_delta, label = "At-the-Money Option", color = "orange")
# plt.plot(itm_delta, label = "In-the-Money Option", color = "blue")
# plt.plot(otm_delta, label = "Out-of-the-Money Option", color = "green")
# plt.legend()
# plt.xlabel("TimeStep")
# plt.ylabel("Value")
# plt.title("Progression of Delta during the life of the option")
# plt.savefig("Moneyness_delta.png")
# plt.show()

# Extract Black-Scholes Greeks
total_costs = transaction_costs.flatten()
bs_delta = bs_env.delta_path.flatten()  # Flatten to match dimensions
bs_gamma = bs_env.gamma_path.flatten()
bs_theta = bs_env.theta_path.flatten()


# Create a DataFrame for correlation analysis
df = pd.DataFrame({
    "Hedging Cost": total_costs,
    "BS Delta": bs_delta,
    "BS Gamma": bs_gamma,
    "BS Theta": bs_theta
})

# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Hedging Cost vs. Delta, Gamma, and Theta")
plt.savefig("HeatMap_HedgeCosts.png")
plt.show()