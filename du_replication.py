from utils import get_sim_path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

asset_sim, option_price, option_delta = get_sim_path(10, freq = 0.2, np_seed= 1, num_sim= 38500)

option_delta_scaled = option_delta * 100
option_delta_scaled = np.insert(option_delta_scaled, 0, 0, axis=1)

# Calculate the change in delta
delta_change = np.diff(option_delta_scaled, axis=1)

def cost(delta_h, multiplier):
    TickSize = 0.1
    return multiplier * TickSize * (np.abs(delta_h) + 0.01 * delta_h**2)

# Calculate the cost of hedging for each path

# #Plotting option delta
# plt.figure(figsize=(10, 5))
# for i in range(option_delta.shape[0]):
#     plt.plot(option_delta[i, :], label='Sim {}'.format(i+1))

# plt.xlabel('Time Steps')
# plt.ylabel('Option Delta')
# plt.title('Cao Option Delta')
# plt.legend()
# plt.grid(True)
# plt.show()

def cost(delta_h, multiplier):
    TickSize = 0.1
    return multiplier * TickSize * (np.abs(delta_h) + 0.01 * delta_h**2)

cost_path = cost(delta_change, 5)
total_cost = np.sum(cost_path, axis=1)

# Kernel density plot showing the total cost of delta hedging

plt.figure(figsize=(10, 5))
sns.kdeplot(total_cost, shade = True, color= "orange")
plt.xlabel("Total Cost")
plt.ylabel("Density")
plt.title('Kernel Density Plot of Total Cost')
plt.grid(True)
plt.savefig("Total_Cost_kde_plot.png")
plt.show()





