import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the CSV file into a Pandas DataFrame
data = pd.read_csv('graph_data_modified_BlueTableWrapper.csv', header=None, names=['timesteps', 'reward'])

# Read the CSV file into a Pandas DataFrame
data = data.rolling(window=10, center=True).mean()
# Plot the data using Seaborn
sns.lineplot(x='timesteps', y='reward', data=data)

# Show the plot
plt.show()


def generate_graph(name,period):
    data = pd.read_csv(name, header=None, names=['timesteps', 'reward'])

    # Read the CSV file into a Pandas DataFrame
    #smoothed_data = data.rolling(window=2, center=True).mean()

    #if period > 10:
    #    smoothed_data = data.rolling(window=5, center=True).mean()
    # Plot the data using Seaborn
    sns.lineplot(x='timesteps', y='reward', data=data)

    # Show the plot
    plt.show()
