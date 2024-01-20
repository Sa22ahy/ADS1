import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import numpy as np

# Create a StandardScaler instance
scaler = StandardScaler()


def read_data(file_paths, selected_country, start_year, end_year):
    dataframes_list = []

    for path in file_paths:
        file_name = path.split('/')[-1].split('.')[0]
        df = pd.read_csv(path, skiprows=4)
        df = df.rename(columns={'Country Name': 'Country'})
        df = df.set_index('Country')
        df_selected_country = df.loc[selected_country, str(start_year):str(end_year)].transpose().reset_index()
        df_selected_country = df_selected_country.rename(columns={'index': 'Year', selected_country: file_name})
        dataframes_list.append(df_selected_country)

    # Concatenate all DataFrames based on the 'Year' column
    result_df = pd.concat(dataframes_list, axis=1)

    # Replace null values with the mean of each column
    result_df = result_df.apply(lambda col: col.fillna(col.mean()))

    return result_df


def calculate_silhouette_score(xy, n):
    """Calculates silhouette score for n clusters"""
    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)
    labels = kmeans.labels_
    # Calculate the silhouette score
    score = skmet.silhouette_score(xy, labels)
    return score


def plot_scatter_matrix(dataframe):
    """Plot scatter matrix for the given DataFrame"""
    pd.plotting.scatter_matrix(dataframe, figsize=(9.0, 9.0))
    plt.tight_layout()  # Helps to avoid overlap of labels
    plt.show()


def plot_heatmap(correlation_matrix):
    """Plot heatmap using Seaborn for the given correlation matrix"""
    plt.figure(figsize=(18, 18))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    # Rotate x-axis labels to 45 degrees
    plt.xticks(rotation=10, fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    # Save the plot with 300 dpi
    plt.savefig('Correlation.png', dpi=300)
    plt.title('Correlation Matrix on Indicators', fontweight='bold')
    plt.show()


def plot_clusters(df_cluster, n_clusters=2):
    """Plot clusters for given DataFrame and number of clusters"""
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    kmeans.fit(df_cluster)
    # Extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    plt.figure(figsize=(6.0, 6.0))
    # Scatter plot with colors selected using the cluster numbers
    plt.scatter(df_cluster.iloc[:, 0], df_cluster.iloc[:, 1], c=labels, cmap="tab10")
    # Colour map Accent selected to increase contrast between colors
    # Show cluster centres
    xc = cen[:, 0]
    yc = cen[:, 1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    # c = colour, s = size
    plt.xlabel(df_cluster.columns[0])
    plt.ylabel(df_cluster.columns[1])
    plt.title(f"Cluster between {df_cluster.columns[0]} and {df_cluster.columns[1]}")
    plt.savefig('Clusters.png', dpi=300)
    plt.show()


# Example usage
file_paths = ['Scientific articles.csv',
              "Research expenditure.csv", 'Patent residents.csv',
              'technology exports.csv', 'Charge intellectual property.csv']
selected_country = "Australia"
start_year = 2007
end_year = 2022

result_df = read_data(file_paths, selected_country, start_year, end_year)
# Remove the 'Year' column
result_df = result_df.drop('Year', axis=1)

# Apply Standard Scaling to all columns in the DataFrame
result_df_scaled = pd.DataFrame(scaler.fit_transform(result_df), columns=result_df.columns)

print(result_df_scaled)

# Scatter plot matrix
plot_scatter_matrix(result_df_scaled)

# Calculate the correlation matrix
correlation_matrix = result_df_scaled.corr()

# Create a heatmap using Seaborn
plot_heatmap(correlation_matrix)

# Select columns for clustering
df_cluster = result_df[['Scientific articles', 'technology exports']]

# Calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = calculate_silhouette_score(df_cluster, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

# Plot clusters
plot_clusters(df_cluster)
