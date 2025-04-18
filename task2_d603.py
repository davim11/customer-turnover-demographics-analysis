import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns

#D3-4: Clean the dataset and provide a copy of the cleaned dataset
# Step 1: Load the dataset
data = pd.read_csv('churn_clean.csv')

# Step 2: Select relevant variables
features = data[['Tenure', 'Bandwidth_GB_Year', 'MonthlyCharge', 'Outage_sec_perweek']]

# Step 3: Check and handle missing values
print(features.isnull().sum())
features = features.fillna(features.mean())

# Step 4: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Save the cleaned dataset
cleaned_data = pd.DataFrame(scaled_features, columns=features.columns)
cleaned_data.to_csv('task2_cleaned_churn_data.csv', index=False)

#E1: Determine the optimial number of clusters
# Load the cleaned dataset
cleaned_data = pd.read_csv('task2_cleaned_churn_data.csv')

# Calculate silhouette scores for different numbers of clusters
k_range = range(2, 11)  # Start at 2 since silhouette needs at least 2 clusters
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # n_init=10 to suppress warnings
    cluster_labels = kmeans.fit_predict(cleaned_data)
    score = silhouette_score(cleaned_data, cluster_labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.4f}")

# Plot the silhouette scores
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Average Silhouette Score')
plt.show()

# Find the optimal k
optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
max_score = max(silhouette_scores)
print(f"\nOptimal number of clusters: k={optimal_k} with Silhouette Score = {max_score:.4f}")

# F1: Visualize Clusters
# Run k-means with k=2
kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(cleaned_data)
print("Cluster Sizes:\n", cleaned_data['Cluster'].value_counts())
print("\nCluster Means:\n", cleaned_data.groupby('Cluster').mean())

# Visualize (e.g., Tenure vs. Bandwidth_GB_Year)
sns.scatterplot(x=cleaned_data['Tenure'], y=cleaned_data['Bandwidth_GB_Year'], hue=cleaned_data['Cluster'], palette='deep')
plt.title('Customer Segments (k=2)')
plt.xlabel('Tenure (standardized)')
plt.ylabel('Bandwidth_GB_Year (standardized)')
plt.show()