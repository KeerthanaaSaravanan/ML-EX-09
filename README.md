# Implementation of Customer Segmentation Using K-Means Clustering
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO.: 212223240070</H3>
<H3>EX. NO.9</H3>
<H3>DATE: 28.10.24</H3>

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load the Data**  
   Import the dataset to start the clustering analysis process.

2. **Explore the Data**  
   Analyze the dataset to understand distributions, patterns, and key characteristics.

3. **Select Relevant Features**  
   Identify the most informative features to improve clustering accuracy and relevance.

4. **Preprocess the Data**  
   Clean and scale the data to prepare it for clustering.

5. **Determine Optimal Number of Clusters**  
   Use techniques like the elbow method to find the ideal number of clusters.

6. **Train the Model with K-Means Clustering**  
   Apply the K-Means algorithm to group data points into clusters based on similarity.

7. **Analyze and Visualize Clusters**  
   Examine and visualize the resulting clusters to interpret patterns and relationships.
   
## Program:
```py
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/CustomerData.csv"
data = pd.read_csv(url)

# Step 2: Select relevant features for clustering
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Identify the optimal number of clusters using the Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Step 5: Train the K-Means model with the optimal number of clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Step 6: Add the cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Step 7: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis', s=100)
plt.title('Customer Segmentation based on Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/e8759283-0114-4bd0-91ec-9c5057162c72)
![image](https://github.com/user-attachments/assets/69009c78-2992-4e54-b861-9bf60764a124)


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
