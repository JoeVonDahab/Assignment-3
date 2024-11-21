import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
from sklearn.metrics import silhouette_score# from sklearn.metrics import  
from cluster.kmeans import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    kmeans = KMeans(k=3, metric='euclidean', max_iter=30000, tol=0.0001)
    
    # Fit model
    kmeans.fit(og_iris)

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    predictions = kmeans.predict(df)
    
    # You can choose which scoring method you'd like to use here:
    silhouette_avg = silhouette_score(df, predictions)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(df, predictions, kmeans.get_centroids(), silhouette_avg)

    
    # Try different numbers of clusters

    inertias = []
    silhouette_scores = []
    k_values = range(2, 11)
    for k in k_values:
        kmeans_temp = KMeans(k=k, metric='euclidean', max_iter=300, tol=0.001)
        kmeans_temp.fit(og_iris)
        inertias.append(kmeans_temp.get_error())
        temp_predictions = kmeans_temp.predict(og_iris)
        silhouette_avg = silhouette_score(og_iris, temp_predictions)
        silhouette_scores.append(silhouette_avg)

    
    # Plot the elbow plot

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, marker='o', linestyle='--', label='Inertia (SSE)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Score')
    plt.title('Elbow Plot for Optimal K')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Silhouette Scores
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', label='Silhouette Score')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for KMeans')
    plt.legend()
    plt.grid()
    plt.show()

    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 

    3
    
    Reasoning: 
    
    best k is 3
    
    
    """

    
if __name__ == "__main__":
    main()