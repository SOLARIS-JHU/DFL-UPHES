# %% Import libraries
import os
os.environ['OMP_NUM_THREADS'] = '2'
import torch
import numpy as np
import dill as pickle
import pandas as pd
import sys
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

# %% Read day-ahead prices
def read_da_price(date, file_path="./Data/Belgium.csv"):
    """
    Input: "YYYY-MM-DD"
    Output: Tensor with hourly prices for that day
    """
    data = pd.read_csv(file_path)
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    filtered_data = data[data['Datetime (UTC)'].dt.date == pd.to_datetime(date).date()]
    return torch.tensor(filtered_data['Price (EUR/MWhe)'].values[:24], dtype=torch.float32)

def hourly_to_quarterly(tensor_data):
    """Convert hourly data to quarterly by repeating each value 4 times"""
    return tensor_data.repeat_interleave(4)

# K-medoids implementation adapted from CentroidAnalysis.py
def k_medoids_with_fixed_medoids(X, k, fixed_medoids_idx, max_iter=100, nb_restarts=50):
    """
    Compute the k-medoids clustering with some fixed medoids.
    
    Parameters:
    -----------
    X : np.ndarray
        The database of samples (n_samples, n_features)
    k : int
        Total number of medoids to find
    fixed_medoids_idx : list
        Indices of fixed medoids in the dataset
    max_iter : int, default=100
        Maximum number of iterations
    nb_restarts : int, default=50
        Number of random initializations
        
    Returns:
    --------
    labels : np.ndarray
        Cluster assignments
    medoids : np.ndarray
        Final medoid vectors
    medoids_idx : np.ndarray
        Indices of medoids in the dataset
    """
    m, n = X.shape
    # Get all the samples that are not fixed medoids
    remaining_idx = np.setdiff1d(range(m), fixed_medoids_idx)
    # save the cost of each iteration
    ttl_distances, labels_l, medoids_idx_l = ([] for _ in range(3))
    
    for r in range(nb_restarts):
        # Sample randomly k - nb of fixed medoids
        rng = np.random.default_rng(seed=r)
        random_medoids_idx = rng.choice(remaining_idx, k-len(fixed_medoids_idx), replace=False)
        # Initial medoids are the fixed ones and the random ones
        medoids_idx = np.concatenate((fixed_medoids_idx, random_medoids_idx))
        medoids = X[medoids_idx]
        labels = np.zeros(m, dtype=int)
        
        for _ in range(max_iter):
            # Compute distances from data points to medoids
            distances = cdist(X, medoids, 'euclidean')
            # Assign each data point to the closest medoid
            new_labels = np.argmin(distances, axis=1)
            
            # Check if labels changed
            if np.array_equal(labels, new_labels):
                break
                
            labels = new_labels
            old_medoids_idx = medoids_idx.copy()
            
            # Update only the non-fixed medoids
            for i in range(len(fixed_medoids_idx), k):
                cluster_idx = np.nonzero(labels == i)[0]
                if len(cluster_idx) > 0:
                    # Find the point in the cluster that minimizes the sum of distances
                    cluster_distances = cdist(X[cluster_idx], X[cluster_idx], 'euclidean')
                    costs = cluster_distances.sum(axis=1)
                    min_cost_idx = np.argmin(costs)
                    medoids_idx[i] = cluster_idx[min_cost_idx]
            
            # Get the new medoids
            medoids = X[medoids_idx]
            
            # Check if medoid indices changed
            if np.array_equal(old_medoids_idx, medoids_idx):
                break
        
        # Calculate final labels and total distance
        distances = cdist(X, medoids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        total_distance = np.sum(np.min(distances, axis=1))
        
        ttl_distances.append(total_distance)
        labels_l.append(labels)
        medoids_idx_l.append(medoids_idx)
    
    # Select the best result (lowest total distance)
    best_run = np.argmin(ttl_distances)
    print(f"Best run {best_run} - Total distance: {ttl_distances[best_run]:.4f}")
    
    best_labels = labels_l[best_run]
    best_medoids_idx = medoids_idx_l[best_run]
    best_medoids = X[best_medoids_idx]
    
    return best_labels, best_medoids, best_medoids_idx

# %% Read full CSV data for Belgium day‐ahead prices
def load_full_da_prices(file_path="./Data/Belgium.csv", year=2023):
    """Load and filter price data for a specific year"""
    data = pd.read_csv(file_path)
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    # Filter for specified year
    data_year = data[data['Datetime (UTC)'].dt.year == year]
    return data_year

# Group the data by day and return a dict of date->daily hourly price vector (length 24)
def group_by_day(data):
    """Group price data by day"""
    daily_prices = {}
    # Group by the date part of the datetime
    data['Date'] = data['Datetime (UTC)'].dt.date
    grouped = data.groupby('Date')
    for day, group in grouped:
        # Ensure we have exactly 24 hourly values for a typical day
        if len(group) >= 24:
            # Sort by time just in case
            group_sorted = group.sort_values('Datetime (UTC)')
            prices = group_sorted['Price (EUR/MWhe)'].values[:24]
            daily_prices[day] = prices
    return daily_prices

# Find extreme days: max deviation, highest price, lowest price
def find_extreme_days(daily_prices):
    """Find extreme price days that will be fixed medoids"""
    # Day with greatest intraday price deviation (max - min)
    max_dev = -np.inf
    dev_day = None
    for day, prices in daily_prices.items():
        deviation = np.max(prices) - np.min(prices)
        if deviation > max_dev:
            max_dev = deviation
            dev_day = day
    
    # Day with highest overall day-ahead price (using average)
    highest_avg = -np.inf
    high_day = None
    for day, prices in daily_prices.items():
        avg_price = np.mean(prices)
        if avg_price > highest_avg:
            highest_avg = avg_price
            high_day = day  # FIXED: This was wrongly assigned to high_day
    
    # Day with lowest overall day-ahead price (using average)
    lowest_avg = np.inf
    low_day = None
    for day, prices in daily_prices.items():
        avg_price = np.mean(prices)
        if avg_price < lowest_avg:
            lowest_avg = avg_price
            low_day = day
    
    extreme_days = {
        'max_deviation': {'day': dev_day, 'prices': daily_prices[dev_day]},
        'highest_price': {'day': high_day, 'prices': daily_prices[high_day]},
        'lowest_price': {'day': low_day, 'prices': daily_prices[low_day]}
    }
    
    return extreme_days

# Find the optimal number of clusters using silhouette score
def find_optimal_clusters(daily_prices, max_clusters=15):
    """Find optimal number of clusters using silhouette score"""
    dates = list(daily_prices.keys())
    X = np.array([daily_prices[day] for day in dates])
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    best_n_clusters = 2
    best_score = -1
    scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        # Use KMeans for silhouette analysis (it's faster than k-medoids for just evaluation)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, max_clusters + 1), scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig("./Data/silhouette_scores.png", dpi=300)
    plt.show()
    
    return best_n_clusters

# Main procedure to build the database using k-medoids
def build_database(file_path="./Data/Belgium.csv", year=2023, n_clusters=10):
    """
    Build a price database using k-medoids clustering
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file with price data
    year : int
        Year to filter data for
    n_clusters : int
        Total number of clusters to find
        
    Returns:
    --------
    database : dict
        Dictionary containing the price database
    """
    # Load the full data for the specified year
    data_year = load_full_da_prices(file_path, year)
    
    # Group the data by day (each day a vector of hourly prices)
    daily_prices = group_by_day(data_year)
    
    # Find extreme days (will be fixed medoids)
    extreme_days = find_extreme_days(daily_prices)
    
    # Create a list of days and their price arrays
    dates = list(daily_prices.keys())
    X = np.array([daily_prices[day] for day in dates])
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get indices of extreme days in the dataset
    extreme_indices = []
    for extreme_type, info in extreme_days.items():
        # Handle potential error where day isn't found
        if info['day'] is None:
            continue
        idx = dates.index(info['day'])
        extreme_indices.append(idx)
    
    print(f"Found {len(extreme_indices)} extreme days at indices: {extreme_indices}")
    
    # Use k-medoids with fixed extreme days
    labels, medoids, medoids_idx = k_medoids_with_fixed_medoids(
        X_scaled, 
        n_clusters, 
        extreme_indices,
        max_iter=100,
        nb_restarts=50
    )
    
    # Convert medoids back to original scale
    medoids_original = scaler.inverse_transform(medoids)
    
    # Create the database: a dictionary where keys are dates and values are price vectors
    database = {}
    
    # Prepare labels for each medoid
    medoid_labels = []
    for i, idx in enumerate(medoids_idx):
        if idx in extreme_indices:
            extreme_idx = extreme_indices.index(idx)
            extreme_type = list(extreme_days.keys())[extreme_idx]
            label = extreme_type
        else:
            label = f"typical_{i}"
        medoid_labels.append(label)
    
    # Add typical days to database
    for i, idx in enumerate(medoids_idx):
        day = dates[idx]
        label = medoid_labels[i]
        database[day] = {
            "type": label,
            "prices_hourly": daily_prices[day],
            "prices_quarterly": np.repeat(daily_prices[day], 4),  # convert hourly to quarterly
            "cluster_index": i
        }
    
    # Visualize the clustering results
    visualize_clusters(X_scaled, labels, medoids, dates, medoids_idx, medoid_labels)
    
    return database, labels, dates, medoid_labels

def visualize_clusters(X_scaled, labels, medoids, dates, medoids_idx, medoid_labels):
    """Visualize clustering results with PCA and plot the medoids"""
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    medoids_pca = pca.transform(medoids)
    
    # Scatter plot of clusters
    plt.figure(figsize=(12, 8))
    for i in range(len(medoids)):
        # Get points in this cluster
        cluster_points = X_pca[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}: {medoid_labels[i]}')
    
    # Mark medoids
    plt.scatter(medoids_pca[:, 0], medoids_pca[:, 1], s=300, c='red', marker='X', label='Medoids')
    
    plt.title('K-Medoids Clustering of Daily Prices')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./Data/cluster_visualization.png", dpi=300)
    plt.show()
    
    # Plot medoid price profiles
    plt.figure(figsize=(14, 8))
    hours = range(24)
    
    # Get original (non-scaled) medoid values
    medoid_days = [dates[idx] for idx in medoids_idx]
    
    for i, day in enumerate(medoid_days):
        plt.plot(hours, X_scaled[medoids_idx[i]], label=f"{day} ({medoid_labels[i]})")
    
    plt.title('Day-Ahead Prices for Selected Medoid Days')
    plt.xlabel('Hour')
    plt.ylabel('Standardized Price')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./Data/medoid_profiles_scaled.png", dpi=300)
    plt.show()
    
    # Plot actual price profiles (non-scaled)
    plt.figure(figsize=(14, 8))
    X_original = np.array([daily_prices[dates[i]] for i in range(len(dates))])
    
    for i, day in enumerate(medoid_days):
        plt.plot(hours, X_original[medoids_idx[i]], label=f"{day} ({medoid_labels[i]})")
    
    plt.title('Day-Ahead Prices for Selected Medoid Days (Original Scale)')
    plt.xlabel('Hour')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./Data/medoid_profiles_original.png", dpi=300)
    plt.show()
    
    # Cluster size distribution
    cluster_sizes = np.bincount(labels)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(cluster_sizes)), cluster_sizes)
    
    # Add labels on top of each bar
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{cluster_sizes[i]}', ha='center')
    
    plt.title('Number of Days in Each Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Days')
    plt.xticks(range(len(cluster_sizes)), [f"{i}: {label}" for i, label in enumerate(medoid_labels)], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./Data/cluster_sizes.png", dpi=300)
    plt.show()

# Save the database to a CSV file
def save_database_to_csv(database, file_path="./Data/price_database.csv"):
    """Save the database to a CSV file"""
    rows = []
    for day, info in database.items():
        row = {
            "date": day,
            "type": info["type"],
            "cluster_index": info["cluster_index"],
            "prices_hourly": ",".join(map(str, info["prices_hourly"])),
            "prices_quarterly": ",".join(map(str, info["prices_quarterly"]))
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)
    print(f"Database saved to {file_path}")

if __name__ == "__main__":
    # Check if the directory exists, if not create it
    os.makedirs("./Data", exist_ok=True)
    
    # Example usage:
    # Load a sample date for initial testing
    sample_date = "2022-01-01"
    DA_price_hour = read_da_price(sample_date)
    DA_price_quarter = hourly_to_quarterly(DA_price_hour)
    print("Sample day-ahead prices (hourly):")
    print(DA_price_hour)
    print("\nSample day-ahead prices (quarterly):")
    print(DA_price_quarter)
    
    # Find optimal number of clusters
    print("\nFinding optimal number of clusters...")
    data_year = load_full_da_prices("./Data/Belgium.csv", year=2023)
    daily_prices = group_by_day(data_year)
    optimal_clusters = find_optimal_clusters(daily_prices, max_clusters=50)
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Build the database with the optimal number of clusters or a user-specified number
    n_clusters = 15  # Can be changed to optimal_clusters if desired
    print(f"\nBuilding database with {n_clusters} clusters...")
    database, labels, dates, medoid_labels = build_database("./Data/Belgium.csv", year=2023, n_clusters=n_clusters)
    
    # Display the keys (dates) and type of each chosen day
    print("\nMedoid days in the database:")
    for day, info in database.items():
        print(f"Date: {day}, Type: {info['type']}")
        # Uncomment to print hourly prices:
        # print("Hourly Prices:", info['prices_hourly'])
        # print("Quarterly Prices:", info['prices_quarterly'])
    
    # Save the database to a CSV file
    save_database_to_csv(database, "./Data/price_database_2.csv")
    
    print("\nDatabase creation completed successfully!")