# %% Import libraries
import os
os.environ['OMP_NUM_THREADS'] = '2'
import torch
import numpy as np
import dill as pickle
import pandas as pd
import sys
from tqdm import tqdm, trange
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# load portfolio data
sys.path.append('/Library')
from Library.V_H_relations import load_portfolio_data, gross_head, get_v_low
load_portfolio_data()
from Library.V_H_relations import r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, min_vol_low, target_vol_up, target_vol_low, target_head

# load preprocessed functions & data
with open('preprocess.pkl', 'rb') as f:
    h_vlow_coeff_lin, coefs_tur_lin, intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur,predict_q_linear_pump, h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound = pickle.load(f)

# %% Read day-ahead prices
def read_da_price(date, file_path="./Data/Belgium.csv"):
    """
    Input: "YYYY-MM-DD"
    """
    data = pd.read_csv(file_path)
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    filtered_data = data[data['Datetime (UTC)'].dt.date == pd.to_datetime(date).date()]
    return torch.tensor(filtered_data['Price (EUR/MWhe)'].values[:24], dtype=torch.float32)

def hourly_to_quarterly(tensor_data):
    return tensor_data.repeat_interleave(4)

if __name__ == "__main__":
    sample_date = "2022-01-01"
    DA_price_hour = read_da_price(sample_date)
    DA_price_quarter = hourly_to_quarterly(DA_price_hour)
    print(DA_price_hour)
    print(DA_price_quarter)

# %%

# Read full CSV data for Belgium day‐ahead prices
def load_full_da_prices(file_path="./Data/Belgium.csv"):
    data = pd.read_csv(file_path)
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    # Filter for year 2023
    data_2023 = data[data['Datetime (UTC)'].dt.year == 2023]
    return data_2023

# Group the data by day and return a dict of date->daily hourly price vector (length 24)
def group_by_day(data):
    daily_prices = {}
    # Group by the date part of the datetime
    data['Date'] = data['Datetime (UTC)'].dt.date
    grouped = data.groupby('Date')
    for day, group in grouped:
        # Ensure we have exactly 24 hourly values for a typical day
        # (if data is hourly and complete)
        if len(group) >= 24:
            # Sort by time just in case
            group_sorted = group.sort_values('Datetime (UTC)')
            prices = group_sorted['Price (EUR/MWhe)'].values[:24]
            daily_prices[day] = prices
    return daily_prices

# Find 10 typical days using clustering (KMeans)
def find_typical_days(daily_prices, n_clusters=10, random_state=42):
    # Build feature matrix: each row is a day (24 hourly prices)
    dates = list(daily_prices.keys())
    X = np.array([daily_prices[day] for day in dates])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    # Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)
    
    plt.figure(figsize=(10, 7))
    for i in range(n_clusters):
        cluster_points = X_pca[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', marker='X', label='Centroids')
    plt.title('KMeans Clustering of Daily Prices')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    
    # For each cluster, choose the day whose daily profile is closest to the centroid
    typical_days = {}
    for cluster in range(n_clusters):
        # Get indices for this cluster
        cluster_indices = [i for i, lab in enumerate(labels) if lab == cluster]
        if not cluster_indices:
            continue
        # Compute distances to the centroid
        distances = [np.linalg.norm(X[i] - centroids[cluster]) for i in cluster_indices]
        # Get the index of the minimum distance
        best_idx = cluster_indices[np.argmin(distances)]
        typical_day = dates[best_idx]
        typical_days[typical_day] = X[best_idx]
    return typical_days

# Find the optimal number of clusters using silhouette score
def find_optimal_clusters(daily_prices, max_clusters=15):
    dates = list(daily_prices.keys())
    X = np.array([daily_prices[day] for day in dates])
    
    best_n_clusters = 2
    best_score = -1
    scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
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
    plt.show()
    
    return best_n_clusters

# Find the day with the greatest intraday price deviation (max - min)
def find_max_deviation_day(daily_prices):
    max_dev = -np.inf
    chosen_day = None
    for day, prices in daily_prices.items():
        deviation = prices.max() - prices.min()
        if deviation > max_dev:
            max_dev = deviation
            chosen_day = day
    return chosen_day, daily_prices[chosen_day]

# Find the day with the highest overall (average) day‐ahead price
def find_highest_price_day(daily_prices):
    highest_avg = -np.inf
    chosen_day = None
    for day, prices in daily_prices.items():
        avg_price = prices.mean()
        if avg_price > highest_avg:
            highest_avg = avg_price
            chosen_day = day
    return chosen_day, daily_prices[chosen_day]

# Find the day with the lowest overall (average) day‐ahead price
def find_lowest_price_day(daily_prices):
    lowest_avg = np.inf
    chosen_day = None
    for day, prices in daily_prices.items():
        avg_price = prices.mean()
        if avg_price < lowest_avg:
            lowest_avg = avg_price
            chosen_day = day
    return chosen_day, daily_prices[chosen_day]

# Main procedure to build the database (12 days)
def build_database(file_path="./Data/Belgium.csv"):
    # Load the full data for 2023
    data_2023 = load_full_da_prices(file_path)
    
    # Group the data by day (each day a vector of hourly prices)
    daily_prices = group_by_day(data_2023)
    
    # Fixed number of clusters to 10
    n_clusters = 10
    print(f"Number of clusters: {n_clusters}")
    
    # 1. Find typical days via clustering with the fixed number of clusters
    typical_days = find_typical_days(daily_prices, n_clusters=n_clusters)
    
    # 2. Day with greatest intraday price deviation
    dev_day, dev_prices = find_max_deviation_day(daily_prices)
    
    # 3. Day with the highest overall day‐ahead price (using average)
    high_day, high_prices = find_highest_price_day(daily_prices)
    
    # 4. Day with the lowest overall day‐ahead price (using average)
    low_day, low_prices = find_lowest_price_day(daily_prices)
    
    # Create the database: a dictionary where keys are dates and values are price vectors
    database = {}
    
    # Add typical days (they are stored with date as key)
    for day, prices in typical_days.items():
        database[day] = {
            "type": "typical",
            "prices_hourly": prices,
            "prices_quarterly": np.repeat(prices, 4)  # convert hourly to quarterly
        }
    
    # Add the extreme days, using a different label
    # They might overlap with typical days. If so, we can update their type to include both labels.
    extreme_days = {
        dev_day: {"type": "max_deviation", "prices_hourly": dev_prices, "prices_quarterly": np.repeat(dev_prices, 4)},
        high_day: {"type": "highest_price", "prices_hourly": high_prices, "prices_quarterly": np.repeat(high_prices, 4)},
        low_day: {"type": "lowest_price", "prices_hourly": low_prices, "prices_quarterly": np.repeat(low_prices, 4)}
    }
    
    # Merge extreme days into the database (if a day already exists, update its label)
    for day, info in extreme_days.items():
        if day in database:
            # Append extreme label if needed
            database[day]["type"] += f" + {info['type']}"
        else:
            database[day] = info

    return database

# Save the database to a CSV file
def save_database_to_csv(database, file_path="./Data/database.csv"):
    rows = []
    for day, info in database.items():
        row = {
            "date": day,
            "type": info["type"],
            "prices_hourly": ",".join(map(str, info["prices_hourly"])),
            "prices_quarterly": ",".join(map(str, info["prices_quarterly"]))
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    find_optimal_clusters(group_by_day(load_full_da_prices()), max_clusters=20)

    db = build_database("./Data/Belgium.csv")
    # Display the keys (dates) and type of each chosen day
    for day, info in db.items():
        print(f"Date: {day}, Type: {info['type']}")
        # print hourly and quarterly prices:
        print("Hourly Prices:", info['prices_hourly'])
        # print("Quarterly Prices:", info['prices_quarterly'])

    # Plot all 13 price plots onto one plot with labeled lines
    plt.figure(figsize=(14, 8))
    for day, info in db.items():
        plt.plot(info['prices_hourly'], label=f"{day} ({info['type']})")
    
    plt.title('Day-Ahead Prices for Selected Days')
    plt.xlabel('Hour')
    plt.ylabel('Price (EUR/MWh)')
    plt.legend()
    plt.show()

    # Save the database to a CSV file
    save_database_to_csv(db, "./Data/price_database.csv")

    # # Profile the forward pass
    # pr = cProfile.Profile()
    # pr.enable()


# %%
