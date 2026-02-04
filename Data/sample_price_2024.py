# %% Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# Set environment variables to avoid threading issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["OPENBLAS_NUM_THREADS"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "Data", "price_figures")


# Load price data
def load_price_data(file_path=None, year=2024):
    """Load and filter price data for a specific year"""
    if file_path is None:
        file_path = os.path.join(REPO_ROOT, "Data", "Belgium.csv")
    data = pd.read_csv(file_path)
    data['Datetime (UTC)'] = pd.to_datetime(data['Datetime (UTC)'])
    # Filter for specified year
    data_year = data[data['Datetime (UTC)'].dt.year == year]
    return data_year

# Group the data by day
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

# K-medoids implementation with random initialization (no K-means)
def k_medoids(X, k, max_iter=100, nb_restarts=50):
    """
    Compute the k-medoids clustering with random initialization.
    
    Parameters:
    -----------
    X : np.ndarray
        The database of samples (n_samples, n_features)
    k : int
        Number of clusters to find
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
    # save the cost of each iteration
    ttl_distances, labels_l, medoids_idx_l = ([] for _ in range(3))
    
    for r in range(nb_restarts):
        # Simple random initialization
        np.random.seed(r)
        medoids_idx = np.random.choice(range(m), k, replace=False)
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
            
            # Update medoids
            for i in range(k):
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

# Visualize clusters
def visualize_clusters(X_scaled, labels, medoids, dates, medoids_idx, X_original):
    """Visualize clustering results with PCA and plot the medoids, excluding 2024-12-12 cluster"""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    try:
        # Identify which medoid index corresponds to 2024-12-12
        exclude_date = pd.Timestamp('2024-12-12').date()
        exclude_medoid_index = None
        for i, idx in enumerate(medoids_idx):
            if dates[idx] == exclude_date:
                exclude_medoid_index = i
                break
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        medoids_pca = pca.transform(medoids)
        
        # Scatter plot of clusters (excluding the one with 2024-12-12 as medoid)
        plt.figure(figsize=(12, 8))
        for i in range(len(medoids)):
            # Skip the cluster where 2024-12-12 is the medoid
            if i == exclude_medoid_index:
                continue
                
            # Get points in this cluster
            cluster_points = X_pca[labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
        
        # Mark medoids (excluding 2024-12-12)
        remaining_medoids_pca = np.delete(medoids_pca, exclude_medoid_index, axis=0) if exclude_medoid_index is not None else medoids_pca
        plt.scatter(remaining_medoids_pca[:, 0], remaining_medoids_pca[:, 1], s=300, c='red', marker='X', label='Medoids')
        
        plt.title('K-Medoids Clustering of Daily Prices (2024) - Excluding 2024-12-12')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "cluster_visualization_2024.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating PCA visualization: {e}")
    
    try:
        # Plot medoid price profiles using matplotlib's built-in color cycle and markers
        plt.figure(figsize=(14, 8))
        hours = range(24)
        
        # Get original (non-scaled) medoid values
        medoid_days = [dates[idx] for idx in medoids_idx]
        
        # Use matplotlib's default markers
        from matplotlib.pyplot import cycler
        import matplotlib.markers as mmarkers
        
        # Get available markers from matplotlib
        marker_list = list(mmarkers.MarkerStyle.markers.keys())
        # Filter for common markers that render well
        common_markers = [m for m in marker_list if m not in (None, '', ' ', '.', ',', 'None', 'none')]
        
        # Filter out 2024-12-12 from the visualization
        exclude_date = pd.Timestamp('2024-12-12').date()
        
        # Create a plot with default coloring but add markers
        marker_index = 0
        for i, day in enumerate(medoid_days):
            # Skip 2024-12-12
            if day == exclude_date:
                continue
                
            marker = common_markers[marker_index % len(common_markers)]
            plt.plot(hours, X_scaled[medoids_idx[i]], label=f"{day}", 
                     marker=marker, markevery=2)
            marker_index += 1
        
        plt.title('Day-Ahead Prices for Selected Medoid Days (2024) - Standardized')
        plt.xlabel('Hour')
        plt.ylabel('Standardized Price')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "medoid_profiles_scaled_2024.png"), dpi=300)
        plt.close()
        
        # Plot original price profiles with markers
        plt.figure(figsize=(14, 8))
        marker_index = 0
        for i, day in enumerate(medoid_days):
            # Skip 2024-12-12
            if day == exclude_date:
                continue
                
            marker = common_markers[marker_index % len(common_markers)]
            plt.plot(hours, X_original[medoids_idx[i]], label=f"{day}", 
                     marker=marker, markevery=2)
            marker_index += 1
        
        plt.title('Day-Ahead Prices for Selected Medoid Days (2024) - Original Scale')
        plt.xlabel('Hour')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "medoid_profiles_original_2024.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating price profile plots: {e}")
    
    try:
        # Cluster size distribution - exclude the cluster where 2024-12-12 is the medoid
        cluster_sizes = np.bincount(labels)
        
        # If we found the cluster to exclude, remove it from the plot
        if exclude_medoid_index is not None:
            cluster_sizes = np.delete(cluster_sizes, exclude_medoid_index)
            
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(cluster_sizes)), cluster_sizes)
        
        # Add labels on top of each bar
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{cluster_sizes[i]}', ha='center')
        
        plt.title('Number of Days in Each Cluster (2024) - Excluding 2024-12-12 Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Days')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "cluster_sizes_2024.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating cluster size plot: {e}")

# Build price database
def build_price_database(file_path="./Belgium.csv", year=2024, n_clusters=20):
    """
    Build a price database using k-medoids clustering for the year 2024
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file with price data
    year : int
        Year to filter data for
    n_clusters : int
        Number of clusters to find
        
    Returns:
    --------
    database : dict
        Dictionary containing the price database
    """
    # Load the full data for the specified year
    data_year = load_price_data(file_path, year)
    print(f"Loaded {len(data_year)} price points from {year}")
    
    # Group the data by day (each day a vector of hourly prices)
    daily_prices = group_by_day(data_year)
    print(f"Found {len(daily_prices)} complete days with 24 hourly values")
    
    # Create a list of days and their price arrays
    dates = list(daily_prices.keys())
    X = np.array([daily_prices[day] for day in dates])
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use k-medoids to find representative days
    print(f"Running k-medoids with {n_clusters} clusters...")
    labels, medoids, medoids_idx = k_medoids(
        X_scaled, 
        n_clusters,
        max_iter=100,
        nb_restarts=50
    )
    
    # Create the database: a dictionary where keys are dates and values are price vectors
    database = {}
    
    # Add medoid days to database
    for i, idx in enumerate(medoids_idx):
        day = dates[idx]
        database[day] = {
            "type": f"typical_{i}",
            "prices_hourly": daily_prices[day],
            "prices_quarterly": np.repeat(daily_prices[day], 4),  # convert hourly to quarterly
            "cluster_index": i
        }
    
    # Visualize the clustering results
    print("Generating visualizations...")
    visualize_clusters(X_scaled, labels, medoids, dates, medoids_idx, X)
    
    return database, labels, dates

# Save the database to a CSV file
def save_database_to_csv(database, file_path=None):
    """Save the database to a CSV file"""
    if file_path is None:
        file_path = os.path.join(REPO_ROOT, "Data", "price_data_2024.csv")
    rows = []
    exclude_date = "2024-12-12"
    for day, info in database.items():
        try:
            day_key = pd.to_datetime(day).strftime("%Y-%m-%d")
        except Exception:
            day_key = str(day)
        if day_key == exclude_date:
            continue
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

# Main execution
if __name__ == "__main__":
    try:
        # Build the database with 20 clusters
        n_clusters = 20
        print(f"Building database with {n_clusters} representative days from 2024...")
        database, labels, dates = build_price_database(
            os.path.join(REPO_ROOT, "Data", "Belgium.csv"),
            year=2024,
            n_clusters=n_clusters
        )
        
        # Display the keys (dates) and type of each chosen day
        print("\nRepresentative days in the database:")
        for day, info in database.items():
            print(f"Date: {day}, Type: {info['type']}")
        
        # Save the database to a CSV file
        save_database_to_csv(database)
        
        print("\nDatabase creation completed successfully!")
    except Exception as e:
        print(f"Error during execution: {e}")
# %%
