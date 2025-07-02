# %% Belgium Price Database Matcher
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_new_price_data(file_path="./Data/price_data_2024.csv"):
    """
    Load new price data for scheduling validation.
    
    Args:
        file_path: Path to the CSV file with new price data
        
    Returns:
        dict: Dictionary with date strings as keys and price tensors as values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check column names from the first line
        if 'date' not in df.columns or 'cluster_index' not in df.columns or 'prices_hourly' not in df.columns:
            # Try to handle the case where column headers might be different
            if len(df.columns) >= 3:
                # Assume first column is date, third column has hourly prices
                df.columns = ['date', 'cluster_index', 'prices_hourly']
            else:
                raise ValueError(f"Expected columns 'date', 'cluster_index', 'prices_hourly' but got {df.columns}")
        
        # Dictionary to store price data by date
        price_data = {}
        
        # Process each row
        for _, row in df.iterrows():
            date_str = row['date']
            prices_str = row['prices_hourly']
            
            # Parse the prices (attempting different delimiter formats)
            try:
                # First try splitting by comma
                prices = [float(p) for p in prices_str.split(',')]
            except:
                try:
                    # If that fails, try splitting by semicolon
                    prices = [float(p) for p in prices_str.split(';')]
                except:
                    # If that fails too, try to interpret as a list-like string
                    prices_str = prices_str.strip('[]')
                    prices = [float(p) for p in prices_str.split()]
            
            # Ensure we have 24 hours of data
            if len(prices) != 24:
                print(f"Warning: Date {date_str} has {len(prices)} price values instead of 24")
                # Pad or truncate as needed
                if len(prices) < 24:
                    prices.extend([prices[-1]] * (24 - len(prices)))  # Pad with last value
                else:
                    prices = prices[:24]  # Truncate
            
            # Convert to tensor
            price_tensor = torch.tensor(prices, dtype=torch.float32, device=device)
            
            # Add to dictionary
            price_data[date_str] = price_tensor
        
        print(f"Successfully loaded price data for {len(price_data)} days.")
        return price_data
    
    except Exception as e:
        print(f"Error loading new price data: {e}")
        return None

def load_belgium_price_database(file_path="./Data/Belgium.csv", end_date="2023-12-31"):
    """
    Load the Belgium price database and organize it by date.
    
    Args:
        file_path (str): Path to Belgium.csv file
        end_date (str): Last date to include (format: "YYYY-MM-DD")
        
    Returns:
        dict: Dictionary with date strings as keys and price arrays as values
    """
    try:
        print("Loading Belgium price database...")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print column information for debugging
        print(f"Columns in file: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        # Handle different possible column name variations
        datetime_col = None
        price_col = None
        
        # Look for datetime column
        for col in df.columns:
            if 'datetime' in col.lower() and 'utc' in col.lower():
                datetime_col = col
                break
            elif 'datetime' in col.lower() and 'local' not in col.lower():
                datetime_col = col
                break
        
        # Look for price column
        for col in df.columns:
            if 'price' in col.lower():
                price_col = col
                break
        
        if datetime_col is None or price_col is None:
            raise ValueError(f"Could not find datetime or price columns. Available columns: {df.columns.tolist()}")
        
        print(f"Using datetime column: {datetime_col}")
        print(f"Using price column: {price_col}")
        
        # Convert datetime column
        df['datetime'] = pd.to_datetime(df[datetime_col])
        df['price'] = pd.to_numeric(df[price_col], errors='coerce')
        
        # Filter to only Belgium data (if multiple countries)
        if 'Country' in df.columns:
            df = df[df['Country'] == 'Belgium']
        
        # Filter by end date
        end_datetime = pd.to_datetime(end_date)
        df = df[df['datetime'] <= end_datetime]
        
        # Create date column for grouping
        df['date'] = df['datetime'].dt.date
        
        # Group by date and create hourly price profiles
        price_by_date = {}
        
        print("Processing daily price profiles...")
        grouped = df.groupby('date')
        
        for date, group in tqdm(grouped, desc="Processing dates"):
            # Sort by datetime to ensure correct hourly order
            group = group.sort_values('datetime')
            
            # Extract hourly prices
            hourly_prices = group['price'].values
            
            # Only include days with complete 24-hour data
            if len(hourly_prices) == 24:
                date_str = date.strftime('%Y-%m-%d')
                price_by_date[date_str] = hourly_prices
            elif len(hourly_prices) == 23:
                # Handle daylight saving time transitions (23 hours)
                # Duplicate the last hour to make it 24 hours
                hourly_prices = np.append(hourly_prices, hourly_prices[-1])
                date_str = date.strftime('%Y-%m-%d')
                price_by_date[date_str] = hourly_prices
            elif len(hourly_prices) == 25:
                # Handle daylight saving time transitions (25 hours)
                # Remove the duplicate hour (usually the 2nd occurrence)
                hourly_prices = hourly_prices[:24]
                date_str = date.strftime('%Y-%m-%d')
                price_by_date[date_str] = hourly_prices
            else:
                print(f"Warning: Date {date} has {len(hourly_prices)} hours, skipping...")
                continue
        
        print(f"Successfully loaded {len(price_by_date)} complete days from Belgium database")
        print(f"Date range: {min(price_by_date.keys())} to {max(price_by_date.keys())}")
        
        return price_by_date
    
    except Exception as e:
        print(f"Error loading Belgium price database: {e}")
        return None

def find_closest_date_belgium(new_price, belgium_price_db, distance_metric='euclidean'):
    """
    Find the date in Belgium database with the most similar price signal.
    
    Args:
        new_price: Tensor of shape [24] with hourly prices
        belgium_price_db: Dictionary from load_belgium_price_database
        distance_metric: 'euclidean', 'manhattan', or 'cosine'
        
    Returns:
        tuple: (closest_date_str, min_distance, closest_price_array)
    """
    if not belgium_price_db:
        raise ValueError("Belgium price database is empty or None")
    
    closest_date = None
    min_distance = float('inf')
    closest_price = None
    
    # Convert new_price to numpy for easier computation
    if isinstance(new_price, torch.Tensor):
        new_price_np = new_price.detach().cpu().numpy()
    else:
        new_price_np = np.array(new_price)
    
    print(f"Searching through {len(belgium_price_db)} days in Belgium database...")
    
    for date_str, price_array in tqdm(belgium_price_db.items(), desc="Finding closest match"):
        
        # Calculate distance based on selected metric
        if distance_metric == 'euclidean':
            distance = np.linalg.norm(new_price_np - price_array)
        elif distance_metric == 'manhattan':
            distance = np.sum(np.abs(new_price_np - price_array))
        elif distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            norm_new = np.linalg.norm(new_price_np)
            norm_hist = np.linalg.norm(price_array)
            if norm_new == 0 or norm_hist == 0:
                distance = 1.0  # Maximum cosine distance
            else:
                cosine_sim = np.dot(new_price_np, price_array) / (norm_new * norm_hist)
                distance = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        if distance < min_distance:
            min_distance = distance
            closest_date = date_str
            closest_price = price_array.copy()
    
    return closest_date, min_distance, closest_price

def find_top_k_closest_dates_belgium(new_price, belgium_price_db, k=5, distance_metric='euclidean'):
    """
    Find the top k most similar dates in Belgium database.
    
    Args:
        new_price: Tensor of shape [24] with hourly prices
        belgium_price_db: Dictionary from load_belgium_price_database
        k: Number of top matches to return
        distance_metric: 'euclidean', 'manhattan', or 'cosine'
        
    Returns:
        list: List of tuples (date_str, distance, price_array) sorted by distance
    """
    if not belgium_price_db:
        raise ValueError("Belgium price database is empty or None")
    
    distances = []
    
    # Convert new_price to numpy for easier computation
    if isinstance(new_price, torch.Tensor):
        new_price_np = new_price.detach().cpu().numpy()
    else:
        new_price_np = np.array(new_price)
    
    print(f"Finding top {k} matches from {len(belgium_price_db)} days...")
    
    for date_str, price_array in tqdm(belgium_price_db.items(), desc="Calculating distances"):
        
        # Calculate distance based on selected metric
        if distance_metric == 'euclidean':
            distance = np.linalg.norm(new_price_np - price_array)
        elif distance_metric == 'manhattan':
            distance = np.sum(np.abs(new_price_np - price_array))
        elif distance_metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            norm_new = np.linalg.norm(new_price_np)
            norm_hist = np.linalg.norm(price_array)
            if norm_new == 0 or norm_hist == 0:
                distance = 1.0  # Maximum cosine distance
            else:
                cosine_sim = np.dot(new_price_np, price_array) / (norm_new * norm_hist)
                distance = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        distances.append((date_str, distance, price_array.copy()))
    
    # Sort by distance and return top k
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def plot_price_comparison(new_price, belgium_matches, new_date_str="New Date", 
                         save_path=None, distance_metric='euclidean'):
    """
    Plot comparison between new price and top Belgium matches.
    
    Args:
        new_price: New price array/tensor
        belgium_matches: List of tuples from find_top_k_closest_dates_belgium
        new_date_str: Label for the new price date
        save_path: Path to save the plot (optional)
        distance_metric: Distance metric used for the title
    """
    # Convert new_price to numpy if needed
    if isinstance(new_price, torch.Tensor):
        new_price_np = new_price.detach().cpu().numpy()
    else:
        new_price_np = np.array(new_price)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: New price vs closest match
    hours = range(24)
    
    ax1.plot(hours, new_price_np, 'b-', linewidth=3, label=f'{new_date_str}', marker='o')
    
    if belgium_matches:
        closest_date, closest_distance, closest_price = belgium_matches[0]
        ax1.plot(hours, closest_price, 'r--', linewidth=2, 
                label=f'Closest: {closest_date} (dist: {closest_distance:.2f})', marker='s')
    
    ax1.set_title(f'Price Comparison: {new_date_str} vs Closest Belgium Match ({distance_metric} distance)')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top 5 matches
    ax2.plot(hours, new_price_np, 'b-', linewidth=3, label=f'{new_date_str}', marker='o')
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (date_str, distance, price_array) in enumerate(belgium_matches[:5]):
        ax2.plot(hours, price_array, '--', linewidth=1.5, color=colors[i % len(colors)],
                label=f'{i+1}. {date_str} (dist: {distance:.2f})', alpha=0.7)
    
    ax2.set_title(f'Top 5 Most Similar Days from Belgium Database')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Price (EUR/MWh)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig

def comprehensive_belgium_validation(new_price_data, belgium_db_path="./Data/Belgium.csv", 
                                   distance_metric='euclidean', top_k=5):
    """
    Perform comprehensive validation using Belgium database for all new price data.
    
    Args:
        new_price_data: Dictionary with new price data (from load_new_price_data)
        belgium_db_path: Path to Belgium.csv file
        distance_metric: Distance metric to use
        top_k: Number of top matches to find for each date
        
    Returns:
        dict: Results for each new date
    """
    # Load Belgium database
    belgium_price_db = load_belgium_price_database(belgium_db_path)
    if not belgium_price_db:
        print("Failed to load Belgium database")
        return None
    
    # Create results directory
    results_dir = Path("./validation_results/belgium_matching")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Store all results
    all_results = {}
    
    # Create summary CSV
    summary_file = results_dir / f"belgium_matching_summary_{distance_metric}.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'New_Date', 'Closest_Belgium_Date', 'Distance', 
            'Belgium_Year', 'Belgium_Month', 'Belgium_Day_of_Week',
            'New_Price_Mean', 'Belgium_Price_Mean',
            'New_Price_Std', 'Belgium_Price_Std',
            'New_Price_Min', 'Belgium_Price_Min',
            'New_Price_Max', 'Belgium_Price_Max'
        ])
    
    print(f"Processing {len(new_price_data)} new dates...")
    
    for new_date, new_price in tqdm(new_price_data.items(), desc="Processing new dates"):
        
        # Find top k matches
        top_matches = find_top_k_closest_dates_belgium(
            new_price, belgium_price_db, k=top_k, distance_metric=distance_metric
        )
        
        if not top_matches:
            print(f"No matches found for {new_date}")
            continue
        
        # Get the closest match
        closest_date, closest_distance, closest_price = top_matches[0]
        
        # Calculate statistics
        new_price_np = new_price.detach().cpu().numpy() if isinstance(new_price, torch.Tensor) else np.array(new_price)
        
        # Parse Belgium date for additional info
        belgium_datetime = datetime.strptime(closest_date, '%Y-%m-%d')
        
        # Store results
        all_results[new_date] = {
            'top_matches': top_matches,
            'closest_date': closest_date,
            'closest_distance': closest_distance,
            'closest_price': closest_price,
            'new_price': new_price_np,
            'statistics': {
                'new_price_mean': float(np.mean(new_price_np)),
                'new_price_std': float(np.std(new_price_np)),
                'new_price_min': float(np.min(new_price_np)),
                'new_price_max': float(np.max(new_price_np)),
                'belgium_price_mean': float(np.mean(closest_price)),
                'belgium_price_std': float(np.std(closest_price)),
                'belgium_price_min': float(np.min(closest_price)),
                'belgium_price_max': float(np.max(closest_price))
            }
        }
        
        # Generate comparison plot
        plot_path = results_dir / f"{new_date}_belgium_comparison.png"
        plot_price_comparison(new_price, top_matches, new_date, plot_path, distance_metric)
        plt.close()
        
        # Add to summary CSV
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                new_date, closest_date, f"{closest_distance:.2f}",
                belgium_datetime.year, belgium_datetime.month, belgium_datetime.strftime('%A'),
                f"{np.mean(new_price_np):.2f}", f"{np.mean(closest_price):.2f}",
                f"{np.std(new_price_np):.2f}", f"{np.std(closest_price):.2f}",
                f"{np.min(new_price_np):.2f}", f"{np.min(closest_price):.2f}",
                f"{np.max(new_price_np):.2f}", f"{np.max(closest_price):.2f}"
            ])
        
        print(f"Processed {new_date}: closest match is {closest_date} (distance: {closest_distance:.2f})")
    
    # Generate overall analysis
    generate_belgium_analysis(all_results, results_dir, distance_metric)
    
    print(f"\nBelgium validation completed!")
    print(f"Results saved to: {results_dir}")
    print(f"Summary CSV: {summary_file}")
    
    return all_results

def generate_belgium_analysis(all_results, results_dir, distance_metric):
    """Generate analysis of Belgium matching results."""
    
    # Extract statistics
    distances = [result['closest_distance'] for result in all_results.values()]
    years = [datetime.strptime(result['closest_date'], '%Y-%m-%d').year for result in all_results.values()]
    months = [datetime.strptime(result['closest_date'], '%Y-%m-%d').month for result in all_results.values()]
    
    # Create analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distance distribution
    ax1.hist(distances, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title(f'Distribution of {distance_metric.title()} Distances')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.2f}')
    ax1.legend()
    
    # Year distribution
    year_counts = pd.Series(years).value_counts().sort_index()
    ax2.bar(year_counts.index, year_counts.values, alpha=0.7)
    ax2.set_title('Distribution of Closest Match Years')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Matches')
    ax2.tick_params(axis='x', rotation=45)
    
    # Month distribution
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_counts = pd.Series(months).value_counts().sort_index()
    ax3.bar(range(1, 13), [month_counts.get(i, 0) for i in range(1, 13)], alpha=0.7)
    ax3.set_title('Distribution of Closest Match Months')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Number of Matches')
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(month_names, rotation=45)
    
    # Distance vs year
    years_for_scatter = [datetime.strptime(result['closest_date'], '%Y-%m-%d').year for result in all_results.values()]
    ax4.scatter(years_for_scatter, distances, alpha=0.7)
    ax4.set_title('Distance vs Match Year')
    ax4.set_xlabel('Year of Closest Match')
    ax4.set_ylabel(f'{distance_metric.title()} Distance')
    
    plt.tight_layout()
    plt.savefig(results_dir / f"belgium_analysis_{distance_metric}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis summary
    summary = {
        'total_dates_processed': len(all_results),
        'distance_metric': distance_metric,
        'distance_statistics': {
            'mean': float(np.mean(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'median': float(np.median(distances))
        },
        'year_distribution': year_counts.to_dict(),
        'month_distribution': month_counts.to_dict(),
        'date_range_belgium': {
            'earliest_match': min([result['closest_date'] for result in all_results.values()]),
            'latest_match': max([result['closest_date'] for result in all_results.values()])
        }
    }
    
    with open(results_dir / f"belgium_analysis_summary_{distance_metric}.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Analysis completed and saved to {results_dir}")

# Example usage function
def test_belgium_matching():
    """Test function to demonstrate Belgium price matching."""
    
    # Load new price data (your existing function)
    new_price_data = load_new_price_data("./Data/price_data_2024.csv")
    
    if not new_price_data:
        print("Could not load new price data")
        return
    
    # Run comprehensive Belgium validation
    results = comprehensive_belgium_validation(
        new_price_data, 
        belgium_db_path="./Data/Belgium.csv",
        distance_metric='euclidean',  # or 'manhattan', 'cosine'
        top_k=5
    )
    
    return results

# # Uncomment to run the test
if __name__ == "__main__":
    test_belgium_matching()