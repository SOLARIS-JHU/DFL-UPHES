#!/usr/bin/env python3
"""
Historical Price Database Generator for UPHES Scheduling

DESCRIPTION:
This script creates historical price databases by finding the most similar days 
to 2024 test samples using multiple similarity metrics (Euclidean distance, 
Pearson correlation, and Cosine similarity). Generates synthetic operational 
data compatible with the recursive linearization pipeline for training 
decision-focused learning models.

INPUT FILES:
- Belgium.csv                     : Historical Belgium electricity price data (pre-2024)
                                   Required columns: 'Datetime (UTC)', 'Price (EUR/MWhe)'
- price_data_2024.csv             : 2024 test price data from k-medoids sampling
                                   Required columns: 'date', 'prices_hourly'

OUTPUT FILES (same directory):
- historical_database_euclidean.csv     : Database using Euclidean distance similarity
- historical_database_pearson.csv       : Database using Pearson correlation similarity  
- historical_database_cosine.csv        : Database using Cosine similarity
- euclidean_similarity_all_days.png     : Visualization comparing all 20 test days (Euclidean)
- pearson_similarity_all_days.png       : Visualization comparing all 20 test days (Pearson)
- cosine_similarity_all_days.png        : Visualization comparing all 20 test days (Cosine)

DATABASE STRUCTURE (compatible with recursive_linearization_pipeline1.py):
Each CSV contains columns: Time, Power, Head, Flow, Price, Date
- Time: Hour of day (0-23)
- Power: Synthetic power output (MW) based on price patterns
- Head: Synthetic hydraulic head (m) with realistic variations
- Flow: Synthetic water flow (m³/s) correlated with power
- Price: Historical Belgium electricity prices (EUR/MWh)
- Date: Date string (YYYY-MM-DD) of similar historical day

SIMILARITY METHODS:
1. Euclidean Distance     : Finds days with similar absolute price levels
2. Pearson Correlation    : Finds days with similar price patterns/shapes  
3. Cosine Similarity      : Finds days with identical patterns regardless of scale
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class SimilarityMethods:
    """Class containing different similarity calculation methods"""
    
    @staticmethod
    def euclidean_distance(series1: np.ndarray, series2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two price series
        
        Args:
            series1, series2: 24-hour price arrays
            
        Returns:
            Euclidean distance (lower = more similar)
        """
        return np.sqrt(np.sum((series1 - series2) ** 2))
    
    @staticmethod
    def pearson_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
        """
        Calculate Pearson correlation between two price series
        
        Args:
            series1, series2: 24-hour price arrays
            
        Returns:
            1 - |correlation| to get distance (lower = more similar)
        """
        correlation, _ = pearsonr(series1, series2)
        # Convert correlation to distance: higher correlation = lower distance
        return 1 - abs(correlation)
    
    @staticmethod
    def cosine_similarity(series1: np.ndarray, series2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two price series
        
        Args:
            series1, series2: 24-hour price arrays
            
        Returns:
            1 - cosine_similarity to get distance (lower = more similar)
        """
        # Calculate dot product
        dot_product = np.dot(series1, series2)
        
        # Calculate magnitudes
        magnitude1 = np.sqrt(np.sum(series1 ** 2))
        magnitude2 = np.sqrt(np.sum(series2 ** 2))
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 1.0  # Maximum distance for zero vectors
        
        # Calculate cosine similarity
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        
        # Convert to distance: higher similarity = lower distance
        return 1 - abs(cosine_sim)
    
    @staticmethod
    def normalized_euclidean_distance(series1: np.ndarray, series2: np.ndarray) -> float:
        """
        Calculate normalized Euclidean distance (z-score normalized)
        
        Args:
            series1, series2: 24-hour price arrays
            
        Returns:
            Normalized Euclidean distance
        """
        # Normalize both series
        series1_norm = (series1 - np.mean(series1)) / np.std(series1)
        series2_norm = (series2 - np.mean(series2)) / np.std(series2)
        return np.sqrt(np.sum((series1_norm - series2_norm) ** 2))

class HistoricalPriceDatabaseGenerator:
    """Main class for generating historical price databases"""
    
    def __init__(self, belgium_file: str = "Belgium.csv", 
                 test_file: str = "price_data_2024.csv"):
        """
        Initialize the database generator
        
        Args:
            belgium_file: Path to historical Belgium price data
            test_file: Path to 2024 test price data
        """
        self.belgium_file = belgium_file
        self.test_file = test_file
        self.historical_data = None
        self.test_data = None
        self.similarity_results = {}
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical Belgium price data (pre-2024)"""
        print("Loading historical Belgium price data...")
        
        try:
            # Load Belgium data
            df = pd.read_csv(self.belgium_file)
            
            # Convert datetime column
            df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
            
            # Filter for pre-2024 data
            df_historical = df[df['Datetime (UTC)'].dt.year < 2024].copy()
            
            # Extract date and hour
            df_historical['Date'] = df_historical['Datetime (UTC)'].dt.date
            df_historical['Hour'] = df_historical['Datetime (UTC)'].dt.hour
            
            # Group by date and ensure we have complete 24-hour days
            daily_data = []
            for date, group in df_historical.groupby('Date'):
                if len(group) == 24:  # Complete day
                    group_sorted = group.sort_values('Hour')
                    daily_data.append({
                        'Date': date,
                        'Prices': group_sorted['Price (EUR/MWhe)'].values
                    })
            
            self.historical_data = pd.DataFrame(daily_data)
            print(f"Loaded {len(self.historical_data)} complete historical days")
            
            return self.historical_data
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            return None
    
    def load_test_data(self) -> pd.DataFrame:
        """Load 2024 test price data"""
        print("Loading 2024 test price data...")
        
        try:
            df = pd.read_csv(self.test_file)
            
            # Parse the price data
            test_data = []
            for idx, row in df.iterrows():
                date_str = row['date']
                prices_str = row['prices_hourly']
                
                # Parse date with flexible format handling
                try:
                    # Try common date formats
                    date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
                    parsed_date = None
                    
                    for date_format in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_str, date_format).date()
                            break
                        except ValueError:
                            continue
                    
                    # If none worked, try pandas to_datetime which is more flexible
                    if parsed_date is None:
                        parsed_date = pd.to_datetime(date_str).date()
                        
                except Exception as date_error:
                    print(f"Error parsing date '{date_str}': {date_error}")
                    continue
                
                # Parse prices (handle different delimiters)
                try:
                    prices = [float(p) for p in prices_str.split(',')]
                except:
                    try:
                        prices = [float(p) for p in prices_str.split(';')]
                    except:
                        try:
                            # Handle bracketed format like [1.0, 2.0, 3.0]
                            prices_str_clean = prices_str.strip('[]')
                            prices = [float(p.strip()) for p in prices_str_clean.split(',')]
                        except:
                            # Handle space-separated
                            try:
                                prices_str_clean = prices_str.strip('[]')
                                prices = [float(p) for p in prices_str_clean.split()]
                            except Exception as price_error:
                                print(f"Error parsing prices for {date_str}: {price_error}")
                                continue
                
                # Ensure 24 hours
                if len(prices) != 24:
                    if len(prices) < 24:
                        prices.extend([prices[-1]] * (24 - len(prices)))
                    else:
                        prices = prices[:24]
                
                test_data.append({
                    'Date': parsed_date,
                    'Prices': np.array(prices)
                })
            
            self.test_data = pd.DataFrame(test_data)
            print(f"Loaded {len(self.test_data)} test days")
            
            return self.test_data
            
        except Exception as e:
            print(f"Error loading test data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def find_similar_days(self, method: str = 'euclidean') -> Dict:
        """
        Find most similar historical days for each test day
        
        Args:
            method: 'euclidean', 'pearson', 'cosine', or 'normalized_euclidean'
            
        Returns:
            Dictionary with test dates as keys and similar day info as values
        """
        if self.historical_data is None or self.test_data is None:
            raise ValueError("Data not loaded. Call load_historical_data() and load_test_data() first.")
        
        print(f"Finding similar days using {method} method...")
        
        # Select similarity function
        if method == 'euclidean':
            similarity_func = SimilarityMethods.euclidean_distance
        elif method == 'pearson':
            similarity_func = SimilarityMethods.pearson_correlation
        elif method == 'cosine':
            similarity_func = SimilarityMethods.cosine_similarity
        elif method == 'normalized_euclidean':
            similarity_func = SimilarityMethods.normalized_euclidean_distance
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: 'euclidean', 'pearson', 'cosine', 'normalized_euclidean'")
        
        results = {}
        
        for _, test_row in self.test_data.iterrows():
            test_date = test_row['Date']
            test_prices = test_row['Prices']
            
            # Calculate similarity with all historical days
            similarities = []
            for _, hist_row in self.historical_data.iterrows():
                hist_date = hist_row['Date']
                hist_prices = hist_row['Prices']
                
                # Skip if same date (shouldn't happen with pre-2024 data)
                if hist_date == test_date:
                    continue
                
                distance = similarity_func(test_prices, hist_prices)
                similarities.append({
                    'historical_date': hist_date,
                    'distance': distance,
                    'historical_prices': hist_prices
                })
            
            # Find most similar (minimum distance)
            similarities.sort(key=lambda x: x['distance'])
            best_match = similarities[0]
            
            results[test_date] = {
                'similar_date': best_match['historical_date'],
                'distance': best_match['distance'],
                'test_prices': test_prices,
                'historical_prices': best_match['historical_prices']
            }
            
            print(f"Test date {test_date} -> Similar date {best_match['historical_date']} "
                  f"(distance: {best_match['distance']:.4f})")
        
        self.similarity_results[method] = results
        return results
    
    def generate_synthetic_operational_data(self, prices: np.ndarray, date: datetime.date) -> Dict:
        """
        Generate synthetic Power, Head, Flow data based on price patterns
        
        Args:
            prices: 24-hour price array
            date: Date for the data
            
        Returns:
            Dictionary with Time, Power, Head, Flow, Price, Date columns
        """
        # Normalize prices to [0, 1] range for scaling
        price_min, price_max = np.min(prices), np.max(prices)
        price_norm = (prices - price_min) / (price_max - price_min) if price_max != price_min else np.zeros_like(prices)
        
        # Generate realistic operational data based on price patterns
        data = []
        
        # Base parameters for UPHES system
        base_head = 77.0  # meters (from your code)
        head_variation = 15.0  # meters variation
        max_power_turbine = 25.0  # MW
        max_power_pump = -20.0  # MW
        
        for hour in range(24):
            price = prices[hour]
            price_normalized = price_norm[hour]
            
            # Determine operation mode based on relative price level
            # High prices -> turbine mode (positive power)
            # Low prices -> pump mode (negative power)  
            # Medium prices -> idle mode (zero power)
            
            if price_normalized > 0.7:  # High price - turbine mode
                power = max_power_turbine * (0.5 + 0.5 * price_normalized)
                flow = power * 0.8 + np.random.normal(0, 0.5)  # Positive flow
                head = base_head + head_variation * (0.5 - price_normalized * 0.3)
            elif price_normalized < 0.3:  # Low price - pump mode
                power = max_power_pump * (0.5 + 0.5 * (1 - price_normalized))
                flow = power * 0.7 + np.random.normal(0, 0.5)  # Negative flow
                head = base_head + head_variation * (0.5 + price_normalized * 0.3)
            else:  # Medium price - idle or minimal operation
                # Add some randomness for idle/minimal operation
                if np.random.random() > 0.6:  # 40% chance of minimal operation
                    power = np.random.uniform(-5, 5)
                    flow = power * 0.6 + np.random.normal(0, 0.3)
                else:  # Idle
                    power = 0.0
                    flow = 0.0
                head = base_head + np.random.uniform(-head_variation*0.2, head_variation*0.2)
            
            # Add some noise for realism
            power += np.random.normal(0, 0.5)
            head += np.random.normal(0, 0.5)
            flow += np.random.normal(0, 0.2)
            
            # Ensure reasonable bounds
            head = max(50, min(100, head))  # Reasonable head bounds
            
            data.append({
                'Time': hour,
                'Power': round(power, 2),
                'Head': round(head, 2),
                'Flow': round(flow, 2),
                'Price': round(price, 2),
                'Date': date.strftime('%Y-%m-%d')
            })
        
        return data
    
    def create_database_csv(self, method: str, output_file: str):
        """
        Create a CSV database file compatible with recursive_linearization_pipeline1.py
        
        Args:
            method: Similarity method used ('euclidean' or 'pearson')
            output_file: Path for output CSV file
        """
        if method not in self.similarity_results:
            raise ValueError(f"Results for method '{method}' not found. Run find_similar_days() first.")
        
        print(f"Creating database CSV for {method} method...")
        
        results = self.similarity_results[method]
        all_data = []
        
        for test_date, similar_info in results.items():
            similar_date = similar_info['similar_date']
            historical_prices = similar_info['historical_prices']
            
            # Generate synthetic operational data for the historical similar day
            day_data = self.generate_synthetic_operational_data(historical_prices, similar_date)
            all_data.extend(day_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_data)
        
        # Ensure proper column order
        df = df[['Time', 'Power', 'Head', 'Flow', 'Price', 'Date']]
        
        # Sort by Date and Time
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Date', 'Time'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Database saved to {output_file}")
        print(f"Total records: {len(df)}")
        print(f"Unique dates: {df['Date'].nunique()}")
        
        return df
    
    def create_visualization(self, method: str, save_path: str = None):
        """
        Create visualization comparing test prices with found similar days
        
        Args:
            method: Similarity method to visualize
            save_path: Path to save the plot (optional)
        """
        if method not in self.similarity_results:
            raise ValueError(f"Results for method '{method}' not found.")
        
        results = self.similarity_results[method]
        n_days = len(results)
        
        # Create subplots - organize in a grid for better visualization
        if n_days <= 4:
            rows, cols = n_days, 1
            figsize = (15, 4*n_days)
        elif n_days <= 8:
            rows, cols = 4, 2
            figsize = (20, 16)
        elif n_days <= 12:
            rows, cols = 4, 3
            figsize = (24, 16)
        elif n_days <= 16:
            rows, cols = 4, 4
            figsize = (28, 16)
        else:
            rows, cols = 5, 4
            figsize = (28, 20)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Flatten axes array for easier indexing
        if n_days == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        # Sort results by test date for consistent visualization
        sorted_results = sorted(results.items(), key=lambda x: x[0])
        
        for i, (test_date, similar_info) in enumerate(sorted_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            test_prices = similar_info['test_prices']
            hist_prices = similar_info['historical_prices']
            similar_date = similar_info['similar_date']
            distance = similar_info['distance']
            
            hours = range(24)
            ax.plot(hours, test_prices, 'b-o', label=f'Test: {test_date}', 
                   linewidth=2, markersize=3, alpha=0.8)
            ax.plot(hours, hist_prices, 'r--s', label=f'Similar: {similar_date}', 
                   linewidth=2, markersize=2, alpha=0.8)
            
            ax.set_title(f'{method.title()} Method - Distance: {distance:.4f}', fontsize=10)
            ax.set_xlabel('Hour', fontsize=9)
            ax.set_ylabel('Price (EUR/MWh)', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient and cosine similarity for reference
            corr, _ = pearsonr(test_prices, hist_prices)
            
            # Calculate cosine similarity for comparison
            dot_product = np.dot(test_prices, hist_prices)
            magnitude1 = np.sqrt(np.sum(test_prices ** 2))
            magnitude2 = np.sqrt(np.sum(hist_prices ** 2))
            cosine_sim = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 != 0 else 0
            
            # Add text box with metrics
            textstr = f'Corr: {corr:.3f}\nCosine: {cosine_sim:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=props)
        
        # Hide unused subplots
        for i in range(n_days, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'{method.title()} Similarity Method - All 20 Test Days Comparison', 
                    fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def compare_methods(self):
        """Compare results from different similarity methods"""
        if len(self.similarity_results) < 2:
            print("Need at least 2 methods to compare. Run find_similar_days() for multiple methods.")
            return
        
        methods = list(self.similarity_results.keys())
        print(f"\nComparing methods: {', '.join(methods)}")
        
        # Create comparison table
        comparison_data = []
        
        for test_date in self.test_data['Date'].values:
            row = {'Test_Date': test_date}
            
            for method in methods:
                if test_date in self.similarity_results[method]:
                    similar_info = self.similarity_results[method][test_date]
                    row[f'{method}_similar_date'] = similar_info['similar_date']
                    row[f'{method}_distance'] = similar_info['distance']
                    
                    # Calculate Pearson correlation for comparison
                    corr, _ = pearsonr(similar_info['test_prices'], similar_info['historical_prices'])
                    row[f'{method}_correlation'] = corr
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display summary statistics
        print("\nSummary Statistics:")
        for method in methods:
            distances = comparison_df[f'{method}_distance'].values
            correlations = comparison_df[f'{method}_correlation'].values
            
            print(f"\n{method.upper()} Method:")
            print(f"  Average distance: {np.mean(distances):.4f} ± {np.std(distances):.4f}")
            print(f"  Average correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
            print(f"  Min/Max distance: {np.min(distances):.4f} / {np.max(distances):.4f}")
        
        return comparison_df

def main():
    """Main execution function"""
    print("=== Historical Price Database Generator ===\n")
    
    # Initialize generator
    generator = HistoricalPriceDatabaseGenerator()
    
    # Load data
    historical_data = generator.load_historical_data()
    test_data = generator.load_test_data()
    
    if historical_data is None or test_data is None:
        print("Failed to load data. Please check file paths and formats.")
        return
    
    # Find similar days using all methods
    print("\n" + "="*50)
    print("FINDING SIMILAR DAYS")
    print("="*50)
    
    # Method 1: Euclidean Distance
    euclidean_results = generator.find_similar_days('euclidean')
    
    # Method 2: Pearson Correlation
    pearson_results = generator.find_similar_days('pearson')
    
    # Method 3: Cosine Similarity
    cosine_results = generator.find_similar_days('cosine')
    
    # Method 4: Normalized Euclidean (bonus)
    norm_euclidean_results = generator.find_similar_days('normalized_euclidean')
    
    # Compare methods
    print("\n" + "="*50)
    print("METHOD COMPARISON")
    print("="*50)
    comparison_df = generator.compare_methods()
    
    # Create database CSV files
    print("\n" + "="*50)
    print("CREATING DATABASE FILES")
    print("="*50)
    
    # Database 1: Euclidean Distance method
    db1_file = "historical_database_euclidean.csv"
    generator.create_database_csv('euclidean', db1_file)
    
    # Database 2: Pearson Correlation method  
    db2_file = "historical_database_pearson.csv"
    generator.create_database_csv('pearson', db2_file)
    
    # Database 3: Cosine Similarity method
    db3_file = "historical_database_cosine.csv"
    generator.create_database_csv('cosine', db3_file)
    
    # Create visualizations for all 20 days
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS (ALL 20 DAYS)")
    print("="*50)
    
    generator.create_visualization('euclidean', 'euclidean_similarity_all_days.png')
    generator.create_visualization('pearson', 'pearson_similarity_all_days.png')
    generator.create_visualization('cosine', 'cosine_similarity_all_days.png')
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"✓ Historical database (Euclidean): {db1_file}")
    print(f"✓ Historical database (Pearson): {db2_file}")
    print(f"✓ Historical database (Cosine): {db3_file}")
    print(f"✓ Visualization files saved to current directory")
    print(f"✓ Found similar days for {len(test_data)} test dates")
    
    print("\nFiles are compatible with recursive_linearization_pipeline1.py")
    print("Use load_historical_data() function to read the generated CSV files.")
    
    # Create a summary comparison table
    print("\n" + "="*50)
    print("DETAILED METHOD COMPARISON")
    print("="*50)
    
    methods = ['euclidean', 'pearson', 'cosine', 'normalized_euclidean']
    summary_stats = {}
    
    for method in methods:
        if method in generator.similarity_results:
            results = generator.similarity_results[method]
            distances = [info['distance'] for info in results.values()]
            correlations = []
            cosine_sims = []
            
            for info in results.values():
                # Calculate correlation
                corr, _ = pearsonr(info['test_prices'], info['historical_prices'])
                correlations.append(corr)
                
                # Calculate cosine similarity
                dot_product = np.dot(info['test_prices'], info['historical_prices'])
                mag1 = np.sqrt(np.sum(info['test_prices'] ** 2))
                mag2 = np.sqrt(np.sum(info['historical_prices'] ** 2))
                cosine_sim = dot_product / (mag1 * mag2) if mag1 * mag2 != 0 else 0
                cosine_sims.append(cosine_sim)
            
            summary_stats[method] = {
                'avg_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'avg_correlation': np.mean(correlations),
                'avg_cosine_sim': np.mean(cosine_sims),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances)
            }
            
            print(f"\n{method.upper()} Method:")
            print(f"  Average distance: {summary_stats[method]['avg_distance']:.4f} ± {summary_stats[method]['std_distance']:.4f}")
            print(f"  Average correlation: {summary_stats[method]['avg_correlation']:.4f}")
            print(f"  Average cosine similarity: {summary_stats[method]['avg_cosine_sim']:.4f}")
            print(f"  Distance range: [{summary_stats[method]['min_distance']:.4f}, {summary_stats[method]['max_distance']:.4f}]")

if __name__ == "__main__":
    main()
# %%
