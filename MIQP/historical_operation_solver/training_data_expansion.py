#!/usr/bin/env python3
"""
Extended Training Database Generator for UPHES Decision-Focused Learning

This script creates an extended training database by:
1. Copying existing euclidean_piecewise operational data (only if training CSV doesn't exist)
2. Finding 2nd-20th most similar historical days for all 2024 dates (except 2024/12/12)
3. Running MIQP piecewise optimization on these additional similar days
4. Appending results to create a comprehensive training dataset
5. Automatically skipping duplicate dates already present in the training dataset

The script supports pause/resume functionality and processes similarity ranks sequentially
(2nd closest for all dates, then 3rd closest, etc., up to 20th closest)

IMPORTANT: This script NEVER overwrites existing training data - it only appends new results.
Progress and results are saved after EACH successful MIQP optimization.

Directory Structure:
extended_training_data/
├── training_data_piecewise.csv          # Main training dataset
├── progress.json                        # Progress tracking for resume
├── similarity_results.json              # Cached similarity computations
└── logs/
    ├── processing_log.txt               # Detailed processing log
    └── failed_dates.json                # Failed optimization attempts

Input Requirements:
- D:/Repositories/DFL-for-UPHES/Data/Belgium.csv
- D:/Repositories/DFL-for-UPHES/Data/price_data_2024.csv  
- D:/Repositories/DFL-for-UPHES/MIQP/historical_operation_solver/euclidean_piecewise/detailed_results.csv
"""

# %% Imports and Setup
import sys
import os
import pandas as pd
import numpy as np
import time
import json
import torch
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set device early
device = torch.device("cpu")

# Define paths
current_dir = Path(__file__).parent
root_dir = Path("D:/Repositories/DFL-for-UPHES")
data_dir = root_dir / "Data"
miqp_dir = root_dir / "MIQP"
solver_dir = miqp_dir / "historical_operation_solver"

# Add paths for imports
sys.path.append(str(miqp_dir / "MIQP_piecewise"))
sys.path.append(str(root_dir / "Library"))

print("Extended Training Database Generator")
print("=" * 60)
print(f"Root directory: {root_dir}")
print(f"Data directory: {data_dir}")
print(f"Solver directory: {solver_dir}")

# %% Load Required Libraries and Data
print("\n" + "=" * 60)
print("LOADING LIBRARIES AND PORTFOLIO DATA")
print("=" * 60)

# Load portfolio data
try:
    from V_H_relations import load_portfolio_data, gross_head, get_v_low
    load_portfolio_data()
    from V_H_relations import (r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, 
                              h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, 
                              min_vol_low, target_vol_up, target_vol_low, target_head)
    print("✓ Portfolio data loaded")
except Exception as e:
    print(f"✗ Error loading portfolio data: {e}")
    sys.exit(1)

# Load preprocessed data
try:
    import dill as pickle
    with open(str(root_dir / 'preprocess.pkl'), 'rb') as f:
        (v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, 
         intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, 
         h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, 
         DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, 
         pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound) = pickle.load(f)
    
    head_init = 77.0
    v_low_init = h_to_v_low_fitted(head_init)
    v_low_init = float(v_low_init)
    print("✓ Preprocessed data loaded")
except Exception as e:
    print(f"✗ Error loading preprocessed data: {e}")
    sys.exit(1)

# Load MIQP Piecewise
try:
    from MIQP_piecewise import (
        PiecewiseMILPOptimizerSOS2,
        HydroParameters as HydroParams_Piecewise,
        SimulationLayer as SimLayer_Piecewise
    )
    print("✓ MIQP Piecewise formulation imported")
except Exception as e:
    print(f"✗ Error importing MIQP Piecewise: {e}")
    sys.exit(1)

# %% Setup Output Directory Structure
print("\n" + "=" * 60)  
print("SETTING UP OUTPUT DIRECTORY")
print("=" * 60)

# Create output directory structure
output_dir = current_dir / "extended_training_data"
output_dir.mkdir(exist_ok=True)

log_dir = output_dir / "logs"
log_dir.mkdir(exist_ok=True)

# Define output files
training_csv = output_dir / "training_data_piecewise.csv"
progress_file = output_dir / "progress.json"
similarity_file = output_dir / "similarity_results.json"
log_file = log_dir / "processing_log.txt"
failed_file = log_dir / "failed_dates.json"

print(f"✓ Output directory: {output_dir}")
print(f"✓ Training CSV: {training_csv}")
print(f"✓ Progress file: {progress_file}")

# %% Data Loading Classes
class SimilarityMethods:
    """Similarity calculation methods"""
    
    @staticmethod
    def euclidean_distance(series1: np.ndarray, series2: np.ndarray) -> float:
        """Calculate Euclidean distance between two price series"""
        return np.sqrt(np.sum((series1 - series2) ** 2))

class DataLoader:
    """Load and process historical and test data"""
    
    def __init__(self, belgium_file: str, test_file: str):
        self.belgium_file = belgium_file
        self.test_file = test_file
        self.historical_data = None
        self.test_data = None
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical Belgium price data (pre-2024)"""
        print("Loading historical Belgium price data...")
        
        df = pd.read_csv(self.belgium_file)
        df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'])
        df_historical = df[df['Datetime (UTC)'].dt.year < 2024].copy()
        
        df_historical['Date'] = df_historical['Datetime (UTC)'].dt.date
        df_historical['Hour'] = df_historical['Datetime (UTC)'].dt.hour
        
        # Group by date and ensure complete 24-hour days
        daily_data = []
        for date, group in df_historical.groupby('Date'):
            if len(group) == 24:
                group_sorted = group.sort_values('Hour')
                daily_data.append({
                    'Date': date,
                    'Prices': group_sorted['Price (EUR/MWhe)'].values
                })
        
        self.historical_data = pd.DataFrame(daily_data)
        print(f"✓ Loaded {len(self.historical_data)} complete historical days")
        return self.historical_data
        
    def load_test_data(self) -> pd.DataFrame:
        """Load 2024 test price data"""
        print("Loading 2024 test price data...")
        
        df = pd.read_csv(self.test_file)
        
        test_data = []
        for idx, row in df.iterrows():
            date_str = row['date']
            prices_str = row['prices_hourly']
            
            # Parse date
            try:
                date_formats = ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y']
                parsed_date = None
                
                for date_format in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str, date_format).date()
                        break
                    except ValueError:
                        continue
                
                if parsed_date is None:
                    parsed_date = pd.to_datetime(date_str).date()
                        
            except Exception as e:
                print(f"Error parsing date '{date_str}': {e}")
                continue
            
            # Parse prices
            try:
                if prices_str.startswith('[') and prices_str.endswith(']'):
                    prices_str_clean = prices_str.strip('[]')
                    prices = [float(p.strip()) for p in prices_str_clean.split(',')]
                else:
                    prices = [float(p) for p in prices_str.split(',')]
            except:
                try:
                    prices = [float(p) for p in prices_str.split(';')]
                except Exception as e:
                    print(f"Error parsing prices for {date_str}: {e}")
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
        print(f"✓ Loaded {len(self.test_data)} test days")
        return self.test_data

# %% Extended Similarity Finder
class ExtendedSimilarityFinder:
    """Find multiple most similar days for each test date"""
    
    def __init__(self, historical_data, test_data, exclude_date=None):
        self.historical_data = historical_data
        self.test_data = test_data
        self.exclude_date = exclude_date
        self.similarity_results = {}
        
    def find_multiple_similar_days(self, num_similar=20):
        """
        Find multiple most similar historical days for each test date
        
        Args:
            num_similar: Number of similar days to find (default 20 for ranks 1-20)
            
        Returns:
            Dictionary with test dates as keys and ranked similar days as values
        """
        print(f"Finding top {num_similar} similar days for each test date...")
        
        results = {}
        
        for _, test_row in self.test_data.iterrows():
            test_date = test_row['Date']
            test_prices = test_row['Prices']
            
            # Skip if this is the excluded date
            if self.exclude_date and test_date == self.exclude_date:
                print(f"Skipping excluded date: {test_date}")
                continue
            
            # Calculate similarity with all historical days
            similarities = []
            for _, hist_row in self.historical_data.iterrows():
                hist_date = hist_row['Date']
                hist_prices = hist_row['Prices']
                
                # Skip if same date
                if hist_date == test_date:
                    continue
                
                distance = SimilarityMethods.euclidean_distance(test_prices, hist_prices)
                similarities.append({
                    'historical_date': hist_date,
                    'distance': distance,
                    'historical_prices': hist_prices
                })
            
            # Sort by distance and get top matches
            similarities.sort(key=lambda x: x['distance'])
            top_matches = similarities[:num_similar]
            
            # Store results with ranks
            test_results = {}
            for rank, match in enumerate(top_matches, 1):
                test_results[f'rank_{rank}'] = {
                    'historical_date': match['historical_date'],
                    'distance': match['distance'],
                    'test_prices': test_prices.tolist(),
                    'historical_prices': match['historical_prices'].tolist()
                }
            
            results[str(test_date)] = test_results
            print(f"Test date {test_date}: Found {len(top_matches)} similar days")
        
        self.similarity_results = results
        return results

# %% Copy Existing Training Data (SAFE VERSION)
def initialize_training_data():
    """Initialize training data ONLY if it doesn't exist - NEVER overwrites existing data"""
    print("\n" + "=" * 60)
    print("INITIALIZING TRAINING DATA")
    print("=" * 60)
    
    # Check if training CSV already exists
    if training_csv.exists():
        existing_data = pd.read_csv(training_csv)
        print(f"✓ Training CSV already exists with {len(existing_data)} records")
        print(f"✓ Unique dates in existing training data: {existing_data['date'].nunique()}")
        print("✓ Skipping initialization - will append new data only")
        return existing_data
    
    print("Training CSV doesn't exist - initializing from euclidean_piecewise data...")
    
    # Load existing euclidean_piecewise data
    existing_file = solver_dir / "euclidean_piecewise" / "detailed_results.csv"
    
    if not existing_file.exists():
        print(f"✗ Existing training data not found: {existing_file}")
        print("✓ Will create new training dataset from scratch")
        return None
        
    existing_data = pd.read_csv(existing_file)
    print(f"✓ Loaded {len(existing_data)} records from existing euclidean_piecewise data")
    
    # Find the closest match to 2024/12/12 to exclude
    exclude_date = datetime(2024, 12, 12).date()
    
    # Load similarity results to find what to exclude
    loader = DataLoader(str(data_dir / "Belgium.csv"), str(data_dir / "price_data_2024.csv"))
    historical_data = loader.load_historical_data()
    test_data = loader.load_test_data()
    
    # Find the closest match to 2024/12/12
    test_row_1212 = test_data[test_data['Date'] == exclude_date]
    if len(test_row_1212) == 0:
        print(f"Warning: 2024/12/12 not found in test data")
        closest_historical_date = None
    else:
        test_prices_1212 = test_row_1212.iloc[0]['Prices']
        
        similarities = []
        for _, hist_row in historical_data.iterrows():
            hist_date = hist_row['Date']
            hist_prices = hist_row['Prices']
            distance = SimilarityMethods.euclidean_distance(test_prices_1212, hist_prices)
            similarities.append({
                'historical_date': hist_date,
                'distance': distance
            })
        
        similarities.sort(key=lambda x: x['distance'])
        closest_historical_date = similarities[0]['historical_date']
        print(f"✓ Closest historical date to 2024/12/12: {closest_historical_date} (distance: {similarities[0]['distance']:.2f})")
    
    # Filter out the closest match to 2024/12/12
    if closest_historical_date:
        original_count = len(existing_data)
        existing_data = existing_data[existing_data['date'] != str(closest_historical_date)]
        filtered_count = len(existing_data)
        print(f"✓ Excluded {original_count - filtered_count} records for date {closest_historical_date}")
    
    # Save to training dataset (ONLY FIRST TIME)
    existing_data.to_csv(training_csv, index=False)
    print(f"✓ Initialized training dataset with {len(existing_data)} records")
    print(f"✓ Unique dates in base training data: {existing_data['date'].nunique()}")
    
    return existing_data

# %% MIQP Piecewise Runner
class MIQPPiecewiseRunner:
    """Run MIQP Piecewise optimization on price data"""
    
    def __init__(self):
        self.failed_dates = []
        
    def run_piecewise_optimization(self, date_str, prices_24h):
        """Run MIQP Piecewise optimization for given date and prices"""
        try:
            start_time = time.time()
            
            # Create and solve optimizer
            T = 24
            optimizer = PiecewiseMILPOptimizerSOS2(
                T=T, 
                DA_prices=prices_24h,
                num_segments_h=10,
                num_segments_p_pump=10,
                num_segments_p_turbine=10
            )
            optimizer.model.Params.MIPGap = 0.01
            optimizer.model.Params.TimeLimit = 3600  # 1 hour time limit
            
            results, metrics = optimizer.solve()
            
            solution_time = time.time() - start_time
            
            if results is None:
                return None, None
            
            # Run simulation
            head_init_val = torch.tensor(head_init, dtype=torch.float32, device=device)
            v_low_init_val = torch.tensor(v_low_init, dtype=torch.float32, device=device)
            
            params = HydroParams_Piecewise(
                head_init=head_init_val,
                v_low_init=v_low_init_val,
                neg_min=neg_min, neg_max=neg_max,
                pos_min=pos_min, pos_max=pos_max,
                predict_q_poly=predict_q_poly,
                h_to_v_low_fitted=h_to_v_low_fitted,
                v_low_to_h_fitted=v_low_to_h_fitted
            )
            
            simulator = SimLayer_Piecewise(params)
            
            # Convert to tensors
            p_tensor = torch.tensor(results['p'], dtype=torch.float32, device=device)
            q_tensor = torch.tensor(results['q'], dtype=torch.float32, device=device)
            h_tensor = torch.tensor(results['h'], dtype=torch.float32, device=device)
            
            # Run simulation
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)
            
            # Calculate simulation profit
            da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
            profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_tensor[:len(p_sim)], v_low_sim, da_prices_tensor[:len(p_sim)]
            )
            
            # Store detailed results
            detailed = []
            for hour in range(T):
                detailed.append({
                    'date': date_str,
                    'hour': hour,
                    'power': results['p'][hour],
                    'head': results['h'][hour],
                    'volume': results['v_low'][hour],
                    'flow': results['q'][hour],
                    'price': prices_24h[hour]
                })
            
            return detailed, {
                'date': date_str,
                'solve_time': solution_time,
                'expected_profit': metrics['ExpectedProfit'],
                'expost_profit': profit.item(),
                'si_penalty': si_penalty.item(),
                'vol_penalty': vol_penalty.item(),
                'op_cost': op_cost.item()
            }
            
        except Exception as e:
            self.failed_dates.append({
                'date': date_str,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return None, None

# %% Progress Management
class ProgressManager:
    """Manage progress tracking for pause/resume functionality"""
    
    def __init__(self, progress_file):
        self.progress_file = progress_file
        self.progress = self.load_progress()
        
    def load_progress(self):
        """Load existing progress or initialize new"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                # Update current_rank for extended processing if ranks 2-10 are complete
                if all(rank in progress.get('completed_ranks', []) for rank in range(2, 11)):
                    if progress.get('current_rank', 2) < 11:
                        progress['current_rank'] = 11
                return progress
        else:
            return {
                'completed_ranks': [],
                'current_rank': 2,
                'completed_dates_in_rank': {},
                'last_update': datetime.now().isoformat(),
                'total_dates_processed': 0
            }
    
    def save_progress(self):
        """Save current progress"""
        self.progress['last_update'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def is_rank_completed(self, rank):
        """Check if a similarity rank is completed"""
        return rank in self.progress['completed_ranks']
    
    def is_date_completed_in_rank(self, rank, date_str):
        """Check if a specific date is completed for a rank"""
        rank_key = f'rank_{rank}'
        return rank_key in self.progress['completed_dates_in_rank'] and \
               date_str in self.progress['completed_dates_in_rank'][rank_key]
    
    def mark_date_completed(self, rank, date_str):
        """Mark a date as completed for a rank"""
        rank_key = f'rank_{rank}'
        if rank_key not in self.progress['completed_dates_in_rank']:
            self.progress['completed_dates_in_rank'][rank_key] = []
        
        if date_str not in self.progress['completed_dates_in_rank'][rank_key]:
            self.progress['completed_dates_in_rank'][rank_key].append(date_str)
        
        self.progress['total_dates_processed'] += 1
    
    def mark_rank_completed(self, rank):
        """Mark an entire similarity rank as completed"""
        if rank not in self.progress['completed_ranks']:
            self.progress['completed_ranks'].append(rank)
        
        # Update current rank to next one
        if rank == self.progress['current_rank']:
            self.progress['current_rank'] = rank + 1

# %% Duplicate Date Checker
def get_existing_dates_in_training():
    """Get set of dates already present in training dataset"""
    if not training_csv.exists():
        return set()
    
    try:
        existing_data = pd.read_csv(training_csv)
        existing_dates = set(existing_data['date'].unique())
        print(f"✓ Found {len(existing_dates)} unique dates already in training dataset")
        return existing_dates
    except Exception as e:
        print(f"Warning: Could not read existing training data: {e}")
        return set()

# %% Main Processing Function
def process_extended_training_data():
    """Main function to process extended training data"""
    print("\n" + "=" * 60)
    print("PROCESSING EXTENDED TRAINING DATA")
    print("=" * 60)
    
    # Initialize progress manager
    progress_mgr = ProgressManager(progress_file)
    
    # Load or compute similarity results
    if similarity_file.exists():
        print("Loading cached similarity results...")
        with open(similarity_file, 'r') as f:
            similarity_results = json.load(f)
        print(f"✓ Loaded similarity results for {len(similarity_results)} test dates")
        
        # Check if we need to extend similarity results for ranks 11-20
        sample_test_date = list(similarity_results.keys())[0]
        max_rank_in_cache = max([int(k.split('_')[1]) for k in similarity_results[sample_test_date].keys() if k.startswith('rank_')])
        
        if max_rank_in_cache < 20:
            print(f"Cached similarity results only go up to rank {max_rank_in_cache}, recomputing for ranks 1-20...")
            loader = DataLoader(str(data_dir / "Belgium.csv"), str(data_dir / "price_data_2024.csv"))
            historical_data = loader.load_historical_data()
            test_data = loader.load_test_data()
            
            # Exclude 2024/12/12 from similarity computation
            exclude_date = datetime(2024, 12, 12).date()
            
            finder = ExtendedSimilarityFinder(historical_data, test_data, exclude_date)
            similarity_results = finder.find_multiple_similar_days(num_similar=20)
            
            # Save updated similarity results
            with open(similarity_file, 'w') as f:
                json.dump(similarity_results, f, indent=2, default=str)
            print(f"✓ Updated similarity results for ranks 1-20")
    else:
        print("Computing similarity results...")
        loader = DataLoader(str(data_dir / "Belgium.csv"), str(data_dir / "price_data_2024.csv"))
        historical_data = loader.load_historical_data()
        test_data = loader.load_test_data()
        
        # Exclude 2024/12/12 from similarity computation
        exclude_date = datetime(2024, 12, 12).date()
        
        finder = ExtendedSimilarityFinder(historical_data, test_data, exclude_date)
        similarity_results = finder.find_multiple_similar_days(num_similar=20)
        
        # Save similarity results
        with open(similarity_file, 'w') as f:
            json.dump(similarity_results, f, indent=2, default=str)
        print(f"✓ Computed and saved similarity results for {len(similarity_results)} test dates")
    
    # Initialize MIQP runner
    runner = MIQPPiecewiseRunner()
    
    # Get existing dates to avoid duplicates
    existing_dates = get_existing_dates_in_training()
    
    # Process ranks 2-20 (excluding rank 1 which was the original euclidean_piecewise data)
    # Start from rank 11 if ranks 2-10 are already completed, otherwise start from current progress
    start_rank = max(2, progress_mgr.progress['current_rank'])
    
    # If ranks 2-10 are completed, start from rank 11
    if all(rank in progress_mgr.progress['completed_ranks'] for rank in range(2, 11)):
        start_rank = max(11, start_rank)
    
    print(f"\nStarting from rank {start_rank}")
    print(f"Previously completed ranks: {progress_mgr.progress['completed_ranks']}")
    print(f"Will skip {len(existing_dates)} dates already in training dataset")
    
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f"\n\n=== Processing started at {datetime.now()} ===\n")
        log.write(f"Starting from rank {start_rank}\n")
        log.write(f"Will skip {len(existing_dates)} dates already in training dataset\n")
        
        for rank in range(start_rank, 21):  # Ranks 2-20
            if progress_mgr.is_rank_completed(rank):
                print(f"Rank {rank} already completed, skipping...")
                continue
                
            print(f"\n{'-' * 40}")
            print(f"PROCESSING RANK {rank}")
            print(f"{'-' * 40}")
            
            log.write(f"\n--- Processing Rank {rank} ---\n")
            
            successful_dates = 0
            failed_dates = 0
            skipped_dates = 0
            total_dates = len(similarity_results)
            
            for date_idx, (test_date_str, date_similarities) in enumerate(similarity_results.items(), 1):
                # Check if this date is already completed for this rank
                if progress_mgr.is_date_completed_in_rank(rank, test_date_str):
                    print(f"  {test_date_str} (rank {rank}) already completed, skipping...")
                    continue
                
                rank_key = f'rank_{rank}'
                if rank_key not in date_similarities:
                    print(f"  {test_date_str}: No rank {rank} data available")
                    continue
                    
                similar_info = date_similarities[rank_key]
                historical_date = similar_info['historical_date']
                historical_prices = similar_info['historical_prices']
                distance = similar_info['distance']
                
                # Check if this historical date is already in the training dataset
                if str(historical_date) in existing_dates:
                    print(f"  {test_date_str} -> {historical_date} (distance: {distance:.2f}) [DUPLICATE - SKIPPING]")
                    skipped_dates += 1
                    # Still mark as completed to avoid processing again
                    progress_mgr.mark_date_completed(rank, test_date_str)
                    # Save progress after each operation
                    progress_mgr.save_progress()
                    continue
                
                print(f"  {test_date_str} -> {historical_date} (distance: {distance:.2f}) [{date_idx}/{total_dates}]...", end=" ")
                
                # Run MIQP optimization
                detailed, summary = runner.run_piecewise_optimization(str(historical_date), historical_prices)
                
                if detailed and summary:
                    # Save results IMMEDIATELY after each successful optimization
                    detailed_df = pd.DataFrame(detailed)
                    
                    # Check if file exists and has headers
                    file_exists = training_csv.exists()
                    if file_exists:
                        # Append without headers
                        detailed_df.to_csv(training_csv, mode='a', header=False, index=False)
                    else:
                        # Create new file with headers
                        detailed_df.to_csv(training_csv, index=False)
                    
                    successful_dates += 1
                    
                    # Add this date to existing_dates to avoid future duplicates within this run
                    existing_dates.add(str(historical_date))
                    
                    expected_profit = summary['expected_profit']
                    expost_profit = summary['expost_profit'] 
                    solve_time = summary['solve_time']
                    
                    print(f"✓ Expected: {expected_profit:.2f} €, Ex-post: {expost_profit:.2f} € ({solve_time:.1f}s) [SAVED]")
                    log.write(f"  {test_date_str} -> {historical_date}: SUCCESS (Expected: {expected_profit:.2f} €, Ex-post: {expost_profit:.2f} €)\n")
                    
                    # Mark as completed
                    progress_mgr.mark_date_completed(rank, test_date_str)
                else:
                    failed_dates += 1
                    print("✗ Failed")
                    log.write(f"  {test_date_str} -> {historical_date}: FAILED\n")
                
                # Save progress after EACH date (successful or failed)
                progress_mgr.save_progress()
                
                # Save failed dates after each failure
                if runner.failed_dates:
                    with open(failed_file, 'w') as f:
                        json.dump(runner.failed_dates, f, indent=2)
            
            # Mark rank as completed
            progress_mgr.mark_rank_completed(rank)
            progress_mgr.save_progress()
            
            print(f"\n📊 Rank {rank} Summary:")
            print(f"   • Successful: {successful_dates}/{total_dates} dates")
            print(f"   • Failed: {failed_dates}/{total_dates} dates") 
            print(f"   • Skipped (duplicates): {skipped_dates}/{total_dates} dates")
            
            log.write(f"Rank {rank} completed: {successful_dates} successful, {failed_dates} failed, {skipped_dates} skipped (duplicates)\n")
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"✓ All ranks 2-20 processed")
    print(f"✓ Training dataset: {training_csv}")
    
    # Final statistics
    if training_csv.exists():
        final_data = pd.read_csv(training_csv)
        print(f"✓ Final training dataset: {len(final_data)} records, {final_data['date'].nunique()} unique dates")

# %% Main Execution
if __name__ == "__main__":
    print("Starting Extended Training Database Generator...")
    
    # Step 1: Initialize training data (SAFE - never overwrites existing data)
    initialize_training_data()
    
    # Step 2: Process extended training data (ranks 2-20)
    process_extended_training_data()
    
    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETED!")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Training dataset: {training_csv}")
    print(f"📝 Progress tracking: {progress_file}")
    print(f"📈 Similarity results: {similarity_file}")
    print("\nThe script supports pause/resume - you can stop and restart anytime!")
    print("IMPORTANT: Your data is SAFE - this script NEVER overwrites existing training data!")