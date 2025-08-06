"""
Historical Operation Database Solver

This script processes historical price databases created by historical_price_database_generator.py
and runs all three MIQP formulations (Global Linear, Piecewise, Neural Network) to create
comprehensive historical operation databases.

Processing Order: Global Linear → Piecewise → Neural Network
For each formulation, all databases are processed before moving to the next formulation.

Input Files:
- ../Data/historical_database_euclidean.csv
- ../Data/historical_database_pearson.csv
- ../Data/historical_database_cosine.csv (commented out)

Output Structure:
- euclidean_global_linear/
  ├── detailed_results.csv
  └── benchmark_results.csv
- pearson_global_linear/
  ├── detailed_results.csv
  └── benchmark_results.csv
- euclidean_piecewise/
  ├── detailed_results.csv
  └── benchmark_results.csv
- pearson_piecewise/
  ├── detailed_results.csv
  └── benchmark_results.csv
- euclidean_neural_network/
  ├── detailed_results.csv
  └── benchmark_results.csv
- pearson_neural_network/
  ├── detailed_results.csv
  └── benchmark_results.csv

Each folder contains results from one MIQP formulation applied to one historical database.
"""

# %% Imports and Setup
import sys
import os
import pandas as pd
import numpy as np
import time
import torch
import traceback
from pathlib import Path

# Set device early
device = torch.device("cpu")

# Add paths to import from other MIQP folders and necessary library paths
current_dir = Path(__file__).parent
miqp_dir = current_dir.parent
root_dir = miqp_dir.parent
sys.path.append(str(miqp_dir / "MIQP_linear"))
sys.path.append(str(miqp_dir / "MIQP_nn")) 
sys.path.append(str(miqp_dir / "MIQP_piecewise"))
sys.path.append(str(root_dir / "Library"))

# Import optimizer classes and functions from each formulation
print("Importing MIQP formulations...")

# Load portfolio data first (required by all formulations)
try:
    from V_H_relations import load_portfolio_data, gross_head, get_v_low
    load_portfolio_data()
    from V_H_relations import (r, m, head_max, head_min, h_dead_up, h_normal_up, height_up, R, height_low, n, 
                              h_dead_low, h_normal_low, max_vol_up, max_vol_low, max_vol, ramp_down, ramp_up, 
                              min_vol_low, target_vol_up, target_vol_low, target_head)
    print("✓ Portfolio data loaded")
except Exception as e:
    print(f"✗ Error loading portfolio data: {e}")

# Load preprocessed functions & data
try:
    import dill as pickle
    with open(str(root_dir / 'preprocess.pkl'), 'rb') as f:
        (v_low_h_coeffs, h_v_coeffs, v_low_to_h_fitted, v_low_h_poly, h_v_coeffs_lin, coefs_tur_lin, 
         intercept_tur_lin, coefs_pump_lin, intercept_pump_lin, predict_q_linear_tur, predict_q_linear_pump, 
         h_to_v_low_lin, h_fit, neg_min_fit, neg_max_fit, pos_min_fit, pos_max_fit, h_v_poly, h_v_coeffs, 
         DA_price_hour, DA_price_quarter, h_to_v_low_fitted, predict_q_poly, neg_min, neg_max, pos_min, 
         pos_max, prepare_and_fit_model, get_UPC_bound, LR_UPC_bound) = pickle.load(f)
    
    head_init = 77.0  # Initial head value
    v_low_init = h_to_v_low_fitted(head_init)  # Initial lower reservoir volume
    v_low_init = float(v_low_init)
    print("✓ Preprocessed data loaded")
except Exception as e:
    print(f"✗ Error loading preprocessed data: {e}")

try:
    # Global Linear imports
    # First ensure h_vlow_coeff_lin is available in numpy format
    h_vlow_coeff_lin = h_v_coeffs_lin.detach().numpy() if hasattr(h_v_coeffs_lin, 'detach') else h_v_coeffs_lin
    
    from MIQP_global_linear import (
        MILPOptimizer, 
        HydroParameters as HydroParams_Linear, 
        SimulationLayer as SimLayer_Linear
    )
    print("✓ Global Linear formulation imported")
except Exception as e:
    print(f"✗ Error importing Global Linear: {e}")

try:
    # Neural Network imports
    # Save current directory
    original_cwd = os.getcwd()
    
    # Change to MIQP_nn directory temporarily for model loading
    miqp_nn_dir = miqp_dir / "MIQP_nn"
    os.chdir(str(miqp_nn_dir))
    
    from MIQP_nn import (
        create_uphes_miqp_model,
        HydroParameters as HydroParams_NN,
        SimulationLayer as SimLayer_NN
    )
    import pyomo.environ as pyo
    
    # Change back to original directory
    os.chdir(original_cwd)
    
    print("✓ Neural Network formulation imported")
except Exception as e:
    # Make sure to change back to original directory even if import fails
    try:
        os.chdir(original_cwd)
    except:
        pass
    print(f"✗ Error importing Neural Network: {e}")

try:
    # Piecewise imports
    from MIQP_piecewise import (
        PiecewiseMILPOptimizerSOS2,
        HydroParameters as HydroParams_Piecewise,
        SimulationLayer as SimLayer_Piecewise
    )
    print("✓ Piecewise formulation imported")
except Exception as e:
    print(f"✗ Error importing Piecewise: {e}")

# %% Historical Database Reader
class HistoricalDatabaseProcessor:
    """Process historical price databases and extract daily price series."""
    
    def __init__(self, database_path):
        """
        Initialize with path to historical database CSV.
        
        Args:
            database_path: Path to historical database CSV file
        """
        self.database_path = database_path
        self.data = None
        self.daily_prices = None
        
    def load_database(self):
        """Load historical database from CSV."""
        print(f"Loading historical database: {self.database_path}")
        
        if not os.path.exists(self.database_path):
            raise FileNotFoundError(f"Database file not found: {self.database_path}")
        
        self.data = pd.read_csv(self.database_path)
        print(f"Loaded {len(self.data)} records")
        
        # Validate required columns
        required_cols = ['Time', 'Power', 'Head', 'Flow', 'Price', 'Date']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return self.data
    
    def extract_daily_prices(self):
        """Extract daily 24-hour price series from the database."""
        if self.data is None:
            self.load_database()
        
        print("Extracting daily price series...")
        
        daily_prices = {}
        unique_dates = self.data['Date'].unique()
        
        for date in unique_dates:
            date_data = self.data[self.data['Date'] == date].copy()
            
            # Sort by time to ensure correct hourly order
            date_data = date_data.sort_values('Time')
            
            # Check if we have complete 24-hour data
            if len(date_data) != 24:
                print(f"Warning: {date} has {len(date_data)} hours, skipping")
                continue
            
            # Check if hours are 0-23
            expected_hours = set(range(24))
            actual_hours = set(date_data['Time'].values)
            if expected_hours != actual_hours:
                print(f"Warning: {date} missing hours {expected_hours - actual_hours}, skipping")
                continue
            
            # Extract price series
            prices = date_data['Price'].values.tolist()
            daily_prices[date] = prices
        
        self.daily_prices = daily_prices
        print(f"Extracted {len(daily_prices)} complete daily price series")
        
        return daily_prices

# %% MIQP Formulation Runner
class MIQPFormulationRunner:
    """Run different MIQP formulations on historical price data."""
    
    def __init__(self):
        """Initialize the formulation runner."""
        self.results = {
            'global_linear': [],
            'neural_network': [],
            'piecewise': []
        }
        self.benchmark_results = {
            'global_linear': [],
            'neural_network': [],
            'piecewise': []
        }
        
    def run_global_linear(self, date_str, prices_24h):
        """Run Global Linear MIQP formulation."""
        try:
            start_time = time.time()
            
            # Create and solve optimizer (using exact same settings as original)
            T = 24
            optimizer = MILPOptimizer(T, prices_24h)
            results, metrics = optimizer.solve()
            
            solution_time = time.time() - start_time
            
            if results is None:
                return None, None
            
            # Post-process results to ensure idle mode values are exactly 0 (same as original)
            z_I_values = results['z_I']
            corrected_p_t_T = results['p_t_T'].copy()
            corrected_p_t_P = results['p_t_P'].copy()
            corrected_q_t = results['q_t'].copy()
            
            for t in range(len(z_I_values)):
                if z_I_values[t] > 0.5:  # If idle mode is active
                    corrected_p_t_T[t] = 0.0
                    corrected_p_t_P[t] = 0.0
                    corrected_q_t[t] = 0.0
            
            results['p_t_T'] = corrected_p_t_T
            results['p_t_P'] = corrected_p_t_P
            results['q_t'] = corrected_q_t
            
            # Calculate total power and volume (same as original)
            power_values = [p_t_T + p_t_P for p_t_T, p_t_P in zip(results['p_t_T'], results['p_t_P'])]
            volume_values = [h_vlow_coeff_lin[0] * h + h_vlow_coeff_lin[1] for h in results['h_t']]
            
            # Run simulation (same as original)
            head_init_val = torch.tensor(head_init, dtype=torch.float32, device=device)
            v_low_init_val = torch.tensor(v_low_init, dtype=torch.float32, device=device)
            
            params = HydroParams_Linear(
                head_init=head_init_val,
                v_low_init=v_low_init_val,
                neg_min=neg_min, neg_max=neg_max,
                pos_min=pos_min, pos_max=pos_max,
                predict_q_poly=predict_q_poly,
                h_to_v_low_fitted=h_to_v_low_fitted,
                v_low_to_h_fitted=v_low_to_h_fitted
            )
            
            simulator = SimLayer_Linear(params)
            
            p_tensor = torch.tensor(power_values, dtype=torch.float32, device=device)
            q_tensor = torch.tensor(results['q_t'], dtype=torch.float32, device=device)
            h_tensor = torch.tensor(results['h_t'], dtype=torch.float32, device=device)
            
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)
            
            # Calculate simulation profit (same as original)
            da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
            profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_tensor[:len(p_sim)], v_low_sim, da_prices_tensor[:len(p_sim)]
            )
            
            # Store detailed results (exact same format as original)
            detailed = []
            for hour in range(T):
                detailed.append({
                    'date': date_str,
                    'hour': hour,
                    'power': power_values[hour],
                    'head': results['h_t'][hour],
                    'volume': volume_values[hour],
                    'flow': results['q_t'][hour],
                    'price': prices_24h[hour]
                })
            
            # Store benchmark results (exact same format as original)
            benchmark = {
                'Date': date_str,
                'Solving Time (s)': solution_time,
                'MIP Gap': metrics['MIPGap'],
                'Binary Variables': metrics['NumBinVars'],
                'Continuous Variables': metrics['NumVars'] - metrics['NumBinVars'],
                'Total Constraints': metrics['NumConstrs'],
                'Expected Profit (€)': metrics['ExpectedProfit'],
                'SI Penalty (€)': si_penalty.item(),
                'Vol Penalty (€)': vol_penalty.item(),
                'Op Cost (€)': op_cost.item(),
                'Ex-post Profit (€)': profit.item()
            }
            
            return detailed, benchmark
            
        except Exception as e:
            return None, None
    
    def run_neural_network(self, date_str, prices_24h):
        """Run Neural Network MIQP formulation."""
        try:
            start_time = time.time()
            
            # Create and solve model (using exact same settings as original)
            model = create_uphes_miqp_model(24, prices_24h)
            solver = pyo.SolverFactory("gurobi")
            solver.options["TimeLimit"] = 3600  # 1 hour time limit
            solver.options["MIPGap"] = 0.01  # 1% MIP gap
            results = solver.solve(model, tee=False)
            
            solution_time = time.time() - start_time
            
            # Check solution status
            term_condition = str(results.solver.termination_condition).lower()
            if term_condition in ("infeasible", "infeasibleorunbounded"):
                return None, None
            
            # Get MIP gap (same as original)
            mip_gap = np.abs((results.problem.upper_bound - results.problem.lower_bound)) / np.abs(results.problem.lower_bound)
            
            # Count variables and constraints (same as original)
            total_vars = model.nvariables()
            total_constraints = model.nconstraints()
            binary_vars = 0
            continuous_vars = 0
            for var in model.component_objects(pyo.Var, active=True):
                for index in var:
                    if var[index].domain is pyo.Binary:
                        binary_vars += 1
                    else:
                        continuous_vars += 1
            
            # Extract optimization results (same as original)
            expected_profit = pyo.value(model.obj)
            
            # Extract schedules (same as original)
            p_total_schedule = [model.p_T[t]() + model.p_P[t]() for t in range(24)]
            q_schedule = [model.q[t]() for t in range(24)]
            h_schedule = [model.h[t]() for t in range(24)]
            v_low_schedule = [model.v_low[t]() for t in range(24)]
            
            # Run simulation (same as original)
            head_init_val = torch.tensor(77.0, dtype=torch.float32, device=device)
            v_low_init_val = h_to_v_low_fitted(head_init_val)
            
            params = HydroParams_NN(
                head_init=head_init_val,
                v_low_init=v_low_init_val,
                neg_min=neg_min, neg_max=neg_max,
                pos_min=pos_min, pos_max=pos_max,
                predict_q_poly=predict_q_poly,
                h_to_v_low_fitted=h_to_v_low_fitted,
                v_low_to_h_fitted=v_low_to_h_fitted
            )
            
            simulator = SimLayer_NN(params)
            
            # Convert to tensors (same as original)
            p_tensor = torch.tensor(p_total_schedule, dtype=torch.float32, device=device)
            q_tensor = torch.tensor(q_schedule, dtype=torch.float32, device=device)
            h_tensor = torch.tensor(h_schedule, dtype=torch.float32, device=device)
            
            # Run simulation (same as original)
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)
            
            # Calculate simulation profit (same as original)
            da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
            profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_tensor[:len(p_sim)], v_low_sim, da_prices_tensor[:len(p_sim)]
            )
            
            # Store detailed results (exact same format as original)
            detailed = []
            for hour in range(24):
                detailed.append({
                    'date': date_str,
                    'hour': hour,
                    'power': p_total_schedule[hour],
                    'head': h_schedule[hour],
                    'volume': v_low_schedule[hour],
                    'flow': q_schedule[hour],
                    'price': prices_24h[hour]
                })
            
            # Store benchmark results (exact same format as original)
            benchmark = {
                'Date': date_str,
                'Solving Time (s)': solution_time,
                'MIP Gap': mip_gap,
                'Binary Variables': binary_vars,
                'Continuous Variables': continuous_vars,
                'Total Constraints': total_constraints,
                'Expected Profit (€)': expected_profit,
                'SI Penalty (€)': si_penalty.item(),
                'Vol Penalty (€)': vol_penalty.item(),
                'Op Cost (€)': op_cost.item(),
                'Ex-post Profit (€)': profit.item()
            }
            
            return detailed, benchmark
            
        except Exception as e:
            return None, None
    
    def run_piecewise(self, date_str, prices_24h):
        """Run Piecewise MIQP formulation."""
        try:
            start_time = time.time()
            
            # Create and solve optimizer (using exact same settings as original)
            T = 24
            optimizer = PiecewiseMILPOptimizerSOS2(
                T=T, 
                DA_prices=prices_24h,
                num_segments_h=10,      # Same as original
                num_segments_p_pump=10,
                num_segments_p_turbine=10
            )
            optimizer.model.Params.MIPGap = 0.01  # 1% optimality gap
            optimizer.model.Params.TimeLimit = 3600  # 60 minute time limit
            
            results, metrics = optimizer.solve()
            
            solution_time = time.time() - start_time
            
            if results is None:
                return None, None
            
            # Run simulation (same as original)
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
            
            # Convert to tensors (same as original)
            p_tensor = torch.tensor(results['p'], dtype=torch.float32, device=device)
            q_tensor = torch.tensor(results['q'], dtype=torch.float32, device=device)
            h_tensor = torch.tensor(results['h'], dtype=torch.float32, device=device)
            
            # Run simulation (same as original)
            p_sim, q_sim, h_sim, v_low_sim = simulator.simulate_operation(p_tensor, q_tensor, h_tensor)
            
            # Calculate simulation profit (same as original)
            da_prices_tensor = torch.tensor(prices_24h, dtype=torch.float32, device=device)
            profit, si_penalty, vol_penalty, op_cost = simulator.calc_profit(
                p_sim, p_tensor[:len(p_sim)], v_low_sim, da_prices_tensor[:len(p_sim)]
            )
            
            # Store detailed results (exact same format as original)
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
            
            # Store benchmark results (exact same format as original)
            benchmark = {
                'Date': date_str,
                'Solving Time (s)': solution_time,
                'MIP Gap': metrics['MIPGap'],
                'Binary Variables': metrics['NumBinVars'],
                'Continuous Variables': metrics['NumVars'] - metrics['NumBinVars'],
                'Total Constraints': metrics['NumConstrs'],
                'Expected Profit (€)': metrics['ExpectedProfit'],
                'SI Penalty (€)': si_penalty.item(),
                'Vol Penalty (€)': vol_penalty.item(),
                'Op Cost (€)': op_cost.item(),
                'Ex-post Profit (€)': profit.item()
            }
            
            return detailed, benchmark
            
        except Exception as e:
            return None, None
    
    def process_all_databases_by_formulation(self, databases):
        """Process all databases by formulation (Global Linear -> Piecewise -> Neural Network)."""
        
        # First load all databases
        all_database_prices = {}
        for db_name, db_path in databases:
            try:
                print(f"Loading {db_name} database...")
                processor = HistoricalDatabaseProcessor(db_path)
                daily_prices = processor.extract_daily_prices()
                
                if not daily_prices:
                    print(f"No valid daily prices found in {db_name} database")
                    continue
                    
                all_database_prices[db_name] = daily_prices
                print(f"✓ {db_name}: {len(daily_prices)} days loaded")
                
            except Exception as e:
                print(f"Error loading {db_name} database: {e}")
                continue
        
        if not all_database_prices:
            print("No databases loaded successfully!")
            return
        
        # Show processing plan
        print(f"\n📋 Processing Plan:")
        print(f"   • Databases: {', '.join(all_database_prices.keys())}")
        print(f"   • Total days: {sum(len(prices) for prices in all_database_prices.values())}")
        print(f"   • Formulations: Global Linear → Piecewise → Neural Network")
        print(f"   • Results will be saved immediately after each database-formulation completion")
        
        # Process formulations in order: Global Linear -> Piecewise -> Neural Network
        formulation_order = [
            # ('global_linear', 'Global Linear'),
            # ('piecewise', 'Piecewise'),
            ('neural_network', 'Neural Network')
        ]
        
        total_databases = len(all_database_prices)
        total_formulations = len(formulation_order)
        
        for form_idx, (formulation_key, formulation_name) in enumerate(formulation_order, 1):
            print(f"\n{'='*80}")
            print(f"PROCESSING ALL DATABASES WITH {formulation_name.upper()} ({form_idx}/{total_formulations})")
            print(f"{'='*80}")
            
            # Check if formulation is available
            if formulation_key == 'global_linear':
                try:
                    MILPOptimizer
                except NameError:
                    print(f"Skipping {formulation_name} (not imported)")
                    continue
            elif formulation_key == 'piecewise':
                try:
                    PiecewiseMILPOptimizerSOS2
                except NameError:
                    print(f"Skipping {formulation_name} (not imported)")
                    continue
            elif formulation_key == 'neural_network':
                try:
                    create_uphes_miqp_model
                except NameError:
                    print(f"Skipping {formulation_name} (not imported)")
                    continue
            
            # Process all databases with this formulation
            for db_idx, (db_name, daily_prices) in enumerate(all_database_prices.items(), 1):
                print(f"\n{'-'*60}")
                print(f"Processing {db_name} database with {formulation_name} ({db_idx}/{total_databases})")
                print(f"{'-'*60}")
                
                detailed_results = []
                benchmark_results = []
                successful_dates = 0
                failed_dates = 0
                
                total_dates = len(daily_prices)
                for date_idx, (date_str, prices_24h) in enumerate(daily_prices.items(), 1):
                    print(f"  {date_str} ({date_idx}/{total_dates})...", end=" ")
                    
                    # Run the specific formulation
                    if formulation_key == 'global_linear':
                        detailed, benchmark = self.run_global_linear(date_str, prices_24h)
                    elif formulation_key == 'piecewise':
                        detailed, benchmark = self.run_piecewise(date_str, prices_24h)
                    elif formulation_key == 'neural_network':
                        detailed, benchmark = self.run_neural_network(date_str, prices_24h)
                    
                    if detailed and benchmark:
                        detailed_results.extend(detailed)
                        benchmark_results.append(benchmark)
                        successful_dates += 1
                        expected_profit = benchmark.get('Expected Profit (€)', 0.0)
                        expost_profit = benchmark.get('Ex-post Profit (€)', 0.0)
                        solve_time = benchmark.get('Solving Time (s)', 0.0)
                        print(f"✓ Expected: {expected_profit:.2f} €, Ex-post: {expost_profit:.2f} € ({solve_time:.1f}s)")
                    else:
                        failed_dates += 1
                        print("❌ Failed")
                
                # Show processing summary
                print(f"\n📈 Processing Summary:")
                print(f"   • Successful: {successful_dates}/{total_dates} days")
                if failed_dates > 0:
                    print(f"   • Failed: {failed_dates}/{total_dates} days")
                
                # Save results for this database-formulation combination immediately
                if detailed_results:
                    self.save_formulation_results(db_name, formulation_key, detailed_results, benchmark_results)
                else:
                    print(f"\n❌ {db_name}-{formulation_key}: No successful solutions - no files saved")
                    print(f"{'─'*60}")
            
            # Summary after completing all databases for this formulation
            print(f"\n🎉 Completed {formulation_name} for all databases!")
            completed_folders = []
            for db_name in all_database_prices.keys():
                folder_name = f"{db_name}_{formulation_key}"
                if Path(folder_name).exists():
                    completed_folders.append(folder_name)
            
            if completed_folders:
                print(f"✅ Created folders: {', '.join(completed_folders)}")
            else:
                print(f"❌ No folders created for {formulation_name}")
            print(f"{'='*80}")
    
    def save_formulation_results(self, db_name, formulation_key, detailed_results, benchmark_results):
        """Save results for a specific database-formulation combination."""
        # Create folder for this database-formulation combination
        folder_name = f"{db_name}_{formulation_key}"
        folder_path = Path(folder_name)
        folder_path.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_df = pd.DataFrame(detailed_results)
        detailed_file = folder_path / "detailed_results.csv"
        detailed_df.to_csv(detailed_file, index=False)
        
        # Save benchmark results
        benchmark_df = pd.DataFrame(benchmark_results)
        benchmark_file = folder_path / "benchmark_results.csv"
        benchmark_df.to_csv(benchmark_file, index=False)
        
        print(f"\n📁 Results saved to {folder_name}/")
        print(f"   ├── detailed_results.csv ({len(detailed_results)} hourly records)")
        print(f"   └── benchmark_results.csv ({len(benchmark_results)} daily records)")
        
        # Display summary statistics
        if benchmark_results:
            expected_profits = [b.get('Expected Profit (€)', 0.0) for b in benchmark_results]
            expost_profits = [b.get('Ex-post Profit (€)', 0.0) for b in benchmark_results]
            solve_times = [b.get('Solving Time (s)', 0.0) for b in benchmark_results]
            mip_gaps = [b.get('MIP Gap', 0.0) for b in benchmark_results if b.get('MIP Gap', 0.0) > 0]
            si_penalties = [b.get('SI Penalty (€)', 0.0) for b in benchmark_results]
            
            print(f"\n📊 Summary for {folder_name}:")
            print(f"   • Days processed: {len(benchmark_results)}")
            print(f"   • Expected profit: {np.mean(expected_profits):.2f} € (range: {min(expected_profits):.2f} - {max(expected_profits):.2f} €)")
            print(f"   • Ex-post profit: {np.mean(expost_profits):.2f} € (range: {min(expost_profits):.2f} - {max(expost_profits):.2f} €)")
            print(f"   • Average solve time: {np.mean(solve_times):.2f} s (range: {min(solve_times):.2f} - {max(solve_times):.2f} s)")
            print(f"   • Average SI penalty: {np.mean(si_penalties):.2f} €")
            if mip_gaps:
                print(f"   • Average MIP gap: {np.mean(mip_gaps):.3f}")
            else:
                print(f"   • MIP gap: 0.000 (all optimal)")
            
            # Show top 3 most profitable days (by ex-post profit)
            sorted_results = sorted(benchmark_results, key=lambda x: x.get('Ex-post Profit (€)', 0.0), reverse=True)
            print(f"   • Top 3 profitable days (ex-post):")
            for i, result in enumerate(sorted_results[:3], 1):
                print(f"     {i}. {result['Date']}: {result.get('Ex-post Profit (€)', 0.0):.2f} €")
        
        print(f"{'─'*60}")
        
        return folder_path
        
    def process_database(self, database_name, daily_prices):
        """Legacy method - kept for compatibility but not used in new workflow."""
        # This method is no longer used but kept for compatibility
        pass

# %% Main Execution Function
def main():
    """Main execution function to process all historical databases."""
    print("="*80)
    print("HISTORICAL OPERATION DATABASE SOLVER")
    print("="*80)
    
    # Get root directory for file paths
    current_dir = Path(__file__).parent
    miqp_dir = current_dir.parent
    root_dir = miqp_dir.parent
    
    # Define database files to process
    data_dir = root_dir / "Data"  # Updated to use root_dir instead of relative path
    print(f"Looking for database files in: {data_dir.absolute()}")
    
    databases = [
        ("euclidean", data_dir / "historical_database_euclidean.csv"),
        # ("pearson", data_dir / "historical_database_pearson.csv"),
        # ("cosine", data_dir / "historical_database_cosine.csv")  # Commented out
    ]
    
    # Check if database files exist
    for db_name, db_path in databases:
        if db_path.exists():
            print(f"✓ Found {db_name} database: {db_path}")
        else:
            print(f"✗ Missing {db_name} database: {db_path}")
    
    missing_databases = [db_name for db_name, db_path in databases if not db_path.exists()]
    if missing_databases:
        print(f"\n⚠️  Missing database files: {', '.join(missing_databases)}")
        print("Please run the historical_price_database_generator.py script first to create these files.")
        print("Expected files:")
        for db_name, db_path in databases:
            print(f"  - {db_path}")
        return
    
    print()
    
    # Check if all required imports succeeded
    required_modules = []
    try:
        MILPOptimizer
        required_modules.append("Global Linear")
    except NameError:
        pass
    
    try:
        create_uphes_miqp_model
        required_modules.append("Neural Network") 
    except NameError:
        pass
        
    try:
        PiecewiseMILPOptimizerSOS2
        required_modules.append("Piecewise")
    except NameError:
        pass
    
    if not required_modules:
        print("✗ No MIQP formulations successfully imported. Cannot proceed.")
        return
    else:
        print(f"✓ Successfully imported formulations: {', '.join(required_modules)}")
    
    # Initialize runner
    runner = MIQPFormulationRunner()
    
    # Process all databases by formulation (new workflow)
    runner.process_all_databases_by_formulation(databases)
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print("Processing order: Global Linear → Piecewise → Neural Network")
    print("All results saved immediately after each database-formulation completion.")
    print("\nFinal output structure:")
    formulations = ['global_linear', 'piecewise', 'neural_network']
    for db_name, _ in databases:
        for formulation in formulations:
            folder_name = f"{db_name}_{formulation}"
            folder_path = Path(folder_name)
            if folder_path.exists():
                print(f"  ✅ {folder_name}/")
                print(f"    ├── detailed_results.csv")
                print(f"    └── benchmark_results.csv")
            else:
                print(f"  ❌ {folder_name}/ (not created - check for errors above)")

if __name__ == "__main__":
    main()