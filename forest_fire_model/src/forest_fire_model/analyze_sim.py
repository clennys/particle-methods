"""
Simulation Data Analysis Script

This script loads saved simulation data from pickle files and creates
comprehensive visualizations and analysis for research reports.

Usage:
    python analyze_simulation.py path/to/simulation_data.pkl
    python analyze_simulation.py --directory simulation_data/  # Analyze all files
    python analyze_simulation.py --compare file1.pkl file2.pkl  # Compare simulations
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class SimulationAnalyzer:
    """Analyzes and visualizes forest fire simulation data"""
    
    def __init__(self, data_file):
        """Load simulation data from pickle file"""
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.metadata = self.data['metadata']
        self.time_series = self.data['time_series']
        self.spatial_data = self.data['spatial_data']
        self.summary_stats = self.data['summary_stats']
        
        # Convert time series to DataFrame for easier analysis
        self.df = pd.DataFrame(self.time_series)
        
        print(f"Loaded simulation data from {data_file}")
        print(f"Map type: {self.metadata['map_type']}")
        print(f"Grid size: {self.metadata['grid_size']}")
        print(f"Simulation duration: {len(self.df)} frames")
    
    def create_comprehensive_report(self, output_dir="analysis_output"):
        """Generate a comprehensive analysis report with all visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating comprehensive analysis report in {output_dir}/")
        
        # 1. Fire progression over time
        self.plot_fire_progression(save_path=f"{output_dir}/fire_progression.png")
        
        # 2. Particle dynamics
        self.plot_particle_dynamics(save_path=f"{output_dir}/particle_dynamics.png")
        
        # 3. Spatial analysis
        self.plot_spatial_analysis(save_path=f"{output_dir}/spatial_analysis.png")
        
        # 4. Burn rate analysis
        self.plot_burn_rate_analysis(save_path=f"{output_dir}/burn_rate_analysis.png")
        
        # 5. Environmental effects
        self.plot_environmental_effects(save_path=f"{output_dir}/environmental_effects.png")
        
        # 6. Summary statistics
        self.create_summary_table(save_path=f"{output_dir}/summary_stats.png")
        
        # 7. Parameter correlation analysis
        self.plot_parameter_correlations(save_path=f"{output_dir}/parameter_correlations.png")
        
        # 8. Fire spread pattern analysis
        self.plot_fire_spread_patterns(save_path=f"{output_dir}/fire_spread_patterns.png")
        
        print(f"Analysis complete! Check {output_dir}/ for all generated plots.")
        
        return output_dir
    
    def plot_fire_progression(self, save_path=None):
        """Plot fire progression over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fire Progression Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Cell state evolution
        axes[0,0].plot(self.df['frame'], self.df['fuel_cells'], label='Fuel Remaining', color='green', linewidth=2)
        axes[0,0].plot(self.df['frame'], self.df['burning_cells'], label='Currently Burning', color='red', linewidth=2)
        axes[0,0].plot(self.df['frame'], self.df['burned_cells'], label='Burned', color='black', linewidth=2)
        axes[0,0].set_xlabel('Time (frames)')
        axes[0,0].set_ylabel('Number of Cells')
        axes[0,0].set_title('Cell State Evolution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Fire spread distance
        axes[0,1].plot(self.df['frame'], self.df['fire_spread_distance'], color='orange', linewidth=2)
        axes[0,1].set_xlabel('Time (frames)')
        axes[0,1].set_ylabel('Distance (grid units)')
        axes[0,1].set_title('Fire Spread Distance from Ignition')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Active particles vs burning cells
        axes[1,0].plot(self.df['frame'], self.df['active_particles'], label='Active Particles', color='yellow', linewidth=2)
        ax2 = axes[1,0].twinx()
        ax2.plot(self.df['frame'], self.df['burning_cells'], label='Burning Cells', color='red', linewidth=2, alpha=0.7)
        axes[1,0].set_xlabel('Time (frames)')
        axes[1,0].set_ylabel('Active Particles', color='yellow')
        ax2.set_ylabel('Burning Cells', color='red')
        axes[1,0].set_title('Particles vs Burning Cells')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative burned percentage
        total_cells = self.metadata['grid_size'][0] * self.metadata['grid_size'][1]
        burned_percentage = (self.df['burned_cells'] / total_cells) * 100
        axes[1,1].plot(self.df['frame'], burned_percentage, color='darkred', linewidth=2)
        axes[1,1].set_xlabel('Time (frames)')
        axes[1,1].set_ylabel('Burned Area (%)')
        axes[1,1].set_title('Cumulative Burned Area')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fire progression plot saved to {save_path}")
        plt.show()
    
    def plot_particle_dynamics(self, save_path=None):
        """Analyze particle behavior and intensity"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Particle Dynamics Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Particle count over time
        axes[0,0].plot(self.df['frame'], self.df['active_particles'], color='orange', linewidth=2)
        axes[0,0].set_xlabel('Time (frames)')
        axes[0,0].set_ylabel('Number of Active Particles')
        axes[0,0].set_title('Active Particles Over Time')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Particle intensity evolution
        axes[0,1].plot(self.df['frame'], self.df['particle_intensity_avg'], label='Average Intensity', color='blue', linewidth=2)
        axes[0,1].plot(self.df['frame'], self.df['particle_intensity_max'], label='Maximum Intensity', color='red', linewidth=2)
        axes[0,1].set_xlabel('Time (frames)')
        axes[0,1].set_ylabel('Particle Intensity')
        axes[0,1].set_title('Particle Intensity Evolution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Particle efficiency (particles per burning cell)
        efficiency = np.where(self.df['burning_cells'] > 0, 
                            self.df['active_particles'] / self.df['burning_cells'], 0)
        axes[1,0].plot(self.df['frame'], efficiency, color='green', linewidth=2)
        axes[1,0].set_xlabel('Time (frames)')
        axes[1,0].set_ylabel('Particles per Burning Cell')
        axes[1,0].set_title('Particle Generation Efficiency')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Wind effect on particles
        axes[1,1].scatter(self.df['wind_effect_strength'], self.df['active_particles'], 
                         alpha=0.6, c=self.df['frame'], cmap='viridis')
        axes[1,1].set_xlabel('Wind Effect Strength')
        axes[1,1].set_ylabel('Active Particles')
        axes[1,1].set_title('Wind Effect on Particle Count')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Particle dynamics plot saved to {save_path}")
        plt.show()
    
    def plot_spatial_analysis(self, save_path=None):
        """Analyze spatial patterns and maps"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spatial Analysis', fontsize=16, fontweight='bold')
        
        # Initial vs Final state
        axes[0,0].imshow(self.spatial_data['initial_grid'].T, origin='lower', cmap='RdYlGn_r')
        axes[0,0].set_title('Initial State')
        axes[0,0].set_xlabel('X')
        axes[0,0].set_ylabel('Y')
        
        axes[0,1].imshow(self.spatial_data['final_grid'].T, origin='lower', cmap='RdYlGn_r')
        axes[0,1].set_title('Final State')
        axes[0,1].set_xlabel('X')
        axes[0,1].set_ylabel('Y')
        
        # Fuel types
        im3 = axes[0,2].imshow(self.spatial_data['fuel_types'].T, origin='lower', cmap='YlOrBr')
        axes[0,2].set_title('Fuel Type Distribution')
        axes[0,2].set_xlabel('X')
        axes[0,2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[0,2], label='Fuel Type')
        
        # Moisture map
        im4 = axes[1,0].imshow(self.spatial_data['moisture_map'].T, origin='lower', cmap='Blues')
        axes[1,0].set_title('Moisture Distribution')
        axes[1,0].set_xlabel('X')
        axes[1,0].set_ylabel('Y')
        plt.colorbar(im4, ax=axes[1,0], label='Moisture Level')
        
        # Terrain
        im5 = axes[1,1].imshow(self.spatial_data['terrain'].T, origin='lower', cmap='terrain')
        axes[1,1].set_title('Terrain Elevation')
        axes[1,1].set_xlabel('X')
        axes[1,1].set_ylabel('Y')
        plt.colorbar(im5, ax=axes[1,1], label='Elevation')
        
        # Wind field
        wind_field = self.spatial_data['wind_field']
        wind_magnitude = np.sqrt(wind_field[:,:,0]**2 + wind_field[:,:,1]**2)
        im6 = axes[1,2].imshow(wind_magnitude.T, origin='lower', cmap='plasma')
        
        # Add wind direction arrows (downsampled)
        skip = max(1, wind_field.shape[0] // 15)
        x = np.arange(0, wind_field.shape[0], skip)
        y = np.arange(0, wind_field.shape[1], skip)
        X, Y = np.meshgrid(x, y)
        U = wind_field[::skip, ::skip, 0].T
        V = wind_field[::skip, ::skip, 1].T
        axes[1,2].quiver(X, Y, U, V, color='white', alpha=0.7, scale=10)
        
        axes[1,2].set_title('Wind Field')
        axes[1,2].set_xlabel('X')
        axes[1,2].set_ylabel('Y')
        plt.colorbar(im6, ax=axes[1,2], label='Wind Magnitude')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spatial analysis plot saved to {save_path}")
        plt.show()
    
    def plot_burn_rate_analysis(self, save_path=None):
        """Analyze burn rates and fire behavior"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Burn Rate Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Burn rate over time
        axes[0,0].plot(self.df['frame'], self.df['burn_rate'], color='red', linewidth=2)
        axes[0,0].set_xlabel('Time (frames)')
        axes[0,0].set_ylabel('Cells Burned per Frame')
        axes[0,0].set_title('Instantaneous Burn Rate')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Burn rate histogram
        axes[0,1].hist(self.df['burn_rate'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Burn Rate (cells/frame)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Burn Rate Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Cumulative burn rate vs particle count
        axes[1,0].scatter(self.df['active_particles'], self.df['burn_rate'], 
                         alpha=0.6, c=self.df['frame'], cmap='plasma')
        axes[1,0].set_xlabel('Active Particles')
        axes[1,0].set_ylabel('Burn Rate')
        axes[1,0].set_title('Particles vs Burn Rate')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Moving average burn rate
        window = min(10, len(self.df) // 5)
        if window > 1:
            moving_avg = self.df['burn_rate'].rolling(window=window).mean()
            axes[1,1].plot(self.df['frame'], self.df['burn_rate'], alpha=0.3, color='red', label='Raw')
            axes[1,1].plot(self.df['frame'], moving_avg, color='darkred', linewidth=2, label=f'{window}-frame Moving Avg')
            axes[1,1].set_xlabel('Time (frames)')
            axes[1,1].set_ylabel('Burn Rate')
            axes[1,1].set_title('Smoothed Burn Rate')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Burn rate analysis plot saved to {save_path}")
        plt.show()
    
    def plot_environmental_effects(self, save_path=None):
        """Analyze environmental factor effects"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Environmental Effects Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Wind effect on fire spread
        axes[0,0].scatter(self.df['wind_effect_strength'], self.df['fire_spread_distance'], 
                         alpha=0.6, c=self.df['frame'], cmap='viridis')
        axes[0,0].set_xlabel('Wind Strength')
        axes[0,0].set_ylabel('Fire Spread Distance')
        axes[0,0].set_title('Wind Effect on Fire Spread')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Fuel type effectiveness
        fuel_types = self.spatial_data['fuel_types']
        final_grid = self.spatial_data['final_grid']
        
        # Calculate burn probability by fuel type
        fuel_values = np.unique(fuel_types)
        burn_probabilities = []
        fuel_labels = []
        
        for fuel_val in fuel_values:
            fuel_mask = (fuel_types == fuel_val)
            total_fuel = np.sum(fuel_mask)
            if total_fuel > 0:
                burned_fuel = np.sum((final_grid == 2) & fuel_mask)  # CellState.BURNED = 2
                burn_prob = burned_fuel / total_fuel
                burn_probabilities.append(burn_prob)
                fuel_labels.append(f'Fuel {fuel_val:.1f}')
        
        axes[0,1].bar(fuel_labels, burn_probabilities, color='orange', alpha=0.7)
        axes[0,1].set_xlabel('Fuel Type')
        axes[0,1].set_ylabel('Burn Probability')
        axes[0,1].set_title('Fuel Type Burn Effectiveness')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Moisture vs burn probability
        moisture_map = self.spatial_data['moisture_map']
        
        # Create moisture bins and calculate burn probability
        moisture_bins = np.linspace(0, 1, 10)
        moisture_burn_probs = []
        bin_centers = []
        
        for i in range(len(moisture_bins)-1):
            moisture_mask = ((moisture_map >= moisture_bins[i]) & 
                           (moisture_map < moisture_bins[i+1]))
            total_cells = np.sum(moisture_mask)
            if total_cells > 0:
                burned_cells = np.sum((final_grid == 2) & moisture_mask)
                burn_prob = burned_cells / total_cells
                moisture_burn_probs.append(burn_prob)
                bin_centers.append((moisture_bins[i] + moisture_bins[i+1]) / 2)
        
        axes[1,0].plot(bin_centers, moisture_burn_probs, 'o-', color='blue', linewidth=2, markersize=6)
        axes[1,0].set_xlabel('Moisture Level')
        axes[1,0].set_ylabel('Burn Probability')
        axes[1,0].set_title('Moisture Effect on Burning')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Parameter timeline
        params = self.metadata['parameters']
        key_params = ['spread_rate', 'ignition_probability', 'wind_strength', 'base_moisture']
        param_values = [params.get(p, 0) for p in key_params]
        
        bars = axes[1,1].bar(key_params, param_values, color=['red', 'orange', 'blue', 'green'], alpha=0.7)
        axes[1,1].set_ylabel('Parameter Value')
        axes[1,1].set_title('Key Simulation Parameters')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, param_values):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Environmental effects plot saved to {save_path}")
        plt.show()
    
    def create_summary_table(self, save_path=None):
        """Create a summary statistics table"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare summary data
        summary_data = []
        for key, value in self.summary_stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            summary_data.append([key.replace('_', ' ').title(), formatted_value])
        
        # Add parameter information
        params = self.metadata['parameters']
        key_params = ['map_type', 'spread_rate', 'ignition_probability', 'wind_strength', 
                     'base_moisture', 'particle_lifetime', 'burnout_rate']
        
        summary_data.append(['', ''])  # Empty row
        summary_data.append(['SIMULATION PARAMETERS', ''])
        
        for param in key_params:
            if param in params:
                value = params[param]
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                summary_data.append([param.replace('_', ' ').title(), formatted_value])
        
        # Create table
        table = ax.table(cellText=summary_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                elif i > 0 and summary_data[i-1][0] == 'SIMULATION PARAMETERS':
                    cell.set_facecolor('#2196F3')
                    cell.set_text_props(weight='bold', color='white')
                elif i > 0 and summary_data[i-1][0] == '':
                    cell.set_facecolor('white')
                else:
                    cell.set_facecolor('#f1f1f1' if i % 2 == 0 else 'white')
        
        plt.title('Simulation Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary table saved to {save_path}")
        plt.show()
    
    def plot_parameter_correlations(self, save_path=None):
        """Analyze correlations between parameters and outcomes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Note about correlation analysis
        axes[0,0].text(0.5, 0.5, 'Correlation analysis requires\nmultiple simulations\nfor meaningful results', 
                      ha='center', va='center', fontsize=12, transform=axes[0,0].transAxes)
        axes[0,0].set_title('Parameter Correlation Matrix')
        axes[0,0].axis('off')
        
        # Plot 2: Time-based correlations
        if len(self.df) > 10:
            time_corr = self.df[['active_particles', 'burning_cells', 'burn_rate', 'fire_spread_distance']].corr()
            im = axes[0,1].imshow(time_corr, cmap='RdBu', vmin=-1, vmax=1)
            axes[0,1].set_xticks(range(len(time_corr.columns)))
            axes[0,1].set_yticks(range(len(time_corr.columns)))
            axes[0,1].set_xticklabels(time_corr.columns, rotation=45, ha='right')
            axes[0,1].set_yticklabels(time_corr.columns)
            
            # Add correlation values to cells
            for i in range(len(time_corr.columns)):
                for j in range(len(time_corr.columns)):
                    text = axes[0,1].text(j, i, f'{time_corr.iloc[i, j]:.2f}',
                                         ha="center", va="center", 
                                         color="black" if abs(time_corr.iloc[i, j]) < 0.5 else "white")
            
            plt.colorbar(im, ax=axes[0,1])
            axes[0,1].set_title('Time Series Correlations')
        
        # Plot 3: Parameter vs outcome scatter
        outcomes = [self.summary_stats['final_burned_percentage'], self.summary_stats['max_fire_spread_distance']]
        param_names = ['Final Burned %', 'Max Spread Distance']
        colors = ['red', 'orange']
        
        for i, (outcome, name, color) in enumerate(zip(outcomes, param_names, colors)):
            axes[1,0].bar(name, outcome, color=color, alpha=0.7)
        
        axes[1,0].set_ylabel('Value')
        axes[1,0].set_title('Key Outcomes')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Parameter sensitivity (relative to typical values)
        param_values = [
            self.metadata['parameters']['spread_rate'] / 0.01,  # Normalize to typical value
            self.metadata['parameters']['ignition_probability'] / 0.03,
            self.metadata['parameters']['wind_strength'] / 0.5,
            self.metadata['parameters']['base_moisture'] / 0.2
        ]
        param_labels = ['Spread Rate\n(rel to 0.01)', 'Ignition Prob\n(rel to 0.03)', 
                       'Wind Strength\n(rel to 0.5)', 'Base Moisture\n(rel to 0.2)']
        
        bars = axes[1,1].bar(param_labels, param_values, color=['red', 'orange', 'blue', 'green'], alpha=0.7)
        axes[1,1].axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Baseline')
        axes[1,1].set_ylabel('Relative to Baseline')
        axes[1,1].set_title('Parameter Values (Normalized)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter correlation plot saved to {save_path}")
        plt.show()
    
    def plot_fire_spread_patterns(self, save_path=None):
        """Analyze fire spread patterns and progression"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fire Spread Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Get burned area progression snapshots
        progression_data = self.spatial_data.get('burned_area_progression', [])
        
        if len(progression_data) >= 3:
            # Show progression at different time points
            time_points = [0, len(progression_data)//2, -1]  # Start, middle, end
            titles = ['Early Stage', 'Mid Stage', 'Final Stage']
            
            for i, (time_idx, title) in enumerate(zip(time_points, titles)):
                snapshot = progression_data[time_idx]
                im = axes[0,i].imshow(snapshot['burned_area'].T, origin='lower', cmap='Reds', vmin=0, vmax=1)
                axes[0,i].set_title(f'{title}\n(Frame {snapshot["frame"]})')
                axes[0,i].set_xlabel('X')
                axes[0,i].set_ylabel('Y')
                
                # Add ignition points
                for ignition_point in self.spatial_data['ignition_points']:
                    axes[0,i].plot(ignition_point[0], ignition_point[1], 'go', markersize=8)
        else:
            # If no progression data, show initial vs final
            axes[0,0].imshow(self.spatial_data['initial_grid'].T, origin='lower', cmap='RdYlGn_r')
            axes[0,0].set_title('Initial State')
            
            axes[0,1].imshow(self.spatial_data['final_grid'].T, origin='lower', cmap='RdYlGn_r')
            axes[0,1].set_title('Final State')
            
            axes[0,2].axis('off')
        
        # Spread rate analysis
        if len(self.df) > 1:
            axes[1,0].plot(self.df['frame'], np.cumsum(self.df['burn_rate']), color='red', linewidth=2)
            axes[1,0].set_xlabel('Time (frames)')
            axes[1,0].set_ylabel('Cumulative Cells Burned')
            axes[1,0].set_title('Cumulative Fire Spread')
            axes[1,0].grid(True, alpha=0.3)
            
            # Fire front velocity (rate of spread distance increase)
            spread_velocity = np.diff(self.df['fire_spread_distance'])
            axes[1,1].plot(self.df['frame'][1:], spread_velocity, color='orange', linewidth=2)
            axes[1,1].set_xlabel('Time (frames)')
            axes[1,1].set_ylabel('Spread Velocity (units/frame)')
            axes[1,1].set_title('Fire Front Velocity')
            axes[1,1].grid(True, alpha=0.3)
            
            # Fire shape analysis (if progression data available)
            if len(progression_data) >= 3:
                try:
                    from scipy import ndimage
                    perimeters = []
                    areas = []
                    for snapshot in progression_data:
                        burned_area = snapshot['burned_area']
                        area = np.sum(burned_area)
