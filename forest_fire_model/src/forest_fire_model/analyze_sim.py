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
from collections import Counter

warnings.filterwarnings("ignore")

# Set style for publication-quality plots
try:
    plt.style.use("seaborn-v0_8")
except:
    plt.style.use("seaborn")
sns.set_palette("husl")


class SimulationAnalyzer:
    """Analyzes and visualizes forest fire simulation data"""

    def __init__(self, data_file):
        """Load simulation data from pickle file - FIXED for array length issues"""
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        self.metadata = self.data["metadata"]
        self.time_series = self.data["time_series"]
        self.spatial_data = self.data["spatial_data"]
        self.summary_stats = self.data["summary_stats"]

        # Handle arrays of different lengths before creating DataFrame
        print(f"Loading simulation data from {data_file}")
        print(f"Map type: {self.metadata['map_type']}")
        print(f"Grid size: {self.metadata['grid_size']}")
        
        # Check the lengths of all arrays in time_series
        array_lengths = {}
        for key, value in self.time_series.items():
            if isinstance(value, list):
                array_lengths[key] = len(value)
        
        # Find the expected length
        if 'frame' in array_lengths:
            expected_length = array_lengths['frame']
            print(f"Expected array length (from 'frame'): {expected_length}")
        else:
            length_counts = Counter(array_lengths.values())
            expected_length = length_counts.most_common(1)[0][0]
            print(f"Expected array length (most common): {expected_length}")
        
        print(f"Simulation duration: {expected_length} frames")
        
        # Fix any arrays that don't match the expected length
        fixed_time_series = {}
        arrays_fixed = 0
        for key, value in self.time_series.items():
            if isinstance(value, list):
                current_length = len(value)
                if current_length == expected_length:
                    fixed_time_series[key] = value
                elif current_length < expected_length:
                    # Pad with appropriate default values
                    if key in ['frame']:
                        # For frame, continue the sequence
                        last_frame = value[-1] if value else 0
                        padding = list(range(last_frame + 1, last_frame + 1 + (expected_length - current_length)))
                        fixed_time_series[key] = value + padding
                    else:
                        # For other fields, pad with zeros
                        padding_needed = expected_length - current_length
                        fixed_time_series[key] = value + [0] * padding_needed
                    arrays_fixed += 1
                    print(f"⚠ Fixed {key}: {current_length} → {expected_length} elements")
                else:
                    # Truncate to expected length
                    fixed_time_series[key] = value[:expected_length]
                    arrays_fixed += 1
                    print(f"⚠ Fixed {key}: {current_length} → {expected_length} elements (truncated)")
            else:
                fixed_time_series[key] = value
        
        if arrays_fixed > 0:
            print(f"Fixed {arrays_fixed} arrays to consistent length")
        
        # Convert fixed time series to DataFrame
        try:
            self.df = pd.DataFrame(fixed_time_series)
            print(f"✓ Successfully created DataFrame with shape {self.df.shape}")
        except Exception as e:
            print(f"✗ Error creating DataFrame: {e}")
            # Last resort: create DataFrame with only consistent arrays
            consistent_arrays = {}
            for key, value in fixed_time_series.items():
                if isinstance(value, list) and len(value) == expected_length:
                    consistent_arrays[key] = value
            self.df = pd.DataFrame(consistent_arrays)
            print(f"✓ Created DataFrame with consistent arrays only: {self.df.shape}")
        
        # Add new columns with zeros if they don't exist (for backward compatibility)
        new_columns = [
            "particle_velocity_avg", "particle_velocity_max", 
            "wind_effect_std", "fire_front_cells"
        ]
        
        enhanced_data_available = []
        for col in new_columns:
            if col not in self.df.columns:
                self.df[col] = 0
            else:
                if any(self.df[col] != 0):
                    enhanced_data_available.append(col)
        
        if enhanced_data_available:
            print(f"Enhanced data available: {', '.join(enhanced_data_available)}")
        else:
            print("Note: Using legacy data format (some enhanced plots may be limited)")

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
        self.plot_environmental_effects(
            save_path=f"{output_dir}/environmental_effects.png"
        )

        # 6. Summary statistics
        self.create_summary_table(save_path=f"{output_dir}/summary_stats.png")

        # 7. Parameter correlation analysis
        self.plot_parameter_correlations(
            save_path=f"{output_dir}/parameter_correlations.png"
        )

        # 8. Fire spread pattern analysis
        self.plot_fire_spread_patterns(
            save_path=f"{output_dir}/fire_spread_patterns.png"
        )

        # 9. NEW: Create report-ready visualization with key graphics
        self.create_report_summary(save_path=f"{output_dir}/report_summary.png")

        print(f"Analysis complete! Check {output_dir}/ for all generated plots.")

        return output_dir

    def create_report_summary(self, save_path=None):
        """Create a comprehensive 6-panel summary for research reports"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Main title for the entire figure
        fig.suptitle(f"Fire Simulation Analysis Summary - {self.metadata['map_type'].title()} Map", 
                    fontsize=18, fontweight='bold', y=0.95)

        # Panel 1: Fire Progression Evolution (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.df["frame"], self.df["fuel_cells"], label="Fuel Remaining", 
                color="green", linewidth=2.5)
        ax1.plot(self.df["frame"], self.df["burning_cells"], label="Currently Burning", 
                color="red", linewidth=2.5)
        ax1.plot(self.df["frame"], self.df["burned_cells"], label="Burned", 
                color="black", linewidth=2.5)
        ax1.set_xlabel("Time (frames)", fontsize=12)
        ax1.set_ylabel("Number of Cells", fontsize=12)
        ax1.set_title("A) Cell State Evolution Over Time", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Burn Rate and Fire Spread Distance (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        color1 = 'tab:red'
        ax2.set_xlabel('Time (frames)', fontsize=12)
        ax2.set_ylabel('Burn Rate (cells/frame)', color=color1, fontsize=12)
        line1 = ax2.plot(self.df["frame"], self.df["burn_rate"], color=color1, 
                        linewidth=2.5, label='Burn Rate')
        ax2.tick_params(axis='y', labelcolor=color1)
        
        ax2_twin = ax2.twinx()
        color2 = 'tab:orange'
        ax2_twin.set_ylabel('Fire Spread Distance', color=color2, fontsize=12)
        line2 = ax2_twin.plot(self.df["frame"], self.df["fire_spread_distance"], 
                             color=color2, linewidth=2.5, label='Spread Distance')
        ax2_twin.tick_params(axis='y', labelcolor=color2)
        
        ax2.set_title("B) Fire Intensity and Spread Metrics", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = ['Burn Rate', 'Spread Distance']
        ax2.legend(lines, labels, loc='upper left', fontsize=10)

        # Panel 3: Particle Dynamics (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.df["frame"], self.df["active_particles"], color="orange", 
                linewidth=2.5, label="Active Particles")
        
        if "particle_velocity_avg" in self.df.columns and any(self.df["particle_velocity_avg"]):
            ax3_twin = ax3.twinx()
            ax3_twin.plot(self.df["frame"], self.df["particle_velocity_avg"], 
                         color="purple", linewidth=2.5, alpha=0.8, label="Avg Velocity")
            ax3_twin.set_ylabel("Particle Velocity", color="purple", fontsize=12)
            ax3_twin.tick_params(axis='y', labelcolor="purple")
        
        ax3.set_xlabel("Time (frames)", fontsize=12)
        ax3.set_ylabel("Active Particles", color="orange", fontsize=12)
        ax3.tick_params(axis='y', labelcolor="orange")
        ax3.set_title("C) Particle Activity and Movement", fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Panel 4: Cumulative Burned Area Percentage (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0])
        total_cells = self.metadata["grid_size"][0] * self.metadata["grid_size"][1]
        burned_percentage = (self.df["burned_cells"] / total_cells) * 100
        ax4.plot(self.df["frame"], burned_percentage, color="darkred", linewidth=3)
        ax4.fill_between(self.df["frame"], burned_percentage, alpha=0.3, color="red")
        ax4.set_xlabel("Time (frames)", fontsize=12)
        ax4.set_ylabel("Burned Area (%)", fontsize=12)
        ax4.set_title("D) Cumulative Burned Area Progression", fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add final percentage as text
        final_percentage = burned_percentage.iloc[-1] if len(burned_percentage) > 0 else 0
        ax4.text(0.98, 0.95, f'Final: {final_percentage:.1f}%', 
                transform=ax4.transAxes, fontsize=11, fontweight='bold',
                ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Panel 5: Environmental Effects Summary (bottom-center)
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Wind effect over time
        if "wind_effect_std" in self.df.columns and any(self.df["wind_effect_std"]):
            ax5.plot(self.df["frame"], self.df["wind_effect_strength"], 
                    label="Wind Strength", color="blue", linewidth=2.5)
            ax5_twin = ax5.twinx()
            ax5_twin.plot(self.df["frame"], self.df["wind_effect_std"], 
                         label="Wind Variation", color="cyan", linewidth=2.5, alpha=0.7)
            ax5_twin.set_ylabel("Wind Variation", color="cyan", fontsize=12)
            ax5_twin.tick_params(axis='y', labelcolor="cyan")
        else:
            ax5.plot(self.df["frame"], self.df["wind_effect_strength"], 
                    color="blue", linewidth=2.5, label="Wind Strength")
        
        ax5.set_xlabel("Time (frames)", fontsize=12)
        ax5.set_ylabel("Wind Strength", color="blue", fontsize=12)
        ax5.tick_params(axis='y', labelcolor="blue")
        ax5.set_title("E) Environmental Wind Effects", fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Panel 6: Key Simulation Parameters and Final Statistics (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Create summary statistics text
        stats_text = "F) Simulation Summary\n" + "="*25 + "\n"
        stats_text += f"Map Type: {self.metadata['map_type'].title()}\n"
        stats_text += f"Grid Size: {self.metadata['grid_size'][0]}×{self.metadata['grid_size'][1]}\n"
        stats_text += f"Duration: {len(self.df)} frames\n\n"
        
        stats_text += "Key Results:\n" + "-"*15 + "\n"
        stats_text += f"Final Burned: {self.summary_stats.get('final_burned_percentage', 0):.1f}%\n"
        stats_text += f"Max Spread: {self.summary_stats.get('max_fire_spread_distance', 0):.1f} units\n"
        stats_text += f"Peak Particles: {self.summary_stats.get('max_active_particles', 0)}\n"
        stats_text += f"Avg Burn Rate: {self.summary_stats.get('average_burn_rate', 0):.2f} cells/frame\n"
        stats_text += f"Fire Duration: {self.summary_stats.get('fire_duration', 0)} frames\n\n"
        
        stats_text += "Parameters:\n" + "-"*12 + "\n"
        params = self.metadata["parameters"]
        stats_text += f"Spread Rate: {params.get('spread_rate', 'N/A')}\n"
        stats_text += f"Ignition Prob: {params.get('ignition_probability', 'N/A')}\n"
        stats_text += f"Wind Strength: {params.get('wind_strength', 'N/A')}\n"
        stats_text += f"Base Moisture: {params.get('base_moisture', 'N/A')}\n"
        
        # Add ignition strategy info
        ignition_points = self.spatial_data.get("ignition_points", [])
        if len(ignition_points) > 1:
            stats_text += f"Ignition Points: {len(ignition_points)} locations\n"
        
        ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                family='monospace')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
            print(f"Report summary saved to {save_path}")
        # plt.show()

    def plot_fire_progression(self, save_path=None):
        """Plot fire progression over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Fire Progression Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Cell state evolution
        axes[0, 0].plot(
            self.df["frame"],
            self.df["fuel_cells"],
            label="Fuel Remaining",
            color="green",
            linewidth=2,
        )
        axes[0, 0].plot(
            self.df["frame"],
            self.df["burning_cells"],
            label="Currently Burning",
            color="red",
            linewidth=2,
        )
        axes[0, 0].plot(
            self.df["frame"],
            self.df["burned_cells"],
            label="Burned",
            color="black",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("Time (frames)")
        axes[0, 0].set_ylabel("Number of Cells")
        axes[0, 0].set_title("Cell State Evolution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Fire spread distance
        axes[0, 1].plot(
            self.df["frame"],
            self.df["fire_spread_distance"],
            color="orange",
            linewidth=2,
        )
        axes[0, 1].set_xlabel("Time (frames)")
        axes[0, 1].set_ylabel("Distance (grid units)")
        axes[0, 1].set_title("Fire Spread Distance from Ignition")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Active particles vs burning cells
        axes[1, 0].plot(
            self.df["frame"],
            self.df["active_particles"],
            label="Active Particles",
            color="yellow",
            linewidth=2,
        )
        ax2 = axes[1, 0].twinx()
        ax2.plot(
            self.df["frame"],
            self.df["burning_cells"],
            label="Burning Cells",
            color="red",
            linewidth=2,
            alpha=0.7,
        )
        axes[1, 0].set_xlabel("Time (frames)")
        axes[1, 0].set_ylabel("Active Particles", color="yellow")
        ax2.set_ylabel("Burning Cells", color="red")
        axes[1, 0].set_title("Particles vs Burning Cells")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cumulative burned percentage
        total_cells = self.metadata["grid_size"][0] * self.metadata["grid_size"][1]
        burned_percentage = (self.df["burned_cells"] / total_cells) * 100
        axes[1, 1].plot(
            self.df["frame"], burned_percentage, color="darkred", linewidth=2
        )
        axes[1, 1].set_xlabel("Time (frames)")
        axes[1, 1].set_ylabel("Burned Area (%)")
        axes[1, 1].set_title("Cumulative Burned Area")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Fire progression plot saved to {save_path}")
        # plt.show()

    def plot_particle_dynamics(self, save_path=None):
        """Analyze particle behavior and intensity - UPDATED"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Particle Dynamics Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Particle count over time
        axes[0, 0].plot(
            self.df["frame"], self.df["active_particles"], color="orange", linewidth=2
        )
        axes[0, 0].set_xlabel("Time (frames)")
        axes[0, 0].set_ylabel("Number of Active Particles")
        axes[0, 0].set_title("Active Particles Over Time")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Particle intensity evolution
        axes[0, 1].plot(
            self.df["frame"],
            self.df["particle_intensity_avg"],
            label="Average Intensity",
            color="blue",
            linewidth=2,
        )
        axes[0, 1].plot(
            self.df["frame"],
            self.df["particle_intensity_max"],
            label="Maximum Intensity",
            color="red",
            linewidth=2,
        )
        axes[0, 1].set_xlabel("Time (frames)")
        axes[0, 1].set_ylabel("Particle Intensity")
        axes[0, 1].set_title("Particle Intensity Evolution")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: NEW - Particle velocity evolution
        if "particle_velocity_avg" in self.df.columns and any(self.df["particle_velocity_avg"]):
            axes[1, 0].plot(
                self.df["frame"],
                self.df["particle_velocity_avg"],
                label="Average Velocity",
                color="green",
                linewidth=2,
            )
            axes[1, 0].plot(
                self.df["frame"],
                self.df["particle_velocity_max"],
                label="Maximum Velocity",
                color="darkgreen",
                linewidth=2,
            )
            axes[1, 0].set_xlabel("Time (frames)")
            axes[1, 0].set_ylabel("Particle Velocity")
            axes[1, 0].set_title("Particle Velocity Evolution")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            # Fallback to original particle efficiency plot
            efficiency = np.where(
                self.df["burning_cells"] > 0,
                self.df["active_particles"] / self.df["burning_cells"],
                0,
            )
            axes[1, 0].plot(self.df["frame"], efficiency, color="green", linewidth=2)
            axes[1, 0].set_xlabel("Time (frames)")
            axes[1, 0].set_ylabel("Particles per Burning Cell")
            axes[1, 0].set_title("Particle Generation Efficiency")
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Enhanced wind effect analysis
        if "wind_effect_std" in self.df.columns:
            # Show both average wind strength and variation
            ax_wind = axes[1, 1]
            ax_wind.plot(
                self.df["frame"],
                self.df["wind_effect_strength"],
                label="Wind Strength (Avg)",
                color="blue",
                linewidth=2,
            )
            
            ax_wind_std = ax_wind.twinx()
            ax_wind_std.plot(
                self.df["frame"],
                self.df["wind_effect_std"],
                label="Wind Variation (Std)",
                color="red",
                linewidth=2,
                alpha=0.7,
            )
            
            ax_wind.set_xlabel("Time (frames)")
            ax_wind.set_ylabel("Wind Strength", color="blue")
            ax_wind_std.set_ylabel("Wind Variation", color="red")
            ax_wind.set_title("Wind Effect Analysis")
            ax_wind.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax_wind.get_legend_handles_labels()
            lines2, labels2 = ax_wind_std.get_legend_handles_labels()
            ax_wind.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            # Original wind scatter plot
            axes[1, 1].scatter(
                self.df["wind_effect_strength"],
                self.df["active_particles"],
                alpha=0.6,
                c=self.df["frame"],
                cmap="viridis",
            )
            axes[1, 1].set_xlabel("Wind Effect Strength")
            axes[1, 1].set_ylabel("Active Particles")
            axes[1, 1].set_title("Wind Effect on Particle Count")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Particle dynamics plot saved to {save_path}")
        # plt.show()

    def plot_spatial_analysis(self, save_path=None):
        """Analyze spatial patterns and maps"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Spatial Analysis", fontsize=16, fontweight="bold")

        # Initial vs Final state
        axes[0, 0].imshow(
            self.spatial_data["initial_grid"].T, origin="lower", cmap="RdYlGn_r"
        )
        axes[0, 0].set_title("Initial State")
        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")

        axes[0, 1].imshow(
            self.spatial_data["final_grid"].T, origin="lower", cmap="RdYlGn_r"
        )
        axes[0, 1].set_title("Final State")
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Y")

        # Fuel types
        im3 = axes[0, 2].imshow(
            self.spatial_data["fuel_types"].T, origin="lower", cmap="YlOrBr"
        )
        axes[0, 2].set_title("Fuel Type Distribution")
        axes[0, 2].set_xlabel("X")
        axes[0, 2].set_ylabel("Y")
        plt.colorbar(im3, ax=axes[0, 2], label="Fuel Type")

        # Moisture map
        im4 = axes[1, 0].imshow(
            self.spatial_data["moisture_map"].T, origin="lower", cmap="Blues"
        )
        axes[1, 0].set_title("Moisture Distribution")
        axes[1, 0].set_xlabel("X")
        axes[1, 0].set_ylabel("Y")
        plt.colorbar(im4, ax=axes[1, 0], label="Moisture Level")

        # Terrain
        im5 = axes[1, 1].imshow(
            self.spatial_data["terrain"].T, origin="lower", cmap="terrain"
        )
        axes[1, 1].set_title("Terrain Elevation")
        axes[1, 1].set_xlabel("X")
        axes[1, 1].set_ylabel("Y")
        plt.colorbar(im5, ax=axes[1, 1], label="Elevation")

        # Wind field
        wind_field = self.spatial_data["wind_field"]
        wind_magnitude = np.sqrt(wind_field[:, :, 0] ** 2 + wind_field[:, :, 1] ** 2)
        im6 = axes[1, 2].imshow(wind_magnitude.T, origin="lower", cmap="plasma")

        # Add wind direction arrows (downsampled)
        skip = max(1, wind_field.shape[0] // 15)
        x = np.arange(0, wind_field.shape[0], skip)
        y = np.arange(0, wind_field.shape[1], skip)
        X, Y = np.meshgrid(x, y)
        U = wind_field[::skip, ::skip, 0].T
        V = wind_field[::skip, ::skip, 1].T
        axes[1, 2].quiver(X, Y, U, V, color="white", alpha=0.7, scale=10)

        axes[1, 2].set_title("Wind Field")
        axes[1, 2].set_xlabel("X")
        axes[1, 2].set_ylabel("Y")
        plt.colorbar(im6, ax=axes[1, 2], label="Wind Magnitude")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Spatial analysis plot saved to {save_path}")
        # plt.show()

    def plot_burn_rate_analysis(self, save_path=None):
        """Analyze burn rates and fire behavior"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Burn Rate Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Burn rate over time
        axes[0, 0].plot(
            self.df["frame"], self.df["burn_rate"], color="red", linewidth=2
        )
        axes[0, 0].set_xlabel("Time (frames)")
        axes[0, 0].set_ylabel("Cells Burned per Frame")
        axes[0, 0].set_title("Instantaneous Burn Rate")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Burn rate histogram
        axes[0, 1].hist(
            self.df["burn_rate"], bins=30, color="orange", alpha=0.7, edgecolor="black"
        )
        axes[0, 1].set_xlabel("Burn Rate (cells/frame)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Burn Rate Distribution")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Cumulative burn rate vs particle count
        axes[1, 0].scatter(
            self.df["active_particles"],
            self.df["burn_rate"],
            alpha=0.6,
            c=self.df["frame"],
            cmap="plasma",
        )
        axes[1, 0].set_xlabel("Active Particles")
        axes[1, 0].set_ylabel("Burn Rate")
        axes[1, 0].set_title("Particles vs Burn Rate")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Moving average burn rate
        window = min(10, len(self.df) // 5)
        if window > 1:
            moving_avg = self.df["burn_rate"].rolling(window=window).mean()
            axes[1, 1].plot(
                self.df["frame"],
                self.df["burn_rate"],
                alpha=0.3,
                color="red",
                label="Raw",
            )
            axes[1, 1].plot(
                self.df["frame"],
                moving_avg,
                color="darkred",
                linewidth=2,
                label=f"{window}-frame Moving Avg",
            )
            axes[1, 1].set_xlabel("Time (frames)")
            axes[1, 1].set_ylabel("Burn Rate")
            axes[1, 1].set_title("Smoothed Burn Rate")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Burn rate analysis plot saved to {save_path}")
        # plt.show()

    def plot_environmental_effects(self, save_path=None):
        """Analyze environmental factor effects"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Environmental Effects Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Wind effect on fire spread
        axes[0, 0].scatter(
            self.df["wind_effect_strength"],
            self.df["fire_spread_distance"],
            alpha=0.6,
            c=self.df["frame"],
            cmap="viridis",
        )
        axes[0, 0].set_xlabel("Wind Strength")
        axes[0, 0].set_ylabel("Fire Spread Distance")
        axes[0, 0].set_title("Wind Effect on Fire Spread")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Fuel type effectiveness
        fuel_types = self.spatial_data["fuel_types"]
        final_grid = self.spatial_data["final_grid"]

        # Calculate burn probability by fuel type
        fuel_values = np.unique(fuel_types)
        burn_probabilities = []
        fuel_labels = []

        for fuel_val in fuel_values:
            fuel_mask = fuel_types == fuel_val
            total_fuel = np.sum(fuel_mask)
            if total_fuel > 0:
                burned_fuel = np.sum(
                    (final_grid == 2) & fuel_mask
                )  # CellState.BURNED = 2
                burn_prob = burned_fuel / total_fuel
                burn_probabilities.append(burn_prob)
                fuel_labels.append(f"Fuel {fuel_val:.1f}")

        axes[0, 1].bar(fuel_labels, burn_probabilities, color="orange", alpha=0.7)
        axes[0, 1].set_xlabel("Fuel Type")
        axes[0, 1].set_ylabel("Burn Probability")
        axes[0, 1].set_title("Fuel Type Burn Effectiveness")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Moisture vs burn probability
        moisture_map = self.spatial_data["moisture_map"]

        # Create moisture bins and calculate burn probability
        moisture_bins = np.linspace(0, 1, 10)
        moisture_burn_probs = []
        bin_centers = []

        for i in range(len(moisture_bins) - 1):
            moisture_mask = (moisture_map >= moisture_bins[i]) & (
                moisture_map < moisture_bins[i + 1]
            )
            total_cells = np.sum(moisture_mask)
            if total_cells > 0:
                burned_cells = np.sum((final_grid == 2) & moisture_mask)
                burn_prob = burned_cells / total_cells
                moisture_burn_probs.append(burn_prob)
                bin_centers.append((moisture_bins[i] + moisture_bins[i + 1]) / 2)

        axes[1, 0].plot(
            bin_centers,
            moisture_burn_probs,
            "o-",
            color="blue",
            linewidth=2,
            markersize=6,
        )
        axes[1, 0].set_xlabel("Moisture Level")
        axes[1, 0].set_ylabel("Burn Probability")
        axes[1, 0].set_title("Moisture Effect on Burning")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Parameter timeline
        params = self.metadata["parameters"]
        key_params = [
            "spread_rate",
            "ignition_probability",
            "wind_strength",
            "base_moisture",
        ]
        param_values = [params.get(p, 0) for p in key_params]

        bars = axes[1, 1].bar(
            key_params,
            param_values,
            color=["red", "orange", "blue", "green"],
            alpha=0.7,
        )
        axes[1, 1].set_ylabel("Parameter Value")
        axes[1, 1].set_title("Key Simulation Parameters")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, param_values):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Environmental effects plot saved to {save_path}")
        # plt.show()

    def create_summary_table(self, save_path=None):
        """Create a summary statistics table"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("tight")
        ax.axis("off")

        # Prepare summary data
        summary_data = []
        for key, value in self.summary_stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            summary_data.append([key.replace("_", " ").title(), formatted_value])

        # Add parameter information
        params = self.metadata["parameters"]
        key_params = [
            "map_type",
            "spread_rate",
            "ignition_probability",
            "wind_strength",
            "base_moisture",
            "particle_lifetime",
            "burnout_rate",
        ]

        summary_data.append(["", ""])  # Empty row
        summary_data.append(["SIMULATION PARAMETERS", ""])

        for param in key_params:
            if param in params:
                value = params[param]
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                summary_data.append([param.replace("_", " ").title(), formatted_value])

        # Create table
        table = ax.table(
            cellText=summary_data,
            colLabels=["Metric", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.7, 0.3],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor("#4CAF50")
                    cell.set_text_props(weight="bold", color="white")
                elif i > 0 and summary_data[i - 1][0] == "SIMULATION PARAMETERS":
                    cell.set_facecolor("#2196F3")
                    cell.set_text_props(weight="bold", color="white")
                elif i > 0 and summary_data[i - 1][0] == "":
                    cell.set_facecolor("white")
                else:
                    cell.set_facecolor("#f1f1f1" if i % 2 == 0 else "white")

        plt.title(
            "Simulation Summary Statistics", fontsize=16, fontweight="bold", pad=20
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Summary table saved to {save_path}")
        # plt.show()

    def plot_parameter_correlations(self, save_path=None):
        """Analyze correlations between parameters and outcomes"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Parameter Correlation Analysis", fontsize=16, fontweight="bold")

        # Plot 1: Note about correlation analysis
        axes[0, 0].text(
            0.5,
            0.5,
            "Correlation analysis requires\nmultiple simulations\nfor meaningful results",
            ha="center",
            va="center",
            fontsize=12,
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_title("Parameter Correlation Matrix")
        axes[0, 0].axis("off")

        # Plot 2: Time-based correlations
        if len(self.df) > 10:
            time_corr = self.df[
                [
                    "active_particles",
                    "burning_cells",
                    "burn_rate",
                    "fire_spread_distance",
                ]
            ].corr()
            im = axes[0, 1].imshow(time_corr, cmap="RdBu", vmin=-1, vmax=1)
            axes[0, 1].set_xticks(range(len(time_corr.columns)))
            axes[0, 1].set_yticks(range(len(time_corr.columns)))
            axes[0, 1].set_xticklabels(time_corr.columns, rotation=45, ha="right")
            axes[0, 1].set_yticklabels(time_corr.columns)

            # Add correlation values to cells
            for i in range(len(time_corr.columns)):
                for j in range(len(time_corr.columns)):
                    text = axes[0, 1].text(
                        j,
                        i,
                        f"{time_corr.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if abs(time_corr.iloc[i, j]) < 0.5 else "white",
                    )

            plt.colorbar(im, ax=axes[0, 1])
            axes[0, 1].set_title("Time Series Correlations")

        # Plot 3: Parameter vs outcome scatter
        outcomes = [
            self.summary_stats["final_burned_percentage"],
            self.summary_stats["max_fire_spread_distance"],
        ]
        param_names = ["Final Burned %", "Max Spread Distance"]
        colors = ["red", "orange"]

        for i, (outcome, name, color) in enumerate(zip(outcomes, param_names, colors)):
            axes[1, 0].bar(name, outcome, color=color, alpha=0.7)

        axes[1, 0].set_ylabel("Value")
        axes[1, 0].set_title("Key Outcomes")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Plot 4: Parameter sensitivity (relative to typical values)
        param_values = [
            self.metadata["parameters"]["spread_rate"]
            / 0.01,  # Normalize to typical value
            self.metadata["parameters"]["ignition_probability"] / 0.03,
            self.metadata["parameters"]["wind_strength"] / 0.5,
            self.metadata["parameters"]["base_moisture"] / 0.2,
        ]
        param_labels = [
            "Spread Rate\n(rel to 0.01)",
            "Ignition Prob\n(rel to 0.03)",
            "Wind Strength\n(rel to 0.5)",
            "Base Moisture\n(rel to 0.2)",
        ]

        bars = axes[1, 1].bar(
            param_labels,
            param_values,
            color=["red", "orange", "blue", "green"],
            alpha=0.7,
        )
        axes[1, 1].axhline(
            y=1, color="black", linestyle="--", alpha=0.7, label="Baseline"
        )
        axes[1, 1].set_ylabel("Relative to Baseline")
        axes[1, 1].set_title("Parameter Values (Normalized)")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Parameter correlation plot saved to {save_path}")
        # plt.show()

    def plot_fire_spread_patterns(self, save_path=None):
        """Analyze fire spread patterns and progression - UPDATED"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Fire Spread Pattern Analysis", fontsize=16, fontweight="bold")

        # Get burned area progression snapshots
        progression_data = self.spatial_data.get("burned_area_progression", [])

        if len(progression_data) >= 3:
            # Show progression at different time points
            time_points = [0, len(progression_data) // 2, -1]
            titles = ["Early Stage", "Mid Stage", "Final Stage"]

            for i, (time_idx, title) in enumerate(zip(time_points, titles)):
                snapshot = progression_data[time_idx]
                
                # UPDATED: Show both burned and currently burning areas
                burned_area = snapshot["burned_area"]
                burning_area = snapshot.get("burning_area", np.zeros_like(burned_area))
                
                # Create combined visualization
                combined_display = burned_area.astype(float)
                combined_display[burning_area == 1] = 0.5  # Burning areas in middle value
                
                im = axes[0, i].imshow(
                    combined_display.T,
                    origin="lower",
                    cmap="Reds",
                    vmin=0,
                    vmax=1,
                )
                axes[0, i].set_title(f'{title}\n(Frame {snapshot["frame"]})')
                axes[0, i].set_xlabel("X")
                axes[0, i].set_ylabel("Y")

                # Add ignition points
                for ignition_point in self.spatial_data["ignition_points"]:
                    axes[0, i].plot(
                        ignition_point[0], ignition_point[1], "go", markersize=8
                    )
                    
                # Add colorbar for first plot
                if i == 0:
                    plt.colorbar(im, ax=axes[0, i], label="Burned (1) / Burning (0.5) / Unburned (0)")
        else:
            # Fallback visualization
            axes[0, 0].imshow(
                self.spatial_data["initial_grid"].T, origin="lower", cmap="RdYlGn_r"
            )
            axes[0, 0].set_title("Initial State")

            axes[0, 1].imshow(
                self.spatial_data["final_grid"].T, origin="lower", cmap="RdYlGn_r"
            )
            axes[0, 1].set_title("Final State")

            axes[0, 2].axis("off")

        # UPDATED: Enhanced spread analysis with new metrics
        if len(self.df) > 1:
            # Plot 1: Cumulative fire spread
            axes[1, 0].plot(
                self.df["frame"],
                np.cumsum(self.df["burn_rate"]),
                color="red",
                linewidth=2,
                label="Cumulative Burned"
            )
            
            # Add fire front cells if available
            if "fire_front_cells" in self.df.columns:
                ax2 = axes[1, 0].twinx()
                ax2.plot(
                    self.df["frame"],
                    self.df["fire_front_cells"],
                    color="orange",
                    linewidth=2,
                    alpha=0.7,
                    label="Fire Front Cells"
                )
                ax2.set_ylabel("Fire Front Cells", color="orange")
                
            axes[1, 0].set_xlabel("Time (frames)")
            axes[1, 0].set_ylabel("Cumulative Cells Burned", color="red")
            axes[1, 0].set_title("Cumulative Fire Spread")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend(loc='upper left')

            # Plot 2: Fire front velocity
            spread_velocity = np.diff(self.df["fire_spread_distance"])
            # Handle case where spread_velocity might be all zeros
            if np.any(spread_velocity != 0):
                axes[1, 1].plot(
                    self.df["frame"][1:], spread_velocity, color="orange", linewidth=2
                )
                axes[1, 1].set_xlabel("Time (frames)")
                axes[1, 1].set_ylabel("Spread Velocity (units/frame)")
                axes[1, 1].set_title("Fire Front Velocity")
                axes[1, 1].grid(True, alpha=0.3)
            else:
                # Alternative plot if no velocity data
                axes[1, 1].plot(
                    self.df["frame"], self.df["burn_rate"], color="orange", linewidth=2
                )
                axes[1, 1].set_xlabel("Time (frames)")
                axes[1, 1].set_ylabel("Burn Rate (cells/frame)")
                axes[1, 1].set_title("Instantaneous Burn Rate")
                axes[1, 1].grid(True, alpha=0.3)

            # Plot 3: UPDATED shape analysis with better error handling
            if len(progression_data) >= 3:
                try:
                    from scipy import ndimage
                    
                    perimeters = []
                    areas = []
                    frames = []
                    
                    for snapshot in progression_data:
                        burned_area = snapshot["burned_area"]
                        area = np.sum(burned_area)
                        
                        if area > 5:  # Only analyze if significant area burned
                            # Calculate perimeter using edge detection
                            edges = ndimage.sobel(burned_area.astype(float))
                            perimeter = np.sum(edges > 0.1)
                            perimeters.append(perimeter)
                            areas.append(area)
                            frames.append(snapshot["frame"])

                    if len(areas) > 3:
                        # Plot fire shape evolution over time
                        scatter = axes[1, 2].scatter(areas, perimeters, c=frames, cmap="viridis", alpha=0.7)
                        axes[1, 2].set_xlabel("Burned Area (cells)")
                        axes[1, 2].set_ylabel("Fire Perimeter (cells)")
                        axes[1, 2].set_title("Fire Shape Evolution")
                        axes[1, 2].grid(True, alpha=0.3)
                        
                        # Add colorbar for time
                        plt.colorbar(scatter, ax=axes[1, 2], label="Time (frame)")
                    else:
                        axes[1, 2].text(
                            0.5, 0.5, "Insufficient burned area\nfor shape analysis",
                            ha="center", va="center", transform=axes[1, 2].transAxes
                        )
                        axes[1, 2].set_title("Fire Shape Evolution")

                except (ImportError, Exception) as e:
                    axes[1, 2].text(
                        0.5, 0.5, f"Shape analysis unavailable:\n{str(e)[:50]}...",
                        ha="center", va="center", transform=axes[1, 2].transAxes
                    )
                    axes[1, 2].set_title("Fire Shape Analysis")
            else:
                axes[1, 2].text(
                    0.5, 0.5, "No progression data\navailable for analysis",
                    ha="center", va="center", transform=axes[1, 2].transAxes
                )
                axes[1, 2].set_title("Fire Shape Analysis")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Fire spread patterns plot saved to {save_path}")
        plt.show()


    def quick_summary(self):
        """Print a quick text summary of key results - FIXED pandas Series issue"""
        print("\n" + "=" * 60)
        print("FOREST FIRE SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Map Type: {self.metadata['map_type']}")
        print(f"Grid Size: {self.metadata['grid_size']}")
        print(f"Simulation Duration: {len(self.df)} frames")
        print(f"Total Burned Area: {self.summary_stats.get('final_burned_percentage', 0):.1f}%")
        print(f"Max Fire Spread: {self.summary_stats.get('max_fire_spread_distance', 0):.1f} units")

        # Enhanced statistics with new metrics
        peak_burn_rate = self.summary_stats.get(
            "peak_burning_cells",
            max(self.df["burn_rate"]) if "burn_rate" in self.df.columns and len(self.df["burn_rate"]) > 0 else 0,
        )
        peak_particles = self.summary_stats.get(
            "max_active_particles",
            max(self.df["active_particles"]) if "active_particles" in self.df.columns and len(self.df["active_particles"]) > 0 else 0,
        )
        
        print(f"Peak Burn Rate: {peak_burn_rate:.1f} cells/frame")
        print(f"Peak Active Particles: {peak_particles}")
        
        # FIXED: Particle velocity statistics with proper pandas Series handling
        if "particle_velocity_avg" in self.df.columns and not self.df["particle_velocity_avg"].empty:
            # Check if there are any non-zero values
            non_zero_velocities = self.df["particle_velocity_avg"][self.df["particle_velocity_avg"] > 0]
            if not non_zero_velocities.empty:
                avg_velocity = non_zero_velocities.mean()
                max_velocity = self.df["particle_velocity_max"].max()
                print(f"Average Particle Velocity: {avg_velocity:.3f}")
                print(f"Maximum Particle Velocity: {max_velocity:.3f}")
        
        # FIXED: Wind effect statistics with proper pandas Series handling
        if "wind_effect_std" in self.df.columns and not self.df["wind_effect_std"].empty:
            avg_wind_variation = self.df["wind_effect_std"].mean()
            print(f"Average Wind Variation: {avg_wind_variation:.3f}")
        
        # FIXED: Fire front statistics with proper pandas Series handling
        if "fire_front_cells" in self.df.columns and not self.df["fire_front_cells"].empty:
            # Check if there are any non-zero values
            non_zero_front = self.df["fire_front_cells"][self.df["fire_front_cells"] > 0]
            if not non_zero_front.empty:
                max_fire_front = self.df["fire_front_cells"].max()
                avg_fire_front = non_zero_front.mean()
                print(f"Peak Fire Front Size: {max_fire_front} cells")
                print(f"Average Fire Front Size: {avg_fire_front:.1f} cells")
        
        # Existing statistics with fixes
        avg_intensity = self.summary_stats.get(
            "fire_intensity_average",
            0  # Default fallback
        )
        
        # If not in summary stats, calculate from DataFrame
        if avg_intensity == 0 and "particle_intensity_avg" in self.df.columns:
            non_zero_intensity = self.df["particle_intensity_avg"][self.df["particle_intensity_avg"] > 0]
            if not non_zero_intensity.empty:
                avg_intensity = non_zero_intensity.mean()
        
        print(f"Average Particle Intensity: {avg_intensity:.3f}")
        print(f"Fire Duration: {self.summary_stats.get('fire_duration', len(self.df))} frames")
        print(f"Total Area Burned: {self.summary_stats.get('total_area_burned', 0)} cells")
        
        # Multi-ignition point summary for coastal simulations
        ignition_points = self.spatial_data.get("ignition_points", [])
        if len(ignition_points) > 1:
            print(f"Ignition Points: {len(ignition_points)} locations")
            print(f"Ignition Strategy: Multi-point ignition")
        
        print("=" * 60)
    
def compare_simulations(file_list, output_dir="comparison_output"):
    """Compare multiple simulation files"""
    analyzers = []
    labels = []

    for file_path in file_list:
        analyzer = SimulationAnalyzer(file_path)
        analyzers.append(analyzer)

        # Extract meaningful label from filename
        filename = os.path.basename(file_path)
        label = filename.replace("fire_simulation_", "").replace(".pkl", "")
        labels.append(label)

    os.makedirs(output_dir, exist_ok=True)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Simulation Comparison", fontsize=16, fontweight="bold")

    colors = sns.color_palette("husl", len(analyzers))

    # Plot 1: Burned area over time
    for analyzer, label, color in zip(analyzers, labels, colors):
        total_cells = (
            analyzer.metadata["grid_size"][0] * analyzer.metadata["grid_size"][1]
        )
        burned_percentage = (analyzer.df["burned_cells"] / total_cells) * 100
        axes[0, 0].plot(
            analyzer.df["frame"],
            burned_percentage,
            label=label,
            color=color,
            linewidth=2,
        )

    axes[0, 0].set_xlabel("Time (frames)")
    axes[0, 0].set_ylabel("Burned Area (%)")
    axes[0, 0].set_title("Burned Area Progression Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Burn rate comparison
    for analyzer, label, color in zip(analyzers, labels, colors):
        axes[0, 1].plot(
            analyzer.df["frame"],
            analyzer.df["burn_rate"],
            label=label,
            color=color,
            linewidth=2,
        )

    axes[0, 1].set_xlabel("Time (frames)")
    axes[0, 1].set_ylabel("Burn Rate (cells/frame)")
    axes[0, 1].set_title("Burn Rate Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Fire spread distance comparison
    for analyzer, label, color in zip(analyzers, labels, colors):
        axes[1, 0].plot(
            analyzer.df["frame"],
            analyzer.df["fire_spread_distance"],
            label=label,
            color=color,
            linewidth=2,
        )

    axes[1, 0].set_xlabel("Time (frames)")
    axes[1, 0].set_ylabel("Fire Spread Distance")
    axes[1, 0].set_title("Fire Spread Distance Comparison")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Final summary comparison
    final_burned = [a.summary_stats["final_burned_percentage"] for a in analyzers]
    max_spread = [a.summary_stats["max_fire_spread_distance"] for a in analyzers]

    x = np.arange(len(labels))
    width = 0.35

    axes[1, 1].bar(
        x - width / 2, final_burned, width, label="Final Burned %", alpha=0.7
    )
    ax2 = axes[1, 1].twinx()
    ax2.bar(
        x + width / 2,
        max_spread,
        width,
        label="Max Spread Distance",
        alpha=0.7,
        color="orange",
    )

    axes[1, 1].set_xlabel("Simulation")
    axes[1, 1].set_ylabel("Burned Area (%)")
    ax2.set_ylabel("Max Spread Distance")
    axes[1, 1].set_title("Final Results Comparison")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45)
    axes[1, 1].legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    save_path = f"{output_dir}/comparison_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {save_path}")
    # plt.show()

    # Create comparison summary table
    comparison_data = []
    for analyzer, label in zip(analyzers, labels):
        row = [
            label,
            f"{analyzer.summary_stats['final_burned_percentage']:.1f}%",
            f"{analyzer.summary_stats['max_fire_spread_distance']:.1f}",
            f"{analyzer.summary_stats.get('max_burn_rate', 0):.1f}",
            f"{analyzer.summary_stats.get('simulation_duration', len(analyzer.df))}",
            f"{analyzer.summary_stats.get('max_particles', analyzer.summary_stats.get('max_active_particles', 0))}",
        ]
        comparison_data.append(row)

    # Save comparison table
    comparison_df = pd.DataFrame(
        comparison_data,
        columns=[
            "Simulation",
            "Final Burned %",
            "Max Spread",
            "Max Burn Rate",
            "Duration",
            "Max Particles",
        ],
    )
    comparison_df.to_csv(f"{output_dir}/comparison_table.csv", index=False)
    print(f"Comparison table saved to {output_dir}/comparison_table.csv")

    return analyzers


def main():
    """Main script entry point"""
    parser = argparse.ArgumentParser(description="Analyze forest fire simulation data")
    parser.add_argument("files", nargs="*", help="Simulation data files (.pkl)")
    parser.add_argument(
        "--directory", "-d", help="Directory containing simulation files"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare multiple simulations"
    )
    parser.add_argument(
        "--output", "-o", default="analysis_output", help="Output directory"
    )
    parser.add_argument("--quick", "-q", action="store_true", help="Quick summary only")

    args = parser.parse_args()

    # Determine which files to analyze
    files_to_analyze = []

    if args.directory:
        # Analyze all pickle files in directory
        pattern = os.path.join(args.directory, "*.pkl")
        files_to_analyze = glob.glob(pattern)
        print(f"Found {len(files_to_analyze)} simulation files in {args.directory}")
    elif args.files:
        files_to_analyze = args.files
    else:
        print("Please provide either files or a directory containing simulation data.")
        return

    if not files_to_analyze:
        print("No simulation files found!")
        return

    # Process files
    if len(files_to_analyze) == 1:
        # Single file analysis
        analyzer = SimulationAnalyzer(files_to_analyze[0])

        if args.quick:
            analyzer.quick_summary()
        else:
            analyzer.quick_summary()
            output_dir = analyzer.create_comprehensive_report(args.output)
            print(f"\nComplete analysis saved to: {output_dir}/")

    elif args.compare or len(files_to_analyze) > 1:
        # Multiple file comparison
        print(f"Comparing {len(files_to_analyze)} simulations...")
        analyzers = compare_simulations(files_to_analyze, args.output)

        # Also show individual quick summaries
        for analyzer, filename in zip(analyzers, files_to_analyze):
            print(f"\n--- {os.path.basename(filename)} ---")
            analyzer.quick_summary()


if __name__ == "__main__":
    main()
