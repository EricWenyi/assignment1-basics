#!/usr/bin/env python3
"""
Experiment Tracking Infrastructure for Transformer Training

This module provides comprehensive experiment tracking capabilities including:
- Automatic experiment organization and naming
- Real-time metrics logging (loss, learning rate, etc.)
- Hyperparameter tracking and configuration saving
- Loss curve visualization with both step-based and time-based views
- Experiment comparison and analysis tools
- JSON-based experiment log for reproducibility

Key features:
- Tracks both training and validation metrics over time
- Records wallclock time for performance analysis
- Saves hyperparameters and model configurations
- Generates publication-ready plots
- Maintains experiment history for comparison
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

import matplotlib.pyplot as plt
import numpy as np


class ExperimentTracker:
    """
    Comprehensive experiment tracking system for machine learning experiments.
    
    This class handles:
    1. Experiment metadata and configuration storage
    2. Real-time metrics logging during training
    3. Automatic experiment naming and organization
    4. Loss curve generation and visualization
    5. Experiment comparison and analysis
    
    Each experiment gets a unique ID and organized directory structure.
    """
    
    def __init__(
        self, 
        experiment_name: str,
        base_dir: str = "experiments",
        description: str = "",
        tags: List[str] = None,
        auto_save_interval: int = 100
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Human-readable name for the experiment
            base_dir: Root directory for all experiments
            description: Detailed description of experiment purpose
            tags: List of tags for categorizing experiments
            auto_save_interval: Save metrics every N steps (0 = manual save only)
        """
        # Generate unique experiment ID and timestamp
        self.experiment_id = str(uuid.uuid4())[:8]
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags or []
        
        # Create experiment directory structure
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / f"{self.timestamp}_{self.experiment_name}_{self.experiment_id}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "plots").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize tracking variables
        self.start_time = time.time()
        self.metrics = {
            'train': {},
            'val': {},
            'final': {},  # Add final split for end-of-training metrics
            'step_times': [],
            'wall_times': []
        }
        self.current_step = 0
        self.auto_save_interval = auto_save_interval
        self.hyperparameters = {}
        self.model_config = {}
        
        print(f"🚀 Started experiment: {experiment_name}")
        print(f"📁 Experiment ID: {self.experiment_id}")
        print(f"📂 Directory: {self.exp_dir}")
        print(f"🏷️  Tags: {self.tags}")
        if description:
            print(f"📝 Description: {description}")
        print()
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log experiment hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameter names and values
        """
        self.hyperparameters.update(hyperparams)
        
        # Save hyperparameters to file
        hyperparam_file = self.exp_dir / "hyperparameters.json"
        with open(hyperparam_file, 'w') as f:
            json.dump(self.hyperparameters, f, indent=2, default=str)
        
        print("💾 Logged hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        print()
    
    def log_model_config(self, model_config: Dict[str, Any]) -> None:
        """
        Log model architecture configuration.
        
        Args:
            model_config: Dictionary of model configuration parameters
        """
        self.model_config.update(model_config)
        
        # Save model config to file
        config_file = self.exp_dir / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.model_config, f, indent=2, default=str)
        
        print("🏗️  Logged model configuration:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        print()
    
    def log_metrics(
        self, 
        step: int, 
        metrics: Dict[str, float], 
        split: str = 'train',
        force_save: bool = False
    ) -> None:
        """
        Log metrics for current training step.
        
        Args:
            step: Current training step/iteration
            metrics: Dictionary of metric names and values
            split: Data split ('train' or 'val')
            force_save: Force immediate save regardless of auto_save_interval
        """
        self.current_step = step
        current_time = time.time()
        wall_time = current_time - self.start_time
        
        # Initialize metric lists if first time
        for metric_name in metrics.keys():
            if metric_name not in self.metrics[split]:
                self.metrics[split][metric_name] = []
        
        # Store metrics with step and time information
        for metric_name, metric_value in metrics.items():
            self.metrics[split][metric_name].append({
                'step': step,
                'value': metric_value,
                'wall_time': wall_time,
                'timestamp': datetime.now().isoformat()
            })
        
        # Track step timing for throughput analysis
        self.metrics['step_times'].append(current_time)
        self.metrics['wall_times'].append(wall_time)
        
        # Print metrics
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        time_str = f"{wall_time:.1f}s"
        print(f"📊 Step {step:6d} [{split:5s}] {time_str:>8s} | {metric_str}")
        
        # Auto-save if interval reached
        if force_save or (self.auto_save_interval > 0 and step % self.auto_save_interval == 0):
            self.save_metrics()
    
    def save_metrics(self) -> None:
        """Save all metrics to JSON file."""
        metrics_file = self.exp_dir / "metrics.json"
        
        # Prepare data for JSON serialization
        save_data = {
            'experiment_info': {
                'experiment_id': self.experiment_id,
                'experiment_name': self.experiment_name,
                'description': self.description,
                'tags': self.tags,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'current_step': self.current_step
            },
            'metrics': self.metrics,
            'hyperparameters': self.hyperparameters,
            'model_config': self.model_config
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
    
    def plot_loss_curves(
        self, 
        save_plot: bool = True, 
        show_plot: bool = False,
        metrics_to_plot: List[str] = None
    ) -> None:
        """
        Generate comprehensive loss curve visualizations.
        
        Args:
            save_plot: Whether to save plots to disk
            show_plot: Whether to display plots interactively
            metrics_to_plot: Specific metrics to plot (default: all available)
        """
        if metrics_to_plot is None:
            # Plot all available metrics
            all_metrics = set()
            for split_data in self.metrics.values():
                if isinstance(split_data, dict):
                    all_metrics.update(split_data.keys())
            metrics_to_plot = list(all_metrics - {'step_times', 'wall_times'})
        
        if not metrics_to_plot:
            print("⚠️  No metrics to plot")
            return
        
        # Create subplot grid
        n_metrics = len(metrics_to_plot)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = {'train': '#1f77b4', 'val': '#ff7f0e'}
        
        for i, metric_name in enumerate(metrics_to_plot):
            ax = axes[i] if len(axes) > 1 else axes[0]
            
            # Plot each split (train/val)
            for split in ['train', 'val']:
                if metric_name in self.metrics[split] and self.metrics[split][metric_name]:
                    data = self.metrics[split][metric_name]
                    steps = [d['step'] for d in data]
                    values = [d['value'] for d in data]
                    
                    ax.plot(steps, values, label=f'{split}', color=colors[split], linewidth=2)
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} vs Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Use log scale for loss if values span multiple orders of magnitude
            if 'loss' in metric_name.lower():
                if len(values) > 1:
                    val_range = max(values) / min(values) if min(values) > 0 else 1
                    if val_range > 100:  # Use log scale if range > 100x
                        ax.set_yscale('log')
        
        # Hide empty subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.exp_dir / "plots" / f"loss_curves_step_{self.current_step}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved loss curves: {plot_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_time_based_curves(
        self, 
        save_plot: bool = True, 
        show_plot: bool = False,
        metrics_to_plot: List[str] = None
    ) -> None:
        """
        Generate time-based (wallclock) loss curve visualizations.
        
        Useful for comparing training efficiency and convergence speed.
        """
        if metrics_to_plot is None:
            all_metrics = set()
            for split_data in self.metrics.values():
                if isinstance(split_data, dict):
                    all_metrics.update(split_data.keys())
            metrics_to_plot = list(all_metrics - {'step_times', 'wall_times'})
        
        if not metrics_to_plot:
            print("⚠️  No metrics to plot")
            return
        
        # Create subplot grid
        n_metrics = len(metrics_to_plot)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        colors = {'train': '#1f77b4', 'val': '#ff7f0e'}
        
        for i, metric_name in enumerate(metrics_to_plot):
            ax = axes[i] if len(axes) > 1 else axes[0]
            
            # Plot each split (train/val) vs wall time
            for split in ['train', 'val']:
                if metric_name in self.metrics[split] and self.metrics[split][metric_name]:
                    data = self.metrics[split][metric_name]
                    wall_times = [d['wall_time'] / 60 for d in data]  # Convert to minutes
                    values = [d['value'] for d in data]
                    
                    ax.plot(wall_times, values, label=f'{split}', color=colors[split], linewidth=2)
            
            ax.set_xlabel('Wall Time (minutes)')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} vs Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Use log scale for loss if appropriate
            if 'loss' in metric_name.lower():
                if len(values) > 1:
                    val_range = max(values) / min(values) if min(values) > 0 else 1
                    if val_range > 100:
                        ax.set_yscale('log')
        
        # Hide empty subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_file = self.exp_dir / "plots" / f"time_curves_step_{self.current_step}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"💾 Saved time-based curves: {plot_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Generate experiment summary with key statistics.
        
        Returns:
            Dictionary containing experiment summary information
        """
        current_time = time.time()
        duration = current_time - self.start_time
        
        summary = {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'description': self.description,
            'tags': self.tags,
            'duration_seconds': duration,
            'duration_formatted': f"{duration/3600:.1f}h" if duration > 3600 else f"{duration/60:.1f}m",
            'current_step': self.current_step,
            'directory': str(self.exp_dir)
        }
        
        # Add final metric values
        for split in ['train', 'val']:
            summary[f'{split}_metrics'] = {}
            for metric_name, metric_data in self.metrics[split].items():
                if metric_data:
                    final_value = metric_data[-1]['value']
                    summary[f'{split}_metrics'][metric_name] = final_value
        
        # Calculate training speed
        if self.current_step > 0 and duration > 0:
            summary['steps_per_second'] = self.current_step / duration
        
        return summary
    
    def print_summary(self) -> None:
        """Print experiment summary to console."""
        summary = self.get_summary()
        
        print("=" * 60)
        print(f"📋 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Name: {summary['experiment_name']}")
        print(f"ID: {summary['experiment_id']}")
        if summary['description']:
            print(f"Description: {summary['description']}")
        if summary['tags']:
            print(f"Tags: {', '.join(summary['tags'])}")
        print(f"Duration: {summary['duration_formatted']}")
        print(f"Steps: {summary['current_step']:,}")
        if 'steps_per_second' in summary:
            print(f"Speed: {summary['steps_per_second']:.1f} steps/sec")
        
        # Print final metrics
        for split in ['train', 'val']:
            split_metrics = summary[f'{split}_metrics']
            if split_metrics:
                print(f"\n{split.capitalize()} Metrics:")
                for metric_name, value in split_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
        
        print(f"\nDirectory: {summary['directory']}")
        print("=" * 60)
    
    def finalize(self, create_plots: bool = True) -> None:
        """
        Finalize experiment by saving all data and generating final plots.
        
        Args:
            create_plots: Whether to generate final visualization plots
        """
        print(f"🏁 Finalizing experiment: {self.experiment_name}")
        
        # Save final metrics
        self.save_metrics()
        
        # Generate final plots
        if create_plots:
            self.plot_loss_curves(save_plot=True, show_plot=False)
            self.plot_time_based_curves(save_plot=True, show_plot=False)
        
        # Print final summary
        self.print_summary()
        
        print(f"✅ Experiment finalized in: {self.exp_dir}")


def load_experiment(experiment_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a previously saved experiment for analysis.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary containing all experiment data
    """
    experiment_dir = Path(experiment_dir)
    metrics_file = experiment_dir / "metrics.json"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"No metrics.json found in {experiment_dir}")
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    return data


def compare_experiments(experiment_dirs: List[Union[str, Path]], metric_name: str = 'loss') -> None:
    """
    Compare multiple experiments by plotting their loss curves together.
    
    Args:
        experiment_dirs: List of experiment directory paths
        metric_name: Metric to compare across experiments
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_dirs)))
    
    for i, exp_dir in enumerate(experiment_dirs):
        try:
            data = load_experiment(exp_dir)
            exp_name = data['experiment_info']['experiment_name']
            
            # Plot training curve
            if metric_name in data['metrics']['train']:
                train_data = data['metrics']['train'][metric_name]
                steps = [d['step'] for d in train_data]
                values = [d['value'] for d in train_data]
                plt.plot(steps, values, color=colors[i], linewidth=2, label=f'{exp_name} (train)')
            
            # Plot validation curve if available
            if metric_name in data['metrics']['val']:
                val_data = data['metrics']['val'][metric_name]
                steps = [d['step'] for d in val_data]
                values = [d['value'] for d in val_data]
                plt.plot(steps, values, color=colors[i], linewidth=2, linestyle='--', label=f'{exp_name} (val)')
                
        except Exception as e:
            print(f"⚠️  Could not load experiment {exp_dir}: {e}")
    
    plt.xlabel('Training Steps')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use log scale for loss if appropriate
    if 'loss' in metric_name.lower():
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Experiment Tracker")
    
    # Create test experiment
    tracker = ExperimentTracker(
        experiment_name="test_transformer_training",
        description="Testing experiment tracking infrastructure",
        tags=["test", "infrastructure", "transformer"]
    )
    
    # Log some hyperparameters
    tracker.log_hyperparameters({
        'learning_rate': 0.0001,
        'batch_size': 32,
        'model_dim': 512,
        'num_layers': 6
    })
    
    # Log model configuration
    tracker.log_model_config({
        'vocab_size': 50000,
        'context_length': 512,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6
    })
    
    # Simulate training loop with metrics
    import random
    random.seed(42)
    
    for step in range(1, 101):
        # Simulate decreasing loss with noise
        train_loss = 5.0 * np.exp(-step / 50) + 0.1 * random.random()
        train_ppl = np.exp(train_loss)
        
        tracker.log_metrics(
            step=step,
            metrics={'loss': train_loss, 'perplexity': train_ppl},
            split='train'
        )
        
        # Log validation metrics every 10 steps
        if step % 10 == 0:
            val_loss = train_loss + 0.2 + 0.05 * random.random()
            val_ppl = np.exp(val_loss)
            
            tracker.log_metrics(
                step=step,
                metrics={'loss': val_loss, 'perplexity': val_ppl},
                split='val'
            )
    
    # Finalize experiment
    tracker.finalize()
    
    print("✅ Test completed successfully!")