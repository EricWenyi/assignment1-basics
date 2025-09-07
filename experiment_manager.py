#!/usr/bin/env python3
"""
Experiment Management Utilities

This script provides utilities for managing and analyzing experiments:
- List all experiments with summary information
- Compare experiments by plotting their metrics together
- Generate experiment reports and summaries
- Create an experiment log document
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from experiment_tracker import load_experiment, compare_experiments


def list_experiments(experiments_dir: str = "experiments", verbose: bool = False) -> List[Dict[str, Any]]:
    """
    List all experiments in the experiments directory.
    
    Args:
        experiments_dir: Directory containing experiments
        verbose: Whether to print detailed information
        
    Returns:
        List of experiment summaries
    """
    exp_dir = Path(experiments_dir)
    if not exp_dir.exists():
        print(f"⚠️  Experiments directory {exp_dir} does not exist")
        return []
    
    experiments = []
    
    # Find all experiment directories
    for exp_path in sorted(exp_dir.glob("*")):
        if exp_path.is_dir():
            metrics_file = exp_path / "metrics.json"
            if metrics_file.exists():
                try:
                    data = load_experiment(exp_path)
                    exp_info = data['experiment_info']
                    
                    # Calculate duration
                    start_time = datetime.fromisoformat(exp_info['start_time'])
                    
                    # Get final metrics
                    final_train_loss = None
                    final_val_loss = None
                    
                    if 'train' in data['metrics'] and 'loss' in data['metrics']['train']:
                        train_losses = data['metrics']['train']['loss']
                        if train_losses:
                            final_train_loss = train_losses[-1]['value']
                    
                    if 'val' in data['metrics'] and 'loss' in data['metrics']['val']:
                        val_losses = data['metrics']['val']['loss']
                        if val_losses:
                            final_val_loss = val_losses[-1]['value']
                    
                    experiment = {
                        'experiment_id': exp_info['experiment_id'],
                        'name': exp_info['experiment_name'],
                        'description': exp_info.get('description', ''),
                        'tags': exp_info.get('tags', []),
                        'start_time': start_time,
                        'current_step': exp_info['current_step'],
                        'final_train_loss': final_train_loss,
                        'final_val_loss': final_val_loss,
                        'directory': str(exp_path),
                        'hyperparameters': data.get('hyperparameters', {}),
                        'model_config': data.get('model_config', {})
                    }
                    
                    experiments.append(experiment)
                    
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Could not load experiment {exp_path}: {e}")
    
    # Print summary
    if verbose:
        print(f"📋 Found {len(experiments)} experiments in {exp_dir}")
        print("=" * 120)
        print(f"{'ID':<10} {'Name':<25} {'Steps':<8} {'Train Loss':<12} {'Val Loss':<12} {'Date':<12} {'Tags'}")
        print("=" * 120)
        
        for exp in experiments:
            train_loss_str = f"{exp['final_train_loss']:.4f}" if exp['final_train_loss'] is not None else "N/A"
            val_loss_str = f"{exp['final_val_loss']:.4f}" if exp['final_val_loss'] is not None else "N/A"
            tags_str = ", ".join(exp['tags'][:2]) + ("..." if len(exp['tags']) > 2 else "")
            
            print(f"{exp['experiment_id']:<10} {exp['name'][:24]:<25} "
                  f"{exp['current_step']:<8} {train_loss_str:<12} {val_loss_str:<12} "
                  f"{exp['start_time'].strftime('%m/%d/%Y'):<12} {tags_str}")
    
    return experiments


def compare_experiments_by_tags(experiments_dir: str, tags: List[str], metric: str = 'loss') -> None:
    """
    Compare experiments that have specific tags.
    
    Args:
        experiments_dir: Directory containing experiments
        tags: List of tags to filter by
        metric: Metric to compare
    """
    experiments = list_experiments(experiments_dir, verbose=False)
    
    # Filter experiments by tags
    matching_experiments = []
    for exp in experiments:
        exp_tags = set(exp['tags'])
        if set(tags).issubset(exp_tags):
            matching_experiments.append(exp)
    
    if not matching_experiments:
        print(f"⚠️  No experiments found with tags: {tags}")
        return
    
    print(f"📊 Comparing {len(matching_experiments)} experiments with tags: {tags}")
    
    # Get experiment directories for comparison
    exp_dirs = [exp['directory'] for exp in matching_experiments]
    compare_experiments(exp_dirs, metric)


def generate_experiment_report(experiments_dir: str = "experiments", output_file: str = "EXPERIMENT_LOG.md") -> None:
    """
    Generate a comprehensive experiment report in Markdown format.
    
    Args:
        experiments_dir: Directory containing experiments
        output_file: Output markdown file name
    """
    experiments = list_experiments(experiments_dir, verbose=False)
    
    if not experiments:
        print("⚠️  No experiments found to report")
        return
    
    # Sort experiments by start time
    experiments.sort(key=lambda x: x['start_time'])
    
    # Generate report
    report = []
    report.append("# Experiment Log")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nTotal Experiments: {len(experiments)}")
    report.append("\n---\n")
    
    # Summary table
    report.append("## Experiment Summary\n")
    report.append("| ID | Name | Steps | Train Loss | Val Loss | Date | Tags |")
    report.append("|----|----|----|----|----|----|----|\n")
    
    for exp in experiments:
        train_loss = f"{exp['final_train_loss']:.4f}" if exp['final_train_loss'] is not None else "N/A"
        val_loss = f"{exp['final_val_loss']:.4f}" if exp['final_val_loss'] is not None else "N/A"
        tags = ", ".join(exp['tags']) if exp['tags'] else "-"
        
        report.append(f"| {exp['experiment_id']} | {exp['name']} | {exp['current_step']} | "
                     f"{train_loss} | {val_loss} | {exp['start_time'].strftime('%m/%d/%Y')} | {tags} |")
    
    # Detailed experiment descriptions
    report.append("\n## Detailed Experiments\n")
    
    for i, exp in enumerate(experiments, 1):
        report.append(f"### {i}. {exp['name']} ({exp['experiment_id']})")
        report.append(f"**Date:** {exp['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Steps:** {exp['current_step']}")
        
        if exp['description']:
            report.append(f"**Description:** {exp['description']}")
        
        if exp['tags']:
            report.append(f"**Tags:** {', '.join(exp['tags'])}")
        
        # Results
        report.append("**Results:**")
        if exp['final_train_loss'] is not None:
            report.append(f"- Final Training Loss: {exp['final_train_loss']:.4f}")
        if exp['final_val_loss'] is not None:
            report.append(f"- Final Validation Loss: {exp['final_val_loss']:.4f}")
        
        # Key hyperparameters
        if exp['hyperparameters']:
            report.append("**Key Hyperparameters:**")
            key_params = ['learning_rate', 'batch_size', 'weight_decay', 'max_iterations']
            for param in key_params:
                if param in exp['hyperparameters']:
                    report.append(f"- {param}: {exp['hyperparameters'][param]}")
        
        # Model configuration
        if exp['model_config']:
            report.append("**Model Configuration:**")
            key_config = ['vocab_size', 'd_model', 'num_layers', 'num_heads', 'total_parameters']
            for config in key_config:
                if config in exp['model_config']:
                    value = exp['model_config'][config]
                    if config == 'total_parameters':
                        value = f"{value:,}"
                    report.append(f"- {config}: {value}")
        
        report.append("")  # Empty line between experiments
    
    # Analysis and insights
    report.append("## Analysis and Insights\n")
    
    # Best performing experiments
    train_experiments = [exp for exp in experiments if exp['final_train_loss'] is not None]
    val_experiments = [exp for exp in experiments if exp['final_val_loss'] is not None]
    
    if train_experiments:
        best_train = min(train_experiments, key=lambda x: x['final_train_loss'])
        report.append(f"**Best Training Loss:** {best_train['name']} ({best_train['experiment_id']}) - {best_train['final_train_loss']:.4f}")
    
    if val_experiments:
        best_val = min(val_experiments, key=lambda x: x['final_val_loss'])
        report.append(f"**Best Validation Loss:** {best_val['name']} ({best_val['experiment_id']}) - {best_val['final_val_loss']:.4f}")
    
    # Common tags analysis
    all_tags = []
    for exp in experiments:
        all_tags.extend(exp['tags'])
    
    if all_tags:
        from collections import Counter
        tag_counts = Counter(all_tags)
        report.append(f"\n**Most Common Tags:** {', '.join([f'{tag}({count})' for tag, count in tag_counts.most_common(5)])}")
    
    # Write report to file
    report_content = "\n".join(report)
    
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    print(f"📝 Generated experiment report: {output_file}")
    print(f"📊 Analyzed {len(experiments)} experiments")


def main():
    """Main function for experiment management CLI."""
    parser = argparse.ArgumentParser(description="Experiment Management Utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List all experiments')
    list_parser.add_argument('--experiments_dir', default='experiments', help='Experiments directory')
    list_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information')
    
    # Compare experiments
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('experiment_dirs', nargs='+', help='Experiment directories to compare')
    compare_parser.add_argument('--metric', default='loss', help='Metric to compare')
    
    # Compare by tags
    tags_parser = subparsers.add_parser('compare-tags', help='Compare experiments by tags')
    tags_parser.add_argument('tags', nargs='+', help='Tags to filter by')
    tags_parser.add_argument('--experiments_dir', default='experiments', help='Experiments directory')
    tags_parser.add_argument('--metric', default='loss', help='Metric to compare')
    
    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate experiment report')
    report_parser.add_argument('--experiments_dir', default='experiments', help='Experiments directory')
    report_parser.add_argument('--output', default='EXPERIMENT_LOG.md', help='Output file')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_experiments(args.experiments_dir, args.verbose)
    
    elif args.command == 'compare':
        compare_experiments(args.experiment_dirs, args.metric)
    
    elif args.command == 'compare-tags':
        compare_experiments_by_tags(args.experiments_dir, args.tags, args.metric)
    
    elif args.command == 'report':
        generate_experiment_report(args.experiments_dir, args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()