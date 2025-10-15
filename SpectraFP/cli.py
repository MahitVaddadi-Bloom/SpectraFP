#!/usr/bin/env python3
"""
SpectraFP CLI interface.

Modern command-line interface for spectroscopy-based molecular fingerprints 
with rich output formatting.
"""

import click
import sys
import json
import csv
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.json import JSON
import warnings

try:
    from spectrafp.numpy_compat import ensure_numpy_compatibility, validate_spectral_data
    from SpectraFP.spectrafp import SpectraFP, SpectraFP1H
except ImportError:
    try:
        from numpy_compat import ensure_numpy_compatibility, validate_spectral_data
        from SpectraFP.spectrafp import SpectraFP, SpectraFP1H
    except ImportError:
        # Fallback for standalone testing
        def ensure_numpy_compatibility():
            return True
        def validate_spectral_data(*args, **kwargs):
            return True
        SpectraFP = None
        SpectraFP1H = None

console = Console()


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """
    SpectraFP: Spectroscopy-based molecular fingerprints.
    
    Generate molecular fingerprints from NMR and other spectroscopic data.
    """
    # Check NumPy compatibility on startup
    if not ensure_numpy_compatibility():
        console.print("[yellow]Warning: NumPy compatibility issues detected[/yellow]")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output file for fingerprints (default: input_file_fingerprints.csv)')
@click.option('--range-start', '-s', default=0.0, type=float,
              help='Spectral range start (default: 0.0)')
@click.option('--range-stop', '-e', default=190.0, type=float,
              help='Spectral range stop (default: 190.0)')
@click.option('--range-step', '-t', default=0.1, type=float,
              help='Spectral range step size (default: 0.1)')
@click.option('--correction', '-c', default=0, type=int,
              help='Degree of freedom for signal correction (default: 0)')
@click.option('--precision', '-p', default=1, type=int,
              help='Precision for spectroscopic measures (default: 1)')
@click.option('--no-spurious', is_flag=True,
              help='Remove spurious variables (positions always 0)')
@click.option('--format', 'output_format', default='csv',
              type=click.Choice(['csv', 'json', 'npy']),
              help='Output format')
@click.option('--verbose', '-v', is_flag=True,
              help='Show progress bars')
def fingerprint(input_file, output, range_start, range_stop, range_step, 
                correction, precision, no_spurious, output_format, verbose):
    """
    Generate spectral fingerprints from input data.
    
    INPUT_FILE: CSV file with spectral data (columns: signals or peak lists)
    """
    try:
        if SpectraFP is None:
            console.print("[red]Error: SpectraFP modules not available[/red]")
            console.print("[yellow]Please ensure the package is properly installed[/yellow]")
            sys.exit(1)
        
        # Load input data
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            console.print(f"[red]Error loading input file: {e}[/red]")
            sys.exit(1)
        
        # Set up output path
        if output is None:
            input_path = Path(input_file)
            output = input_path.parent / f"{input_path.stem}_fingerprints.{output_format}"
        
        console.print(f"[bold blue]Generating spectral fingerprints[/bold blue]")
        console.print(f"Input: {input_file}")
        console.print(f"Output: {output}")
        console.print(f"Range: {range_start} - {range_stop} (step: {range_step})")
        console.print(f"Samples: {len(df)}")
        
        # Create SpectraFP object
        spectra_range = [range_start, range_stop, range_step]
        fp_generator = SpectraFP(range_spectra=spectra_range)
        
        # Extract spectral data from DataFrame
        # Assuming the DataFrame has columns with spectral signals
        signals_data = []
        
        if 'signals' in df.columns:
            # If there's a 'signals' column containing lists
            for idx, row in df.iterrows():
                if isinstance(row['signals'], str):
                    # Parse string representation of list
                    signals = eval(row['signals'])
                else:
                    signals = row['signals']
                signals_data.append(signals)
        else:
            # Use all numeric columns as signals
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                console.print("[red]Error: No numeric data found in input file[/red]")
                sys.exit(1)
            
            for idx, row in df.iterrows():
                signals = row[numeric_cols].values.tolist()
                signals_data.append(signals)
        
        # Generate fingerprints
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn() if verbose else TextColumn(""),
            TaskProgressColumn() if verbose else TextColumn(""),
            console=console
        ) as progress:
            task = progress.add_task("Computing fingerprints...", total=len(signals_data))
            
            fingerprints = fp_generator.genBatchFPs(
                data_signs=signals_data,
                correction=correction,
                precision=precision,
                spurious_variables=not no_spurious,
                returnAsDataframe=True,
                verbose=verbose
            )
            
            progress.update(task, completed=len(signals_data),
                          description="✅ Fingerprints generated successfully!")
        
        # Save results
        if output_format == 'csv':
            fingerprints.to_csv(output, index=False)
        elif output_format == 'json':
            fingerprints.to_json(output, orient='records', indent=2)
        elif output_format == 'npy':
            np.save(output, fingerprints.values)
        
        # Display summary
        table = Table(title="Fingerprint Generation Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input samples", str(len(signals_data)))
        table.add_row("Fingerprint length", str(fingerprints.shape[1]))
        table.add_row("Spectral range", f"{range_start} - {range_stop}")
        table.add_row("Step size", str(range_step))
        table.add_row("Correction factor", str(correction))
        table.add_row("Spurious variables", "Removed" if no_spurious else "Kept")
        table.add_row("Output format", output_format.upper())
        table.add_row("Output file", str(output))
        
        console.print(table)
        
        # Statistics
        stats_table = Table(title="Fingerprint Statistics")
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", style="green")
        
        fp_values = fingerprints.values
        stats_table.add_row("Mean density", f"{np.mean(fp_values):.6f}")
        stats_table.add_row("Std density", f"{np.std(fp_values):.6f}")
        stats_table.add_row("Non-zero features", f"{np.count_nonzero(fp_values) / fp_values.size:.4f}")
        stats_table.add_row("Max value", f"{np.max(fp_values):.6f}")
        
        console.print(stats_table)
        
    except ImportError as e:
        console.print(f"[red]Error: Required modules not available: {e}[/red]")
        console.print("[yellow]Please ensure pandas and numpy are installed[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error generating fingerprints: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(),
              help='Output file for 1H NMR fingerprints')
@click.option('--range-start', '-s', default=0.0, type=float,
              help='PPM range start (default: 0.0)')
@click.option('--range-stop', '-e', default=10.0, type=float,
              help='PPM range stop (default: 10.0)')
@click.option('--range-step', '-t', default=0.01, type=float,
              help='PPM range step size (default: 0.01)')
@click.option('--correction', '-c', default=0, type=int,
              help='Correction factor for signal compression')
@click.option('--multiplicity', '-m', multiple=True,
              help='Filter by multiplicity (s, d, t, q, etc.). Use multiple times for multiple values.')
@click.option('--binary', is_flag=True,
              help='Return binary fingerprints (0/1)')
@click.option('--format', 'output_format', default='csv',
              type=click.Choice(['csv', 'json', 'npy']),
              help='Output format')
def nmr1h(input_file, output, range_start, range_stop, range_step, 
          correction, multiplicity, binary, output_format):
    """
    Generate 1H NMR specific fingerprints.
    
    INPUT_FILE: CSV file with 1H NMR peak data
    """
    try:
        if SpectraFP1H is None:
            console.print("[red]Error: SpectraFP1H module not available[/red]")
            sys.exit(1)
        
        # Load input data
        df = pd.read_csv(input_file)
        
        # Set up output path
        if output is None:
            input_path = Path(input_file)
            output = input_path.parent / f"{input_path.stem}_1h_fingerprints.{output_format}"
        
        console.print(f"[bold blue]Generating 1H NMR fingerprints[/bold blue]")
        console.print(f"Input: {input_file}")
        console.print(f"Output: {output}")
        console.print(f"PPM range: {range_start} - {range_stop} (step: {range_step})")
        
        # Set up multiplicity filter
        mult_filter = list(multiplicity) if multiplicity else ['All']
        
        console.print(f"Multiplicity filter: {mult_filter}")
        
        # Create SpectraFP1H object
        range_spectra = [range_start, range_stop, range_step]
        fp_generator = SpectraFP1H(
            range_spectra=range_spectra,
            multiplicty_filter=mult_filter
        )
        
        # Process each row to generate fingerprints
        fingerprints = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing NMR data...", total=len(df))
            
            for idx, row in df.iterrows():
                try:
                    # Extract peaks data - adjust based on your data format
                    if 'peaks' in df.columns:
                        peaks = eval(row['peaks']) if isinstance(row['peaks'], str) else row['peaks']
                    else:
                        # Use all numeric columns as peak positions
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        peaks = row[numeric_cols].dropna().values.tolist()
                    
                    # Generate fingerprint
                    fp = fp_generator.genFP(
                        peaks=peaks,
                        correction=correction,
                        returnAsBinaryValues=binary
                    )
                    
                    fingerprints.append(fp)
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to process row {idx}: {e}[/yellow]")
                    continue
            
            progress.update(task, description="✅ 1H NMR fingerprints generated!")
        
        # Convert to DataFrame
        fp_array = np.array(fingerprints)
        ppm_axis = fp_generator.allppms
        fp_df = pd.DataFrame(fp_array, columns=[f"ppm_{ppm:.2f}" for ppm in ppm_axis])
        
        # Save results
        if output_format == 'csv':
            fp_df.to_csv(output, index=False)
        elif output_format == 'json':
            fp_df.to_json(output, orient='records', indent=2)
        elif output_format == 'npy':
            np.save(output, fp_array)
        
        # Display summary
        table = Table(title="1H NMR Fingerprint Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input samples", str(len(fingerprints)))
        table.add_row("Fingerprint length", str(fp_array.shape[1]))
        table.add_row("PPM range", f"{range_start} - {range_stop}")
        table.add_row("Resolution", str(range_step))
        table.add_row("Multiplicity filter", str(mult_filter))
        table.add_row("Binary values", str(binary))
        table.add_row("Output file", str(output))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error generating 1H NMR fingerprints: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('fingerprints_file', type=click.Path(exists=True))
@click.option('--query', '-q', type=str,
              help='Query fingerprint (comma-separated values) or index')
@click.option('--query-file', type=click.Path(exists=True),
              help='File containing query fingerprint')
@click.option('--metric', default='cosine',
              type=click.Choice(['cosine', 'tanimoto', 'euclidean', 'manhattan']),
              help='Similarity metric')
@click.option('--top-k', '-k', default=10, type=int,
              help='Number of top similar results to return')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for similarity results')
def similarity(fingerprints_file, query, query_file, metric, top_k, output):
    """
    Calculate similarity between fingerprints.
    
    FINGERPRINTS_FILE: CSV file with fingerprint data
    """
    try:
        # Load fingerprints
        df = pd.read_csv(fingerprints_file)
        fp_matrix = df.select_dtypes(include=[np.number]).values
        
        # Get query fingerprint
        if query:
            if query.isdigit():
                # Query by index
                query_idx = int(query)
                if query_idx >= len(fp_matrix):
                    console.print(f"[red]Error: Index {query_idx} out of range[/red]")
                    sys.exit(1)
                query_fp = fp_matrix[query_idx]
                console.print(f"Using fingerprint at index {query_idx} as query")
            else:
                # Parse comma-separated values
                query_fp = np.array([float(x.strip()) for x in query.split(',')])
        elif query_file:
            query_df = pd.read_csv(query_file)
            query_fp = query_df.select_dtypes(include=[np.number]).values[0]
        else:
            console.print("[red]Error: Must provide either --query or --query-file[/red]")
            sys.exit(1)
        
        if output is None:
            input_path = Path(fingerprints_file)
            output = input_path.parent / f"{input_path.stem}_similarity_{metric}.csv"
        
        console.print(f"[bold blue]Calculating fingerprint similarities[/bold blue]")
        console.print(f"Database: {fingerprints_file} ({len(fp_matrix)} fingerprints)")
        console.print(f"Metric: {metric}")
        console.print(f"Top-K: {top_k}")
        
        # Calculate similarities
        similarities = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Computing similarities...", total=len(fp_matrix))
            
            for i, fp in enumerate(fp_matrix):
                if metric == 'cosine':
                    dot_product = np.dot(query_fp, fp)
                    norm_query = np.linalg.norm(query_fp)
                    norm_fp = np.linalg.norm(fp)
                    sim = dot_product / (norm_query * norm_fp) if norm_query * norm_fp > 0 else 0
                
                elif metric == 'tanimoto':
                    intersection = np.sum(np.minimum(query_fp, fp))
                    union = np.sum(np.maximum(query_fp, fp))
                    sim = intersection / union if union > 0 else 0
                
                elif metric == 'euclidean':
                    dist = np.linalg.norm(query_fp - fp)
                    sim = 1 / (1 + dist)  # Convert distance to similarity
                
                elif metric == 'manhattan':
                    dist = np.sum(np.abs(query_fp - fp))
                    sim = 1 / (1 + dist)  # Convert distance to similarity
                
                similarities.append((i, sim))
                progress.update(task, advance=1)
            
            progress.update(task, description="✅ Similarities computed!")
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similarities = similarities[:top_k]
        
        # Create results DataFrame
        results = []
        for idx, sim in top_similarities:
            result = {
                'index': idx,
                'similarity': sim,
                **{f'fp_{i}': fp_matrix[idx][i] for i in range(len(fp_matrix[idx]))}
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output, index=False)
        
        # Display results
        table = Table(title=f"Top {top_k} Similar Fingerprints ({metric})")
        table.add_column("Rank", style="cyan")
        table.add_column("Index", style="green")
        table.add_column("Similarity", style="yellow")
        
        for rank, (idx, sim) in enumerate(top_similarities, 1):
            table.add_row(str(rank), str(idx), f"{sim:.6f}")
        
        console.print(table)
        console.print(f"[green]Results saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error calculating similarities: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('fingerprints_file', type=click.Path(exists=True))
@click.option('--method', default='pca',
              type=click.Choice(['pca', 'variance', 'correlation']),
              help='Analysis method')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for analysis results')
def analyze(fingerprints_file, method, output):
    """
    Analyze fingerprint data for patterns and statistics.
    
    FINGERPRINTS_FILE: CSV file with fingerprint data
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Load fingerprints
        df = pd.read_csv(fingerprints_file)
        fp_matrix = df.select_dtypes(include=[np.number]).values
        
        if output is None:
            input_path = Path(fingerprints_file)
            output = input_path.parent / f"{input_path.stem}_analysis_{method}.json"
        
        console.print(f"[bold blue]Analyzing fingerprint data[/bold blue]")
        console.print(f"Input: {fingerprints_file}")
        console.print(f"Method: {method}")
        console.print(f"Data shape: {fp_matrix.shape}")
        
        analysis_results = {'method': method, 'data_shape': fp_matrix.shape}
        
        if method == 'pca':
            # Principal Component Analysis
            scaler = StandardScaler()
            fp_scaled = scaler.fit_transform(fp_matrix)
            
            n_components = min(10, fp_matrix.shape[0], fp_matrix.shape[1])
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(fp_scaled)
            
            analysis_results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components': n_components,
                'total_variance_explained': float(np.sum(pca.explained_variance_ratio_))
            }
        
        elif method == 'variance':
            # Variance analysis
            feature_variance = np.var(fp_matrix, axis=0)
            feature_mean = np.mean(fp_matrix, axis=0)
            
            analysis_results['variance'] = {
                'feature_variance': feature_variance.tolist(),
                'feature_mean': feature_mean.tolist(),
                'high_variance_features': np.where(feature_variance > np.percentile(feature_variance, 90))[0].tolist(),
                'low_variance_features': np.where(feature_variance < np.percentile(feature_variance, 10))[0].tolist(),
                'variance_statistics': {
                    'mean': float(np.mean(feature_variance)),
                    'std': float(np.std(feature_variance)),
                    'min': float(np.min(feature_variance)),
                    'max': float(np.max(feature_variance))
                }
            }
        
        elif method == 'correlation':
            # Correlation analysis
            if fp_matrix.shape[0] > 1:
                corr_matrix = np.corrcoef(fp_matrix)
                
                # Find highly correlated features
                high_corr_pairs = []
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        if abs(corr_matrix[i, j]) > 0.8:
                            high_corr_pairs.append((i, j, float(corr_matrix[i, j])))
                
                analysis_results['correlation'] = {
                    'correlation_matrix_shape': corr_matrix.shape,
                    'high_correlation_pairs': high_corr_pairs[:50],  # Limit to first 50
                    'mean_correlation': float(np.mean(np.abs(corr_matrix))),
                    'max_correlation': float(np.max(np.abs(corr_matrix[corr_matrix < 1]))),
                }
            else:
                analysis_results['correlation'] = "Insufficient data for correlation analysis"
        
        # Save results
        with open(output, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Display summary
        table = Table(title="Analysis Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Analysis method", method)
        table.add_row("Input shape", f"{fp_matrix.shape[0]} × {fp_matrix.shape[1]}")
        
        if method == 'pca' and 'pca' in analysis_results:
            table.add_row("Components analyzed", str(analysis_results['pca']['n_components']))
            table.add_row("Total variance explained", f"{analysis_results['pca']['total_variance_explained']:.3f}")
        
        table.add_row("Output file", str(output))
        
        console.print(table)
        
    except ImportError:
        console.print("[red]Error: scikit-learn not available for analysis[/red]")
        console.print("[yellow]Install with: pip install scikit-learn[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error in analysis: {e}[/red]")
        sys.exit(1)


@cli.command()
def info():
    """Display system information and SpectraFP capabilities."""
    
    # System info panel
    info_table = Table(title="SpectraFP System Information")
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Version/Status", style="green")
    
    # Python and package versions
    info_table.add_row("Python", f"{sys.version.split()[0]}")
    
    try:
        import numpy as np
        info_table.add_row("NumPy", np.__version__)
        compat_status = "✅ Compatible" if ensure_numpy_compatibility() else "⚠️ Issues detected"
        info_table.add_row("NumPy Compatibility", compat_status)
    except ImportError:
        info_table.add_row("NumPy", "❌ Not installed")
    
    try:
        import pandas as pd
        info_table.add_row("Pandas", pd.__version__)
    except ImportError:
        info_table.add_row("Pandas", "❌ Not installed")
    
    try:
        import sklearn
        info_table.add_row("Scikit-learn", sklearn.__version__)
    except ImportError:
        info_table.add_row("Scikit-learn", "❌ Not installed (optional)")
    
    try:
        import tqdm
        info_table.add_row("TQDM", tqdm.__version__)
    except ImportError:
        info_table.add_row("TQDM", "❌ Not installed")
    
    console.print(info_table)
    
    # Features
    features_panel = Panel(
        """
🧬 [bold]SpectraFP Features:[/bold]
• Spectroscopy-based molecular fingerprints
• Support for general spectral data and 1H NMR
• Configurable spectral ranges and resolutions
• Signal correction and compression algorithms
• Multiplicity filtering for NMR data
• Similarity searching with multiple metrics
• Batch processing for large datasets
• Statistical analysis (PCA, variance, correlation)
• NumPy 2.x compatibility
• Rich CLI interface with progress tracking

📚 [bold]Usage Examples:[/bold]
• spectrafp fingerprint spectra.csv --range-start 0 --range-stop 200
• spectrafp nmr1h nmr_data.csv --multiplicity s d t --binary
• spectrafp similarity fingerprints.csv --query 0 --metric cosine
• spectrafp analyze fingerprints.csv --method pca

📖 [bold]Algorithm:[/bold]
SpectraFP converts spectroscopic signals into binary or continuous 
fingerprints by binning spectral ranges and applying optional 
correction factors for signal fluctuations.

🔧 [bold]Requirements:[/bold]
• Core: numpy, pandas, tqdm
• Analysis: scikit-learn (optional)
• CLI: click, rich
        """,
        title="SpectraFP Capabilities",
        border_style="blue"
    )
    
    console.print(features_panel)


if __name__ == '__main__':
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    cli()