"""
NumPy compatibility utilities for SpectraFP.

This module provides compatibility functions to handle differences between 
NumPy versions and ensure optimal performance with spectral fingerprints.
"""

import warnings
from typing import Optional, Union, Tuple, List
import numpy as np


def ensure_numpy_compatibility() -> bool:
    """
    Check NumPy version compatibility and setup optimizations.
    
    Returns:
        bool: True if compatible, False otherwise
    """
    try:
        # Check NumPy version
        np_version = tuple(map(int, np.__version__.split('.')[:2]))
        
        if np_version < (1, 20):
            warnings.warn(
                f"NumPy version {np.__version__} detected. "
                "Please upgrade to NumPy >=1.20.0 for optimal performance.",
                UserWarning
            )
            return False
        
        # Test basic functionality
        test_array = np.random.random((100, 1000))  # Typical spectral data shape
        _ = np.mean(test_array, axis=0)
        _ = np.std(test_array, axis=0)
        
        return True
        
    except Exception as e:
        warnings.warn(f"NumPy compatibility check failed: {e}", UserWarning)
        return False


def safe_arange(start, stop, step, dtype=None) -> np.ndarray:
    """
    Safely create arange with compatibility across NumPy versions.
    
    Args:
        start: Start value
        stop: Stop value  
        step: Step size
        dtype: Data type
        
    Returns:
        NumPy array
    """
    try:
        if dtype is not None:
            return np.arange(start, stop, step, dtype=dtype)
        return np.arange(start, stop, step)
    except Exception as e:
        warnings.warn(f"Arange creation failed: {e}", UserWarning)
        # Fallback with linspace
        n_points = int((stop - start) / step)
        return np.linspace(start, stop - step, n_points)


def safe_dot_product(a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculate dot product with NumPy compatibility.
    
    Args:
        a: First array
        b: Second array
        
    Returns:
        Dot product
    """
    try:
        return np.dot(a, b)
    except Exception as e:
        warnings.warn(f"Dot product failed: {e}", UserWarning)
        # Element-wise multiplication and sum as fallback
        return np.sum(a * b)


def safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity with numerical stability.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity
    """
    try:
        dot_product = safe_dot_product(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    except Exception:
        # Fallback calculation
        dot_product = np.sum(a * b)
        norm_a = np.sqrt(np.sum(a**2))
        norm_b = np.sqrt(np.sum(b**2))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


def safe_array_operations(arr: np.ndarray, operation: str, axis=None) -> np.ndarray:
    """
    Perform array operations with compatibility across NumPy versions.
    
    Args:
        arr: Input array
        operation: Operation name ('sum', 'mean', 'std', 'max', 'min')
        axis: Axis along which to perform operation
        
    Returns:
        Result array
    """
    try:
        if operation == 'sum':
            return np.sum(arr, axis=axis)
        elif operation == 'mean':
            return np.mean(arr, axis=axis)
        elif operation == 'std':
            return np.std(arr, axis=axis)
        elif operation == 'max':
            return np.max(arr, axis=axis)
        elif operation == 'min':
            return np.min(arr, axis=axis)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    except Exception as e:
        warnings.warn(f"Array operation {operation} failed: {e}", UserWarning)
        return arr


def safe_hstack(arrays: List[np.ndarray]) -> np.ndarray:
    """
    Safely horizontally stack arrays with compatibility.
    
    Args:
        arrays: List of arrays to stack
        
    Returns:
        Horizontally stacked array
    """
    try:
        return np.hstack(arrays)
    except Exception as e:
        warnings.warn(f"Horizontal stack failed: {e}", UserWarning)
        # Fallback: concatenate along axis 1
        try:
            return np.concatenate(arrays, axis=1)
        except:
            return arrays[0] if len(arrays) == 1 else np.array([])


def safe_vstack(arrays: List[np.ndarray]) -> np.ndarray:
    """
    Safely vertically stack arrays with compatibility.
    
    Args:
        arrays: List of arrays to stack
        
    Returns:
        Vertically stacked array
    """
    try:
        return np.vstack(arrays)
    except Exception as e:
        warnings.warn(f"Vertical stack failed: {e}", UserWarning)
        # Fallback: concatenate along axis 0
        try:
            return np.concatenate(arrays, axis=0)
        except:
            return arrays[0] if len(arrays) == 1 else np.array([])


def optimize_spectral_data(spectra: np.ndarray, 
                          precision: int = 2,
                          memory_efficient: bool = True) -> np.ndarray:
    """
    Optimize spectral data arrays for memory and precision.
    
    Args:
        spectra: Spectral data array
        precision: Decimal precision for rounding
        memory_efficient: Whether to optimize memory usage
        
    Returns:
        Optimized spectral array
    """
    try:
        # Round to specified precision
        spectra_rounded = np.round(spectra, decimals=precision)
        
        if memory_efficient:
            # Check if we can use lower precision data types
            if spectra_rounded.dtype == np.float64:
                max_val = np.max(np.abs(spectra_rounded))
                if max_val < 1e6:  # Float32 range check
                    return spectra_rounded.astype(np.float32)
            
            # For integer data, check if we can use smaller types
            if np.all(spectra_rounded == spectra_rounded.astype(int)):
                int_data = spectra_rounded.astype(int)
                max_val = np.max(np.abs(int_data))
                
                if max_val < 128:
                    return int_data.astype(np.int8)
                elif max_val < 32768:
                    return int_data.astype(np.int16)
                elif max_val < 2147483648:
                    return int_data.astype(np.int32)
        
        return spectra_rounded
        
    except Exception as e:
        warnings.warn(f"Spectral data optimization failed: {e}", UserWarning)
        return spectra


def validate_spectral_data(spectra: np.ndarray, 
                          ppm_range: Optional[Tuple] = None,
                          name: str = "spectral_data") -> bool:
    """
    Validate spectral data arrays for common issues.
    
    Args:
        spectra: Spectral data array
        ppm_range: Expected PPM range (min, max)
        name: Name for error messages
        
    Returns:
        bool: True if valid
    """
    if not isinstance(spectra, np.ndarray):
        warnings.warn(f"{name} is not a NumPy array", UserWarning)
        return False
    
    if spectra.size == 0:
        warnings.warn(f"{name} is empty", UserWarning)
        return False
    
    if np.any(np.isnan(spectra)):
        warnings.warn(f"{name} contains NaN values", UserWarning)
        return False
    
    if np.any(np.isinf(spectra)):
        warnings.warn(f"{name} contains infinite values", UserWarning)
        return False
    
    if ppm_range is not None:
        min_ppm, max_ppm = ppm_range
        if np.any(spectra < min_ppm) or np.any(spectra > max_ppm):
            warnings.warn(
                f"{name} contains values outside expected range [{min_ppm}, {max_ppm}]",
                UserWarning
            )
            return False
    
    return True


def batch_process_spectra(spectra_list: List[np.ndarray], 
                         func: callable,
                         batch_size: int = 100) -> List:
    """
    Process spectra in batches to manage memory usage.
    
    Args:
        spectra_list: List of spectral arrays
        func: Function to apply to each spectrum
        batch_size: Size of processing batches
        
    Returns:
        List of processed results
    """
    results = []
    n_spectra = len(spectra_list)
    
    for i in range(0, n_spectra, batch_size):
        end_idx = min(i + batch_size, n_spectra)
        batch = spectra_list[i:end_idx]
        
        batch_results = []
        for spectrum in batch:
            try:
                result = func(spectrum)
                batch_results.append(result)
            except Exception as e:
                warnings.warn(f"Error processing spectrum: {e}", UserWarning)
                batch_results.append(None)
        
        results.extend(batch_results)
    
    return results


def safe_fingerprint_matrix(n_samples: int, n_features: int, 
                           dtype=np.float32, fill_value=0) -> np.ndarray:
    """
    Create fingerprint matrix with memory safety.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        dtype: Data type for matrix
        fill_value: Initial fill value
        
    Returns:
        Fingerprint matrix
    """
    try:
        # Check memory requirements (rough estimate)
        memory_gb = (n_samples * n_features * np.dtype(dtype).itemsize) / (1024**3)
        
        if memory_gb > 1.0:  # If more than 1GB
            warnings.warn(
                f"Large matrix requested ({memory_gb:.2f} GB). Consider batch processing.",
                UserWarning
            )
        
        matrix = np.full((n_samples, n_features), fill_value, dtype=dtype)
        return matrix
        
    except MemoryError:
        warnings.warn("Memory error creating matrix. Using smaller data type.", UserWarning)
        # Try with smaller data type
        if dtype == np.float64:
            return safe_fingerprint_matrix(n_samples, n_features, np.float32, fill_value)
        elif dtype == np.float32:
            return safe_fingerprint_matrix(n_samples, n_features, np.int16, fill_value)
        else:
            raise MemoryError("Cannot create matrix even with smallest data type")


def safe_spectral_binning(spectrum: np.ndarray, 
                         ppm_axis: np.ndarray,
                         bin_edges: np.ndarray) -> np.ndarray:
    """
    Bin spectral data with numerical stability.
    
    Args:
        spectrum: Spectral intensity values
        ppm_axis: PPM axis values
        bin_edges: Bin edge positions
        
    Returns:
        Binned spectrum
    """
    try:
        binned = np.zeros(len(bin_edges) - 1)
        
        for i in range(len(bin_edges) - 1):
            mask = (ppm_axis >= bin_edges[i]) & (ppm_axis < bin_edges[i + 1])
            if np.any(mask):
                binned[i] = np.sum(spectrum[mask])
        
        return binned
        
    except Exception as e:
        warnings.warn(f"Spectral binning failed: {e}", UserWarning)
        # Return zeros as fallback
        return np.zeros(len(bin_edges) - 1)


def normalize_fingerprints(fingerprints: np.ndarray, 
                          method: str = 'l2') -> np.ndarray:
    """
    Normalize fingerprint vectors with NumPy compatibility.
    
    Args:
        fingerprints: Fingerprint matrix (n_samples, n_features)
        method: Normalization method ('l1', 'l2', 'max')
        
    Returns:
        Normalized fingerprints
    """
    try:
        if method == 'l2':
            norms = np.sqrt(np.sum(fingerprints**2, axis=1, keepdims=True))
            norms = np.where(norms == 0, 1, norms)
            return fingerprints / norms
        
        elif method == 'l1':
            norms = np.sum(np.abs(fingerprints), axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return fingerprints / norms
        
        elif method == 'max':
            max_vals = np.max(np.abs(fingerprints), axis=1, keepdims=True)
            max_vals = np.where(max_vals == 0, 1, max_vals)
            return fingerprints / max_vals
        
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
            
    except Exception as e:
        warnings.warn(f"Fingerprint normalization failed: {e}", UserWarning)
        return fingerprints