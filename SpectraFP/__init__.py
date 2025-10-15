"""
SpectraFP: Spectroscopy-based molecular fingerprints.

A package for generating molecular fingerprints from NMR and other 
spectroscopic data for cheminformatics applications.
"""

__version__ = "2.0.0"

try:
    from SpectraFP.spectrafp import SpectraFP, SpectraFP1H, SearchEngine, SearchMetabolitesBy1H
    from spectrafp import numpy_compat
    from spectrafp import cli
    
    __all__ = ["SpectraFP", "SpectraFP1H", "SearchEngine", "SearchMetabolitesBy1H", "numpy_compat", "cli"]
    
except ImportError:
    # Fallback for development or partial installations
    from spectrafp import numpy_compat
    from spectrafp import cli
    __all__ = ["numpy_compat", "cli"]
