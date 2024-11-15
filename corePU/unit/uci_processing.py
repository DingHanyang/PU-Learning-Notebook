"""
This module provides functions for processing UCI data
"""
import ssl
from ucimlrepo import fetch_ucirepo
ssl._create_default_https_context = ssl._create_unverified_context

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=15)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# metadata
print(breast_cancer_wisconsin_diagnostic.metadata)

# variable information
print(breast_cancer_wisconsin_diagnostic.variables)
