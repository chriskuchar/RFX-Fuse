"""
Helper to convert categorical_features from column names to indices.

This module provides a utility function to translate user-friendly column names
to 0-based integer indices for the RFX C++ backend.

Usage:
    from RFX import categorical_helper
    
    # With pandas DataFrame
    categorical_features = categorical_helper.convert(['gender', 'category'], X_df)
    
    # With numpy array (must use indices)
    categorical_features = [0, 3, 5]  # Direct indices
"""

def convert_categorical_features(categorical_features, X=None):
    """
    Convert categorical_features from column names (strings) to indices (integers).
    
    Args:
        categorical_features: None, list of column names (str), or list of indices (int)
        X: Input data (numpy array or pandas DataFrame). Required if using column names.
    
    Returns:
        None or list of integer indices (0-based)
    
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'age': [25, 30], 'gender': [0, 1], 'score': [80, 90]})
        >>> convert_categorical_features(['gender'], df)
        [1]
        
        >>> convert_categorical_features([1], df)  # Already indices
        [1]
        
        >>> convert_categorical_features(None, df)  # Auto-detect
        None
    """
    if categorical_features is None:
        return None
    
    # Check if X is a pandas DataFrame with column names
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            # X has column names - convert string names to indices
            result = []
            for feat in categorical_features:
                if isinstance(feat, str):
                    # String column name - find its index
                    if feat in X.columns:
                        result.append(X.columns.get_loc(feat))
                    else:
                        raise ValueError(f"Column '{feat}' not found in DataFrame. Available columns: {list(X.columns)}")
                elif isinstance(feat, int):
                    # Already an integer index - validate it's in range
                    if feat < 0 or feat >= len(X.columns):
                        raise ValueError(f"Index {feat} out of range for DataFrame with {len(X.columns)} columns")
                    result.append(feat)
                else:
                    raise ValueError(f"categorical_features must contain strings (column names) or integers (indices), got {type(feat)}")
            return result
        else:
            # X is a numpy array - categorical_features must be indices
            result = []
            for feat in categorical_features:
                if isinstance(feat, int):
                    # Validate index is in range if X shape is known
                    if hasattr(X, 'shape') and len(X.shape) > 1:
                        if feat < 0 or feat >= X.shape[1]:
                            raise ValueError(f"Index {feat} out of range for array with {X.shape[1]} features")
                    result.append(feat)
                elif isinstance(feat, str):
                    raise ValueError(
                        f"categorical_features contains string '{feat}', but X is not a pandas DataFrame.\n"
                        f"Either:\n"
                        f"  1. Pass a pandas DataFrame as X to use column names, or\n"
                        f"  2. Use integer indices (0-based) for categorical_features"
                    )
                else:
                    raise ValueError(f"categorical_features must contain integers (indices), got {type(feat)}")
            return result
    except ImportError:
        # pandas not available - categorical_features must be indices
        if X is None:
            # No validation possible
            result = []
            for feat in categorical_features:
                if isinstance(feat, int):
                    result.append(feat)
                else:
                    raise ValueError(f"categorical_features must contain integers (indices). To use column names, install pandas.")
            return result
        else:
            # Validate indices against X shape
            result = []
            for feat in categorical_features:
                if isinstance(feat, int):
                    if hasattr(X, 'shape') and len(X.shape) > 1:
                        if feat < 0 or feat >= X.shape[1]:
                            raise ValueError(f"Index {feat} out of range for array with {X.shape[1]} features")
                    result.append(feat)
                else:
                    raise ValueError(f"categorical_features must contain integers (indices). To use column names, install pandas.")
            return result


# Convenience alias
convert = convert_categorical_features

