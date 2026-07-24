import numpy as np
import pandas as pd
from itertools import combinations

class PurgedCombinatorialCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)[cite: 3].
    Generates training and testing splits with strict embargo and purging 
    windows to mathematically prevent Look-Ahead Bias[cite: 1].
    """
    def __init__(self, n_folds: int = 6, n_test_folds: int = 2, embargo_td: pd.Timedelta = pd.Timedelta(hours=24)):
        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.embargo_td = embargo_td

    def _generate_fold_boundaries(self, indices: pd.DatetimeIndex) -> list:
        """Splits the chronological index into N equal folds."""
        fold_size = len(indices) // self.n_folds
        boundaries = []
        
        for i in range(self.n_folds):
            start_idx = i * fold_size
            # Ensure the last fold captures the remainder of the dataset
            end_idx = (i + 1) * fold_size if i < self.n_folds - 1 else len(indices)
            boundaries.append((indices[start_idx], indices[end_idx - 1]))
            
        return boundaries

    def split(self, df: pd.DataFrame):
        """
        Yields (train_indices, test_indices) for every combinatorial path.
        Applies purging and embargo logic around the test folds.
        """
        indices = df.index
        fold_boundaries = self._generate_fold_boundaries(indices)
        
        # Generate all possible combinations of test folds
        fold_indices = list(range(self.n_folds))
        test_combinations = list(combinations(fold_indices, self.n_test_folds))
        
        for test_folds in test_combinations:
            test_mask = pd.Series(False, index=indices)
            train_mask = pd.Series(True, index=indices)
            
            # 1. Assign Test Folds
            for fold_idx in test_folds:
                start_time, end_time = fold_boundaries[fold_idx]
                test_mask[(indices >= start_time) & (indices <= end_time)] = True
                
            # 2. Apply Purge and Embargo to Train Folds
            for fold_idx in test_folds:
                start_time, end_time = fold_boundaries[fold_idx]
                
                # Embargo: Drop training data immediately following the test set
                embargo_end = end_time + self.embargo_td
                train_mask[(indices > end_time) & (indices <= embargo_end)] = False
                
                # Purge: Drop training data overlapping with the test set
                train_mask[(indices >= start_time) & (indices <= end_time)] = False

            # Extract integer locations for PyTorch dataloaders
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices

if __name__ == "__main__":
    # CPCV Smoke Test
    print("Initializing Purged Combinatorial CV Firewall...")
    
    # Create 100 days of dummy 15-minute data
    dates = pd.date_range(start="2026-01-01", periods=100 * 96, freq="15min")
    dummy_df = pd.DataFrame({"close": np.random.randn(len(dates))}, index=dates)
    
    # Create 6 total folds, testing on combinations of 2 folds, with a 24-hour embargo
    cpcv = PurgedCombinatorialCV(n_folds=6, n_test_folds=2, embargo_td=pd.Timedelta(hours=24))
    
    path_count = 0
    for train_idx, test_idx in cpcv.split(dummy_df):
        path_count += 1
        print(f"Path {path_count} | Train Samples: {len(train_idx)} | Test Samples: {len(test_idx)}")
        
        # Verify strict separation
        overlap = set(train_idx).intersection(set(test_idx))
        assert len(overlap) == 0, "CRITICAL FAILURE: Data Leakage Detected!"
        
    print(f"Generated {path_count} unique validation paths with zero overlap.")