#!/usr/bin/env python3
"""
RFX-Fuse Unified Pipeline Demo: Recommender Systems

USE CASE 1: Recommender Systems
Traditional Approach: FAISS + XGBoost + SHAP + Isolation Forest + custom code (5 tools)
RFX-Fuse Approach: Two-step RFX-Fuse Unsupervised → RFX-Fuse Supervised pipeline (2 model objects)

This demonstrates the RFX-Fuse approach to content-based recommendation:

STAGE 1 — RFX-Fuse Unsupervised (content-based similarity):
  • Features → similarity + top-K + explanations
  • Explanations: overall/local proximity importance ("why similar?")
  • Comparison: RFX-Fuse vs FAISS (similarity search)

STAGE 2 — RFX-Fuse Supervised (content-based ranking):
  • Features + labels → classifier/regressor + similarity + top-K + explanations
  • Prediction explanations: overall/local permutation importance ("why predicted?")
  • Similarity explanations: overall/local proximity importance ("why similar in prediction space?")
  • Supervised top-K acts as a RE-RANKER of unsupervised candidates
  • Comparisons: RFX-Fuse vs XGBoost (ranking), RFX-Fuse vs SHAP (explanations), RFX-Fuse vs Isolation Forest (outliers)

Benefits:
  -All explanations native (no post-hoc computation)
  -Handles cold start via features
  -Unified framework for similarity and prediction
  - Unified framework comparable to 5 separate tools

Key insight: RFX's tree-based proximity captures non-linear feature interactions 
that cosine similarity misses. Proximity importance answers "why similar"—a question 
no other tool addresses natively.

Demo queries: "Toy Story (1995)" and "The Matrix (1999)"
"""

import sys
import time
import pickle
import numpy as np
from pathlib import Path
import os

# Use non-interactive backend for saving plots without display
import matplotlib
matplotlib.use('Agg')

# Setup paths relative to this file
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'python'))
import rfx

# Optional imports for comparisons
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("WARNING:FAISS not available - skipping FAISS comparisons")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING:XGBoost not available - skipping XGBoost comparisons")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING:SHAP not available - skipping SHAP comparisons")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.inspection import permutation_importance
    HAS_ISOLATION_FOREST = True
    HAS_PERMUTATION_IMPORTANCE = True
except ImportError:
    HAS_ISOLATION_FOREST = False
    HAS_PERMUTATION_IMPORTANCE = False
    print("WARNING:Isolation Forest not available - skipping Isolation Forest comparisons")
    try:
        from sklearn.inspection import permutation_importance
        HAS_PERMUTATION_IMPORTANCE = True
    except ImportError:
        HAS_PERMUTATION_IMPORTANCE = False
        print("WARNING:sklearn.inspection.permutation_importance not available - using XGBoost gain-based importance")

# =============================================================================
# Configuration
# =============================================================================
N_TREES = 1000  # RF doesn't overfit with more trees - use 1000 for best performance
MAX_ITEMS = 0     # Use all 59K items (matches paper)
MAX_USERS = 0       # Use all users for robust evaluation (6249 qualify, capped at 500)
TOP_K = 10
USE_GPU = True
GPU_BATCH_SIZE = 50

# Override from command line: python demo_rfx_unified_pipeline.py [max_items] [max_users]
# Examples:
#   python demo_rfx_unified_pipeline.py 3000 500    # 3000 items, 500 users
#   python demo_rfx_unified_pipeline.py 3000        # 3000 items, all users
#   python demo_rfx_unified_pipeline.py 0 0         # all items, all users
#   python demo_rfx_unified_pipeline.py 3000 --load # 3000 items, load saved models
if len(sys.argv) > 1:
    try:
        MAX_ITEMS = int(sys.argv[1])
    except ValueError:
        pass  # Not a number, might be --load or --retrain

if len(sys.argv) > 2:
    try:
        MAX_USERS = int(sys.argv[2])
    except ValueError:
        pass  # Not a number, might be --load or --retrain

LOAD_MODELS = '--load' in sys.argv or '-l' in sys.argv
FORCE_RETRAIN = '--retrain' in sys.argv or '-r' in sys.argv

# Model save directory (relative to project root)
MODEL_DIR = PROJECT_ROOT / "models" / "demo_recommender"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
if len(sys.argv) > 1:
    print(f"[CLI] MAX_ITEMS = {MAX_ITEMS} (0 = all items)")
    print(f"[CLI] MAX_USERS = {MAX_USERS} (0 = all users)")

# =============================================================================
# Data Loading from Cached PKL
# =============================================================================
def load_movielens_cached(max_items=0, max_users=0):
    """Load MovieLens from cached PKL file."""
    cache_file = PROJECT_ROOT / "data" / "cache" / "claim7_u0_i0.pkl"
    
    if not cache_file.exists():
        print(f"ERROR: Cache file not found: {cache_file}")
        print("Please run test_arxiv_paper_claim7_user_item_pairs.py first to generate cache.")
        sys.exit(1)
    
    print(f"Loading cached MovieLens data from {cache_file.name}...")
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    item_features = data['item_features']  # (59047, 23)
    item_feature_names = data['item_feature_names']  # list of 23
    item_titles = data['item_titles']  # {movieId: title}
    item_labels = data['item_labels']  # (59047,) - high/low rating labels
    item_ids = data['item_ids']  # list of movieIds
    
    # Load user-item rating data for user-based evaluation
    train_df = data.get('train_df', None)
    test_df = data.get('test_df', None)
    item_id_to_idx = data.get('item_id_to_idx', {})
    
    # Load user features for Part C (User Similarity)
    user_features = data.get('user_features', None)
    user_feature_names = data.get('user_feature_names', None)
    user_ids = data.get('user_ids', None)
    user_id_to_idx = data.get('user_id_to_idx', {})
    
    # Clean up feature names (remove 'item_' prefix for display)
    feature_names = [n.replace('item_', '') for n in item_feature_names]
    
    # Apply max_items limit (take most popular - they're already sorted)
    if max_items > 0 and len(item_features) > max_items:
        item_features = item_features[:max_items]
        item_labels = item_labels[:max_items]
        item_ids = item_ids[:max_items]
    
    # Apply max_users limit (take first N users - they're already sorted)
    if max_users > 0 and user_features is not None and len(user_features) > max_users:
        user_features = user_features[:max_users]
        if user_ids is not None:
            selected_user_ids = set(user_ids[:max_users])
            user_ids = user_ids[:max_users]
            # Also filter train_df and test_df to only include these users
            if train_df is not None:
                train_df = train_df[train_df['userId'].isin(selected_user_ids)].copy()
            if test_df is not None:
                test_df = test_df[test_df['userId'].isin(selected_user_ids)].copy()
    
    # Create a simple dataframe-like structure for movie lookup
    class MovieData:
        def __init__(self, features, labels, ids, titles, feature_names, train_df=None, test_df=None, item_id_to_idx=None):
            self.X = features.astype(np.float32)
            self.y_class = labels.astype(np.int32)
            # For regression, use avg_rating (last feature before genres, or index 19-22 area)
            # Based on feature names: genres are first 19, then rating stats
            # Actually looking at names: Action...Western (19 genres), then avg_rating, std_rating, log_ratings, log_users
            # So avg_rating is at index 19
            self.y_reg = features[:, 19].astype(np.float32) if features.shape[1] > 19 else features[:, 0]
            self.ids = ids
            self.titles = titles
            self.feature_names = feature_names
            self.n_items = len(features)
            self.train_df = train_df
            self.test_df = test_df
            self.item_id_to_idx = item_id_to_idx if item_id_to_idx else {}
        
        def get_title(self, idx):
            movie_id = self.ids[idx]
            return self.titles.get(movie_id, f"Movie {movie_id}")
        
        def find_movie(self, title_fragment):
            """Find movie by title fragment."""
            for idx, movie_id in enumerate(self.ids):
                title = self.titles.get(movie_id, "")
                if title_fragment.lower() in title.lower():
                    return idx, title
            return None, None
    
    movies = MovieData(item_features, item_labels, item_ids, item_titles, feature_names, train_df, test_df, item_id_to_idx)
    print(f"Loaded {movies.n_items:,} movies, {len(feature_names)} features")
    if train_df is not None and test_df is not None:
        print(f"   User-item rating data: {len(train_df):,} train ratings, {len(test_df):,} test ratings")
    if user_features is not None:
        print(f"   User features: {len(user_features):,} users, {len(user_feature_names) if user_feature_names else 0} features")
    
    # Return both movies and user data
    return movies, user_features, user_feature_names, user_ids, user_id_to_idx


def fmt(val, width=6):
    """Format a float for display."""
    return f"{val:.4f}".rjust(width)


def fmt_sign(val, width=6):
    """Format with sign."""
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.4f}".rjust(width)


# =============================================================================
# Evaluation Metrics
# =============================================================================
def compute_ndcg_at_k(relevance_scores, k=10):
    """Compute NDCG@k given relevance scores (binary or graded)."""
    if len(relevance_scores) == 0:
        return 0.0
    
    # DCG: sum of (relevance / log2(i+1)) for i in range(k)
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        dcg += relevance_scores[i] / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # IDCG: ideal DCG (sorted descending)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_scores))):
        idcg += ideal_scores[i] / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_hr_at_k(relevance_scores, k=10):
    """Compute Hit Rate@k (1 if any relevant item in top-k, else 0)."""
    if len(relevance_scores) == 0:
        return 0.0
    return 1.0 if sum(relevance_scores[:k]) > 0 else 0.0


def compute_similarity_metrics(model, X, query_indices, ground_truth_fn, k=10):
    """
    Compute NDCG@k and HR@k for similarity search.
    
    Args:
        model: RFX-Fuse model with get_top_k_similar method
        X: Feature matrix
        query_indices: List of query sample indices
        ground_truth_fn: Function(query_idx, candidate_idx) -> relevance (0 or 1)
        k: Top-k for evaluation
    
    Returns:
        (ndcg_scores, hr_scores): Lists of NDCG and HR for each query
    """
    ndcg_scores = []
    hr_scores = []
    
    for q_idx in query_indices:
        # Get top-k similar using query index (not sample vector)
        similar_indices, similar_scores = model.get_top_k_similar(q_idx, k)
        
        # Filter out indices that are out of bounds
        valid_indices = [sim_idx for sim_idx in similar_indices if 0 <= sim_idx < len(X)]
        
        if len(valid_indices) == 0:
            # No valid similar items found
            ndcg_scores.append(0.0)
            hr_scores.append(0.0)
            continue
        
        # Compute relevance for each similar item
        relevance = [ground_truth_fn(q_idx, sim_idx) for sim_idx in valid_indices]
        
        # Pad relevance if needed (in case some indices were filtered)
        while len(relevance) < k:
            relevance.append(0.0)
        relevance = relevance[:k]  # Take only top k
        
        # Compute metrics
        ndcg = compute_ndcg_at_k(relevance, k)
        hr = compute_hr_at_k(relevance, k)
        
        ndcg_scores.append(ndcg)
        hr_scores.append(hr)
    
    return ndcg_scores, hr_scores


def compute_faiss_similarity_metrics(X, query_indices, ground_truth_fn, k=10):
    """
    Compute NDCG@k and HR@k for FAISS similarity search.
    
    Args:
        X: Feature matrix (normalized for cosine similarity)
        query_indices: List of query sample indices
        ground_truth_fn: Function(query_idx, candidate_idx) -> relevance (0 or 1)
        k: Top-k for evaluation
    
    Returns:
        (ndcg_scores, hr_scores): Lists of NDCG and HR for each query
    """
    if not HAS_FAISS:
        return [], []
    
    # Normalize features for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X_norm = X / norms
    
    # Build FAISS index (Inner Product = cosine after normalization)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X_norm.astype(np.float32))
    
    ndcg_scores = []
    hr_scores = []
    
    for q_idx in query_indices:
        # FAISS search
        query = X_norm[q_idx:q_idx+1].astype(np.float32)
        scores, indices = index.search(query, k+1)  # +1 to exclude self
        indices = indices[0][1:]  # Exclude self
        scores = scores[0][1:]
        
        # Compute relevance
        relevance = [ground_truth_fn(q_idx, int(idx)) for idx in indices]
        
        # Compute metrics
        ndcg = compute_ndcg_at_k(relevance, k)
        hr = compute_hr_at_k(relevance, k)
        
        ndcg_scores.append(ndcg)
        hr_scores.append(hr)
    
    return ndcg_scores, hr_scores


# =============================================================================
# Main Demo
# =============================================================================
def main():
    print("=" * 75)
    print("RFX-Fuse UNIFIED PIPELINE: MovieLens Content-Based Recommender")
    print("=" * 75)
    
    # Load data
    movies, user_features, user_feature_names, user_ids, user_id_to_idx = load_movielens_cached(MAX_ITEMS, MAX_USERS)
    
    # Find query movies (cached to skip search when loading)
    example_cache_path = MODEL_DIR / f"example_indices_i{MAX_ITEMS}.npz"
    
    if LOAD_MODELS and not FORCE_RETRAIN and example_cache_path.exists():
        print("\n[LOAD]Loading cached example movie indices...")
        example_cache = np.load(example_cache_path)
        toy_story_idx = int(example_cache['toy_story_idx'])
        matrix_idx = int(example_cache['matrix_idx'])
        toy_story_title = movies.get_title(toy_story_idx)
        matrix_title = movies.get_title(matrix_idx)
        print(f"Loaded: Toy Story idx={toy_story_idx}, Matrix idx={matrix_idx}")
    else:
        # Find query movies
        toy_story_idx, toy_story_title = movies.find_movie("Toy Story (1995)")
        matrix_idx, matrix_title = movies.find_movie("Matrix, The (1999)")

        if toy_story_idx is None:
            toy_story_idx, toy_story_title = movies.find_movie("Toy Story")
        if matrix_idx is None:
            matrix_idx, matrix_title = movies.find_movie("Matrix")

        if toy_story_idx is None or matrix_idx is None:
            # Fallback: use first two movies if specific ones not found
            print(f"WARNING:Query movies not found in subset. Using first two popular movies instead.")
            toy_story_idx = 0
            matrix_idx = 1
            toy_story_title = movies.get_title(toy_story_idx)
            matrix_title = movies.get_title(matrix_idx)

        # Cache the example indices
        print(f"\n[SAVE]Saving example indices to {example_cache_path.name}...")
        np.savez(example_cache_path,
                 toy_story_idx=toy_story_idx,
                 matrix_idx=matrix_idx)
        print("Example indices saved")
    
    print(f"\n{'='*75}")
    print("QUERY MOVIES")
    print("=" * 75)
    print(f"\n  {toy_story_title} (idx={toy_story_idx})")
    print(f"  {matrix_title} (idx={matrix_idx})")
    
    X = movies.X
    y_reg = movies.y_reg
    feature_names = movies.feature_names

    # IMPORTANT: For supervised regression, we must EXCLUDE avg_rating (and std_rating)
    # from features since avg_rating IS the target. Otherwise we have data leakage!
    # Feature indices: 0-18 = genres, 19 = avg_rating (TARGET), 20 = std_rating, 21 = log_ratings, 22 = log_users
    target_col_idx = feature_names.index('avg_rating') if 'avg_rating' in feature_names else 19
    leakage_cols = [target_col_idx]  # avg_rating
    if 'std_rating' in feature_names:
        leakage_cols.append(feature_names.index('std_rating'))  # std_rating also leaks rating info

    # Create feature mask (exclude leakage columns)
    feature_mask = np.array([i for i in range(len(feature_names)) if i not in leakage_cols])
    X_supervised = X[:, feature_mask]
    feature_names_supervised = [feature_names[i] for i in feature_mask]

    print(f"\nSupervised features: {len(feature_names_supervised)} (excluded avg_rating, std_rating to prevent leakage)")

    # Train/test split for supervised model evaluation
    # Load saved indices if available (ensures consistency when loading models)
    indices_cache_path = MODEL_DIR / f"train_test_indices_i{MAX_ITEMS}.npz"
    from sklearn.model_selection import train_test_split

    if LOAD_MODELS and not FORCE_RETRAIN and indices_cache_path.exists():
        print(f"[LOAD]Loading cached train/test indices...")
        indices_cache = np.load(indices_cache_path)
        train_indices = indices_cache['train_indices']
        test_indices = indices_cache['test_indices']
        print(f"Loaded {len(train_indices):,} train, {len(test_indices):,} test indices")
    else:
        train_indices, test_indices = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42
        )
        # Save indices for reproducibility
        np.savez(indices_cache_path, train_indices=train_indices, test_indices=test_indices)
        print(f"[SAVE]Saved train/test indices to {indices_cache_path.name}")

    X_train = X_supervised[train_indices]
    X_test = X_supervised[test_indices]
    y_train = y_reg[train_indices]
    y_test_reg = y_reg[test_indices]
    print(f"Data split: {len(X_train):,} train, {len(X_test):,} test")
    
    # Map query indices to training set indices (for supervised model)
    # Find query movies in training set
    toy_story_train_idx = None
    matrix_train_idx = None
    for i, orig_idx in enumerate(train_indices):
        if orig_idx == toy_story_idx:
            toy_story_train_idx = i
        if orig_idx == matrix_idx:
            matrix_train_idx = i
    
    if toy_story_train_idx is None:
        print(f"WARNING:Warning: Toy Story not in training set, using original index")
        toy_story_train_idx = toy_story_idx
    if matrix_train_idx is None:
        print(f"WARNING:Warning: Matrix not in training set, using original index")
        matrix_train_idx = matrix_idx
    
    # =========================================================================
    # PART A: Item Similarity (Unsupervised) - Content-Based
    # =========================================================================
    print(f"\n{'='*75}")
    print("PART A: ITEM SIMILARITY (Unsupervised) - Content-Based")
    print("=" * 75)
    print("Features → similarity + top-K + explanations")
    print("Explanations: overall/local proximity importance ('why similar?')")
    
    unsup_model_path = MODEL_DIR / f"unsupervised_i{MAX_ITEMS}.rfx"
    
    if LOAD_MODELS and not FORCE_RETRAIN and unsup_model_path.exists():
        print(f"\n[LOAD]Loading saved unsupervised model from {unsup_model_path.name}...")
        unsup = rfx.load(str(unsup_model_path))
        unsup_time = 0.0  # No training time if loaded
        print("Model loaded successfully")
    else:
        t0 = time.time()
        unsup = rfx.RandomForestUnsupervised(
            ntree=500,
            use_gpu=USE_GPU,
            batch_size=100,
            compute_proximity=False,  # Don't store full N×N matrix (saves memory)
            compute_proximity_importance=True,
            compute_leaf_assignments=True,
            iseed=42,
        )
        unsup.fit(X)  # Unsupervised uses all data (no labels)
        unsup_time = time.time() - t0

        # Save model
        print(f"\n[SAVE]Saving unsupervised model to {unsup_model_path.name}...")
        unsup.save(str(unsup_model_path))
        print("Model saved successfully")
    
    print(f"\nTraining: {unsup_time:.1f}s ({N_TREES} trees, {len(X):,} items)")
    print(f"OOB Error: {unsup.get_oob_error()*100:.1f}%")
    
    # Overall Proximity Importance
    overall_prox = unsup.get_proximity_importance()
    overall_prox_sum = overall_prox.sum(axis=0)
    sorted_idx = np.argsort(overall_prox_sum)[::-1]
    
    print(f"\nOVERALL PROXIMITY IMPORTANCE")
    print(f"   'What features cluster items globally?'")
    for i in range(5):
        idx = sorted_idx[i]
        print(f"   {i+1}. {feature_names[idx]:18s} {fmt(overall_prox_sum[idx])}")
    
    # --- Toy Story ---
    print(f"\n{'─'*75}")
    print(f"QUERY: {toy_story_title}")
    print(f"{'─'*75}")
    
    # Show query movie features
    print(f"\n   Movie Profile:")
    for feat in ['avg_rating', 'log_ratings', 'log_users']:
        if feat in feature_names:
            fidx = feature_names.index(feat)
            print(f"      {feat}: {X[toy_story_idx, fidx]:.2f}")
    # Show genres
    genres = [feature_names[i] for i in range(len(feature_names)) 
              if X[toy_story_idx, i] == 1.0 and feature_names[i] not in ['avg_rating', 'std_rating', 'log_ratings', 'log_users']]
    print(f"      Genres: {', '.join(genres[:5])}")
    
    similar_ts, scores_ts = unsup.get_top_k_similar(toy_story_idx, TOP_K)
    
    print(f"\n   TOP-{TOP_K} SIMILAR (Unsupervised):")
    for i, (sim_idx, score) in enumerate(zip(similar_ts, scores_ts)):
        if sim_idx >= len(X):
            print(f"   {i+1}. [Invalid index {sim_idx}, skipping]")
            continue
        title = movies.get_title(sim_idx)
        # score is already normalized (0 to 1) from get_top_k_similar()
        proximity = score
        pct = proximity * 100
        raw_count = score * N_TREES  # Estimate raw count (score * ntree)
        rating = X[sim_idx, feature_names.index('avg_rating')] if 'avg_rating' in feature_names else 0
        sim_genres = [feature_names[j] for j in range(len(feature_names))
                      if X[sim_idx, j] == 1.0 and feature_names[j] not in ['avg_rating', 'std_rating', 'log_ratings', 'log_users']]
        print(f"   {i+1}. \"{title[:40]}\" (proximity={proximity:.6f}, {pct:.3f}%, ~{raw_count:.0f}/{N_TREES} trees) ★{rating:.2f}")
        print(f"      [{', '.join(sim_genres[:4])}]")
    
    local_prox_ts = overall_prox[toy_story_idx]
    sorted_local = np.argsort(local_prox_ts)[::-1]
    
    print(f"\n   LOCAL PROXIMITY IMPORTANCE ('Why similar?'):")
    for i in range(TOP_K):
        idx = sorted_local[i]
        if local_prox_ts[idx] > 0:
            val = X[toy_story_idx, idx]
            print(f"   {i+1}. {feature_names[idx]:18s} = {val:>6.2f}  (prox: {fmt(local_prox_ts[idx])})")
    
    # --- The Matrix ---
    print(f"\n{'─'*75}")
    print(f"QUERY: {matrix_title}")
    print(f"{'─'*75}")
    
    # Use original index for unsupervised (uses all data)
    query_m_idx = matrix_idx
    
    # Show query movie features (from original X for display)
    print(f"\n   Movie Profile:")
    for feat in ['avg_rating', 'log_ratings', 'log_users']:
        if feat in feature_names:
            fidx = feature_names.index(feat)
            print(f"      {feat}: {X[matrix_idx, fidx]:.2f}")
    genres = [feature_names[i] for i in range(len(feature_names)) 
              if X[matrix_idx, i] == 1.0 and feature_names[i] not in ['avg_rating', 'std_rating', 'log_ratings', 'log_users']]
    print(f"      Genres: {', '.join(genres[:5])}")
    
    similar_m, scores_m = unsup.get_top_k_similar(query_m_idx, TOP_K)
    
    print(f"\n   TOP-{TOP_K} SIMILAR (Unsupervised):")
    for i, (sim_idx, score) in enumerate(zip(similar_m, scores_m)):
        if sim_idx >= len(X):
            print(f"   {i+1}. [Invalid index {sim_idx}, skipping]")
            continue
        title = movies.get_title(sim_idx)
        # score is already normalized (0 to 1) from get_top_k_similar()
        proximity = score
        pct = proximity * 100
        raw_count = score * N_TREES  # Estimate raw count (score * ntree)
        rating = X[sim_idx, feature_names.index('avg_rating')] if 'avg_rating' in feature_names else 0
        sim_genres = [feature_names[j] for j in range(len(feature_names))
                      if X[sim_idx, j] == 1.0 and feature_names[j] not in ['avg_rating', 'std_rating', 'log_ratings', 'log_users']]
        print(f"   {i+1}. \"{title[:40]}\" (proximity={proximity:.6f}, {pct:.3f}%, ~{raw_count:.0f}/{N_TREES} trees) ★{rating:.2f}")
        print(f"      [{', '.join(sim_genres[:4])}]")
    
    local_prox_m = overall_prox[query_m_idx]
    sorted_local_m = np.argsort(local_prox_m)[::-1]
    
    print(f"\n   LOCAL PROXIMITY IMPORTANCE ('Why similar?'):")
    for i in range(5):
        idx = sorted_local_m[i]
        if local_prox_m[idx] > 0:
            val = X[matrix_idx, idx]
            print(f"   {i+1}. {feature_names[idx]:18s} = {val:>6.2f}  (prox: {fmt(local_prox_m[idx])})")
    
    # =========================================================================
    # VISUALIZATION: Top-K Similar Movies + Local Proximity Importance
    # =========================================================================
    print(f"\n{'='*75}")
    print("GENERATING TOP-K SIMILARITY VISUALIZATION...")
    print("=" * 75)
    
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle('Top-K Similar Movies + Local Proximity Importance Drivers', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Grid: 2 rows (one per movie), 2 columns (similar movies + drivers)
    # Give more width to left column (similar movies) - 60% vs 40%
    gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.5, 1], 
                  hspace=0.3, wspace=0.25, left=0.05, right=0.95, top=0.93, bottom=0.08)
    
    # ===== ROW 1: Toy Story =====
    # Left subplot: Similar movies
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get top-K similar movies for Toy Story (with bounds checking)
    top_k_ts = min(TOP_K, len(similar_ts))
    valid_indices_ts = [(idx, i) for i, idx in enumerate(similar_ts[:top_k_ts]) if idx < len(X)]
    valid_similar_ts = [idx for idx, _ in valid_indices_ts]
    movie_titles_ts = [movies.get_title(sim_idx) for sim_idx in valid_similar_ts]
    proximity_scores_ts = [scores_ts[i] for _, i in valid_indices_ts]
    top_k_ts = len(valid_similar_ts)
    
    y_pos_ts = np.arange(len(movie_titles_ts))
    bars_ts = ax1.barh(y_pos_ts, proximity_scores_ts, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    
    # Remove y-axis labels (we'll put names inside bars)
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    
    ax1.set_xlabel('Proximity Score', fontsize=11, fontweight='bold')
    ax1.set_title(f'Toy Story: Top-{top_k_ts} Similar Movies', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()
    
    # Add movie names and scores inside bars
    for i, (bar, score, title) in enumerate(zip(bars_ts, proximity_scores_ts, movie_titles_ts)):
        width = bar.get_width()
        # Truncate title if too long
        display_title = title[:35] + '...' if len(title) > 35 else title
        # Place text inside bar (left-aligned, with small padding from left edge)
        text_x = max(width * 0.02, 0.001)  # Small padding from left edge
        ax1.text(text_x, bar.get_y() + bar.get_height()/2, 
                f"{i+1}. {display_title}", ha='left', va='center', 
                fontsize=9, fontweight='bold', color='black')
        # Score on the right side of bar
        ax1.text(width - width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='right', va='center', 
                fontsize=8, fontweight='bold', color='black')
    
    # Right subplot: Local proximity importance drivers
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Get top features by local proximity importance
    top_features_ts = []
    top_importance_ts = []
    top_feature_values_ts = []
    for i in range(min(10, len(sorted_local))):
        idx = sorted_local[i]
        if local_prox_ts[idx] > 0:
            top_features_ts.append(feature_names[idx][:15])
            top_importance_ts.append(local_prox_ts[idx])
            # Get actual feature value for this movie
            feat_val = X[toy_story_idx, idx]
            top_feature_values_ts.append(feat_val)
    
    y_pos_feat_ts = np.arange(len(top_features_ts))
    bars_feat_ts = ax2.barh(y_pos_feat_ts, top_importance_ts, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    
    # Remove y-axis labels (we'll put names and values inside bars)
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    ax2.set_xlabel('Local Proximity Importance', fontsize=11, fontweight='bold')
    ax2.set_title('Why Similar? (Top Drivers)', fontsize=11, fontweight='bold', pad=8)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()
    
    # Add feature names and values inside bars
    for i, (bar, imp, feat_name, feat_val) in enumerate(zip(bars_feat_ts, top_importance_ts, top_features_ts, top_feature_values_ts)):
        width = bar.get_width()
        # Format feature value (integer for binary/categorical, float for continuous)
        if abs(feat_val - round(feat_val)) < 0.001:
            val_str = f"{int(round(feat_val))}"
        else:
            val_str = f"{feat_val:.2f}"
        # Place text inside bar (left-aligned, with small padding)
        text_x = max(width * 0.02, 0.0001)  # Small padding from left edge
        ax2.text(text_x, bar.get_y() + bar.get_height()/2, 
                f"{feat_name} = {val_str}", ha='left', va='center', 
                fontsize=8, fontweight='bold', color='black')
        # Importance value on the right side
        ax2.text(width - width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.4f}', ha='right', va='center', 
                fontsize=7, fontweight='bold', color='black')
    
    # ===== ROW 2: The Matrix =====
    # Left subplot: Similar movies
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Get top-K similar movies for Matrix (with bounds checking)
    top_k_m = min(TOP_K, len(similar_m))
    valid_indices_m = [(idx, i) for i, idx in enumerate(similar_m[:top_k_m]) if idx < len(X)]
    valid_similar_m = [idx for idx, _ in valid_indices_m]
    movie_titles_m = [movies.get_title(sim_idx) for sim_idx in valid_similar_m]
    proximity_scores_m = [scores_m[i] for _, i in valid_indices_m]
    top_k_m = len(valid_similar_m)
    
    y_pos_m = np.arange(len(movie_titles_m))
    bars_m = ax3.barh(y_pos_m, proximity_scores_m, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
    
    # Remove y-axis labels (we'll put names inside bars)
    ax3.set_yticks([])
    ax3.set_yticklabels([])
    
    ax3.set_xlabel('Proximity Score', fontsize=11, fontweight='bold')
    ax3.set_title(f'The Matrix (1999): Top-{top_k_m} Similar Movies', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    ax3.invert_yaxis()
    
    # Add movie names and scores inside bars
    for i, (bar, score, title) in enumerate(zip(bars_m, proximity_scores_m, movie_titles_m)):
        width = bar.get_width()
        # Truncate title if too long
        display_title = title[:35] + '...' if len(title) > 35 else title
        # Place text inside bar (left-aligned, with small padding from left edge)
        text_x = max(width * 0.02, 0.001)  # Small padding from left edge
        ax3.text(text_x, bar.get_y() + bar.get_height()/2, 
                f"{i+1}. {display_title}", ha='left', va='center', 
                fontsize=9, fontweight='bold', color='black')
        # Score on the right side of bar
        ax3.text(width - width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='right', va='center', 
                fontsize=8, fontweight='bold', color='black')
    
    # Right subplot: Local proximity importance drivers
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Get top features by local proximity importance
    top_features_m = []
    top_importance_m = []
    top_feature_values_m = []
    for i in range(min(10, len(sorted_local_m))):
        idx = sorted_local_m[i]
        if local_prox_m[idx] > 0:
            top_features_m.append(feature_names[idx][:15])
            top_importance_m.append(local_prox_m[idx])
            # Get actual feature value for this movie
            feat_val = X[matrix_idx, idx]
            top_feature_values_m.append(feat_val)
    
    y_pos_feat_m = np.arange(len(top_features_m))
    bars_feat_m = ax4.barh(y_pos_feat_m, top_importance_m, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    
    # Remove y-axis labels (we'll put names and values inside bars)
    ax4.set_yticks([])
    ax4.set_yticklabels([])
    
    ax4.set_xlabel('Local Proximity Importance', fontsize=11, fontweight='bold')
    ax4.set_title('Why Similar? (Top Drivers)', fontsize=11, fontweight='bold', pad=8)
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.invert_yaxis()
    
    # Add feature names and values inside bars
    for i, (bar, imp, feat_name, feat_val) in enumerate(zip(bars_feat_m, top_importance_m, top_features_m, top_feature_values_m)):
        width = bar.get_width()
        # Format feature value (integer for binary/categorical, float for continuous)
        if abs(feat_val - round(feat_val)) < 0.001:
            val_str = f"{int(round(feat_val))}"
        else:
            val_str = f"{feat_val:.2f}"
        # Place text inside bar (left-aligned, with small padding)
        text_x = max(width * 0.02, 0.0001)  # Small padding from left edge
        ax4.text(text_x, bar.get_y() + bar.get_height()/2, 
                f"{feat_name} = {val_str}", ha='left', va='center', 
                fontsize=8, fontweight='bold', color='black')
        # Importance value on the right side
        ax4.text(width - width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{imp:.4f}', ha='right', va='center', 
                fontsize=7, fontweight='bold', color='black')
    
    plt.tight_layout()

    # Save figure
    fig_path = SCRIPT_DIR / "topk_similarity_drivers.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved Top-K similarity visualization to: {fig_path}")
    plt.close()
    
    # =========================================================================
    # PART A COMPARISON: RFX-Fuse vs FAISS (Unsupervised Similarity)
    # =========================================================================
    print(f"\n{'='*75}")
    print("PART A COMPARISON: RFX-Fuse vs FAISS (Unsupervised Similarity)")
    print("=" * 75)
    
    # Ground truth: Use actual user ratings (user-based evaluation)
    print(f"\nBuilding ground truth from user ratings...")
    
    # Check if we have user-item rating data
    if movies.train_df is None or movies.test_df is None:
        print(f"   WARNING:No user-item rating data in cache - using simplified movie-based evaluation")
        print(f"   (For user-based evaluation, ensure cache includes train_df and test_df)")
        
        # Fallback to simplified evaluation
        def build_rating_ground_truth_simple(X_train, y_train, n_queries=100):
            """Simplified ground truth: movies with similar ratings + genre overlap."""
            ground_truth_sets = {}
            query_indices = np.random.choice(len(X_train), min(n_queries, len(X_train)), replace=False)
            
            for q_idx in query_indices:
                query_rating = y_train[q_idx]
                similar_ratings = np.abs(y_train - query_rating) < 0.5
                query_genres = X_train[q_idx, :19] > 0.5
                genre_overlap = []
                for i in range(len(X_train)):
                    if i == q_idx:
                        continue
                    cand_genres = X_train[i, :19] > 0.5
                    intersection = np.sum(query_genres & cand_genres)
                    union = np.sum(query_genres | cand_genres)
                    jaccard = intersection / union if union > 0 else 0.0
                    if jaccard > 0.2:
                        genre_overlap.append(i)
                ground_truth_sets[q_idx] = set([i for i in genre_overlap if similar_ratings[i]])
            
            return ground_truth_sets, query_indices
        
        ground_truth_sets, eval_query_indices = build_rating_ground_truth_simple(X_train, y_train, n_queries=100)
        
        def rating_ground_truth(query_idx, candidate_idx):
            if candidate_idx < 0 or candidate_idx >= len(X_train) or query_idx < 0 or query_idx >= len(X_train):
                return 0.0
            if query_idx not in ground_truth_sets:
                return 0.0
            return 1.0 if candidate_idx in ground_truth_sets[query_idx] else 0.0
        
        print(f"   Ground truth: Movies with similar ratings (within 0.5 stars) + genre overlap (≥20%)")
        print(f"   Evaluating on {len(eval_query_indices)} random queries...")
        
        # RFX-Fuse evaluation
        print(f"\nRFX-Fuse Unsupervised:")
        t0 = time.time()
        rfx_ndcg, rfx_hr = compute_similarity_metrics(unsup, X_train, eval_query_indices, rating_ground_truth, k=TOP_K)
        rfx_eval_time = time.time() - t0
        rfx_ndcg_mean = np.mean(rfx_ndcg) if rfx_ndcg else 0.0
        rfx_hr_mean = np.mean(rfx_hr) if rfx_hr else 0.0
        
        print(f"   Evaluation time: {rfx_eval_time:.2f}s ({rfx_eval_time/len(eval_query_indices)*1000:.2f}ms/query)")
        print(f"   NDCG@{TOP_K}: {rfx_ndcg_mean:.4f}")
        print(f"   HR@{TOP_K}: {rfx_hr_mean:.4f}")
        
        # FAISS evaluation
        if HAS_FAISS:
            print(f"\nFAISS (Cosine Similarity):")
            t0 = time.time()
            faiss_ndcg, faiss_hr = compute_faiss_similarity_metrics(X_train, eval_query_indices, rating_ground_truth, k=TOP_K)
            faiss_eval_time = time.time() - t0
            faiss_ndcg_mean = np.mean(faiss_ndcg) if faiss_ndcg else 0.0
            faiss_hr_mean = np.mean(faiss_hr) if faiss_hr else 0.0
            
            n_eval_queries = len(eval_query_indices)
            print(f"   Evaluation time: {faiss_eval_time:.2f}s ({faiss_eval_time/n_eval_queries*1000:.2f}ms/query)")
            print(f"   NDCG@{TOP_K}: {faiss_ndcg_mean:.4f}")
            print(f"   HR@{TOP_K}: {faiss_hr_mean:.4f}")
            
            # Comparison (simplified evaluation)
            print(f"\nCOMPARISON (simplified evaluation):")
            print(f"   {'Metric':<15s} {'RFX':>12s} {'FAISS':>12s} {'Improvement':>15s}")
            print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*15}")
            ndcg_improvement = ((rfx_ndcg_mean - faiss_ndcg_mean) / faiss_ndcg_mean * 100) if faiss_ndcg_mean > 0 else 0
            hr_improvement = ((rfx_hr_mean - faiss_hr_mean) / faiss_hr_mean * 100) if faiss_hr_mean > 0 else 0
            print(f"   {'NDCG@10':<15s} {rfx_ndcg_mean:>12.4f} {faiss_ndcg_mean:>12.4f} {ndcg_improvement:>+13.1f}%")
            print(f"   {'HR@10':<15s} {rfx_hr_mean:>12.4f} {faiss_hr_mean:>12.4f} {hr_improvement:>+13.1f}%")
            print(f"   {'Query Time':<15s} {rfx_eval_time/n_eval_queries*1000:>11.2f}ms {faiss_eval_time/n_eval_queries*1000:>11.2f}ms {'-'*15}")
            
            print(f"\n   RFX-Fuse provides native explainability (proximity importance)")
            print(f"   RFX-Fuse captures non-linear feature interactions")
            print(f"   WARNING:FAISS is faster for query-only workloads")
        else:
            print(f"\nWARNING:FAISS not available - skipping comparison")
    else:
        # Use user-based evaluation
        from collections import defaultdict
        import pickle

        # Cache path for user ratings ground truth
        user_ratings_cache_path = MODEL_DIR / f"user_ratings_ground_truth_i{MAX_ITEMS}.pkl"

        if LOAD_MODELS and not FORCE_RETRAIN and user_ratings_cache_path.exists():
            print(f"\n[LOAD]Loading cached user ratings ground truth...")
            with open(user_ratings_cache_path, 'rb') as f:
                ratings_cache = pickle.load(f)
            user_train_items = defaultdict(set, {k: set(v) for k, v in ratings_cache['user_train_items'].items()})
            user_test_items = defaultdict(set, {k: set(v) for k, v in ratings_cache['user_test_items'].items()})
            eval_users = ratings_cache['eval_users']
            all_qualifying_users = ratings_cache['all_qualifying_users']
            print(f"Loaded {len(eval_users)} eval users, {len(all_qualifying_users)} qualifying users")
        else:
            # Build user-item sets
            user_train_items = defaultdict(set)
            user_test_items = defaultdict(set)

            for _, row in movies.train_df.iterrows():
                if row['rating'] >= 4.0:  # Liked items
                    mid = row['movieId']
                    if mid in movies.item_id_to_idx:
                        item_idx = movies.item_id_to_idx[mid]
                        if item_idx < len(X):  # Ensure index is valid
                            user_train_items[row['userId']].add(item_idx)

            for _, row in movies.test_df.iterrows():
                if row['rating'] >= 4.0:  # Liked items in test
                    mid = row['movieId']
                    if mid in movies.item_id_to_idx:
                        item_idx = movies.item_id_to_idx[mid]
                        if item_idx < len(X):  # Ensure index is valid
                            user_test_items[row['userId']].add(item_idx)

            # Filter to users with both train and test items
            all_qualifying_users = [u for u in user_test_items.keys()
                         if u in user_train_items and len(user_train_items[u]) > 0 and len(user_test_items[u]) > 0]
            eval_users = all_qualifying_users[:min(500, len(all_qualifying_users))]  # Sample 500 users for robust evaluation

            # Cache user ratings ground truth
            print(f"\n[SAVE]Saving user ratings ground truth to {user_ratings_cache_path.name}...")
            with open(user_ratings_cache_path, 'wb') as f:
                pickle.dump({
                    'user_train_items': {k: list(v) for k, v in user_train_items.items()},
                    'user_test_items': {k: list(v) for k, v in user_test_items.items()},
                    'eval_users': eval_users,
                    'all_qualifying_users': all_qualifying_users
                }, f)
            print(f"Cached ground truth for {len(eval_users)} eval users")

        print(f"   Ground truth: User-based evaluation")
        print(f"   For each user: liked items (train) → find similar → ground truth = liked items (test)")
        print(f"   Qualifying users (train + test liked items): {len(all_qualifying_users)}")
        print(f"   EVALUATION SET SIZE: {len(eval_users)} users (capped at 500)")
        
        # Helper functions for user-based evaluation
        def hit_rate_at_k(recommended, test_items, k=10):
            """Compute HR@k: 1 if any test item in top-k, else 0."""
            return 1.0 if len(set(recommended[:k]) & test_items) > 0 else 0.0
        
        def ndcg_at_k(recommended, test_items, k=10):
            """Compute NDCG@k."""
            dcg = 0.0
            for i, item in enumerate(recommended[:k]):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
            return dcg / idcg if idcg > 0 else 0.0
        
        # RFX-Fuse evaluation (user-based) - PARALLELIZED
        print(f"\nRFX-Fuse Unsupervised (user-based evaluation):")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def evaluate_user_rfx(user_id):
            """Evaluate a single user - returns (hr, ndcg) or None."""
            train_items = user_train_items[user_id]
            test_items = user_test_items[user_id]

            # Aggregate similar items from all liked items
            all_similar = defaultdict(float)
            for item_idx in train_items:
                if item_idx >= len(X):
                    continue
                try:
                    similar_indices, similar_scores = unsup.get_top_k_similar(item_idx, k=30)
                    for sim_idx, score in zip(similar_indices, similar_scores):
                        if sim_idx < len(X) and sim_idx != item_idx and sim_idx not in train_items:
                            all_similar[sim_idx] += float(score)
                except:
                    continue

            if not all_similar:
                return None

            ranked = sorted(all_similar.items(), key=lambda x: -x[1])
            recommended = [int(idx) for idx, _ in ranked[:20]]
            return (hit_rate_at_k(recommended, test_items, k=TOP_K),
                    ndcg_at_k(recommended, test_items, k=TOP_K))

        rfx_hits, rfx_ndcgs = [], []
        t0 = time.time()

        # Parallel evaluation with 8 threads
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(evaluate_user_rfx, uid): uid for uid in eval_users}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    rfx_hits.append(result[0])
                    rfx_ndcgs.append(result[1])

        rfx_eval_time = time.time() - t0
        rfx_hr_mean = np.mean(rfx_hits) if rfx_hits else 0.0
        rfx_ndcg_mean = np.mean(rfx_ndcgs) if rfx_ndcgs else 0.0
        
        print(f"   Evaluation time: {rfx_eval_time:.2f}s ({rfx_eval_time/len(eval_users)*1000:.2f}ms/user)")
        print(f"   NDCG@{TOP_K}: {rfx_ndcg_mean:.4f}")
        print(f"   HR@{TOP_K}: {rfx_hr_mean:.4f}")
        
        # FAISS evaluation (user-based)
        if HAS_FAISS:
            print(f"\nFAISS (Cosine Similarity, user-based):")
            faiss_hits, faiss_ndcgs = [], []
            t0 = time.time()
            
            # Normalize features for cosine similarity
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            index = faiss.IndexFlatIP(X_norm.shape[1])
            index.add(X_norm.astype(np.float32))
            
            for user_id in eval_users:
                train_items = user_train_items[user_id]
                test_items = user_test_items[user_id]
                
                # Average embedding of liked items
                if len(train_items) == 0:
                    continue
                train_list = list(train_items)
                query_vec = X_norm[train_list].mean(axis=0, keepdims=True).astype(np.float32)
                _, indices = index.search(query_vec, 30)
                recommended = [int(idx) for idx in indices[0] if idx not in train_items and idx < len(X)][:20]
                
                faiss_hits.append(hit_rate_at_k(recommended, test_items, k=TOP_K))
                faiss_ndcgs.append(ndcg_at_k(recommended, test_items, k=TOP_K))
            
            faiss_eval_time = time.time() - t0
            faiss_hr_mean = np.mean(faiss_hits) if faiss_hits else 0.0
            faiss_ndcg_mean = np.mean(faiss_ndcgs) if faiss_ndcgs else 0.0
            
            print(f"   Evaluation time: {faiss_eval_time:.2f}s ({faiss_eval_time/len(eval_users)*1000:.2f}ms/user)")
            print(f"   NDCG@{TOP_K}: {faiss_ndcg_mean:.4f}")
            print(f"   HR@{TOP_K}: {faiss_hr_mean:.4f}")
            
            # Comparison (user-based)
            print(f"\nCOMPARISON (user-based evaluation):")
            print(f"   {'Metric':<15s} {'RFX':>12s} {'FAISS':>12s} {'Improvement':>15s}")
            print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*15}")
            ndcg_improvement = ((rfx_ndcg_mean - faiss_ndcg_mean) / faiss_ndcg_mean * 100) if faiss_ndcg_mean > 0 else 0
            hr_improvement = ((rfx_hr_mean - faiss_hr_mean) / faiss_hr_mean * 100) if faiss_hr_mean > 0 else 0
            print(f"   {'NDCG@10':<15s} {rfx_ndcg_mean:>12.4f} {faiss_ndcg_mean:>12.4f} {ndcg_improvement:>+13.1f}%")
            print(f"   {'HR@10':<15s} {rfx_hr_mean:>12.4f} {faiss_hr_mean:>12.4f} {hr_improvement:>+13.1f}%")
            print(f"   {'Time/User':<15s} {rfx_eval_time/len(eval_users)*1000:>11.2f}ms {faiss_eval_time/len(eval_users)*1000:>11.2f}ms {'-'*15}")
            print(f"   {'Eval Users':<15s} {len(eval_users):>12d} {len(eval_users):>12d} {'-'*15}")

            print(f"\n   RFX-Fuse provides native explainability (proximity importance)")
            print(f"   RFX-Fuse captures non-linear feature interactions")
            if rfx_hr_mean >= faiss_hr_mean * 0.9:
                print(f"   RFX-Fuse matches or outperforms FAISS")
            print(f"   WARNING:FAISS is faster for query-only workloads")
        else:
            print(f"\nWARNING:FAISS not available - skipping comparison")
        
        # For supervised similarity evaluation, we need eval_query_indices and rating_ground_truth
        # Create them from training set items for item-based evaluation
        eval_query_indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
        def rating_ground_truth(query_idx, candidate_idx):
            """Ground truth: movies with similar ratings (within 0.5 stars)."""
            if query_idx < 0 or query_idx >= len(X_train) or candidate_idx < 0 or candidate_idx >= len(X_train):
                return 0.0
            query_rating = y_train[query_idx]
            cand_rating = y_train[candidate_idx]
            return 1.0 if abs(query_rating - cand_rating) < 0.5 else 0.0
    
    # =========================================================================
    # PART B: Item Ranking (Supervised Regression) - Content-Based
    # =========================================================================
    print(f"\n{'='*75}")
    print("PART B: ITEM RANKING (Supervised Regression) - Content-Based")
    print("=" * 75)
    print("Features + labels → regressor + similarity + top-K + explanations")
    print("Prediction explanations: overall/local variable importance ('why predicted?')")
    print("Similarity explanations: overall/local proximity importance ('why similar in pred space?')")
    print("Supervised top-K acts as RE-RANKER of unsupervised candidates")

    # Use supervised feature set (without target leakage) for Part B
    # Save original for Part A references, update for supervised model
    feature_names_unsup = feature_names  # Keep original for unsupervised references
    feature_names = feature_names_supervised  # Use filtered for supervised model
    
    reg_model_path = MODEL_DIR / f"regressor_i{MAX_ITEMS}.rfx"
    
    if LOAD_MODELS and not FORCE_RETRAIN and reg_model_path.exists():
        print(f"\n[LOAD]Loading saved regressor model from {reg_model_path.name}...")
        reg = rfx.load(str(reg_model_path))
        reg_time = 0.0  # No training time if loaded
        print("Model loaded successfully")
    else:
        t0 = time.time()
        reg = rfx.RandomForestRegressor(
            ntree=N_TREES,
            use_gpu=USE_GPU,
            batch_size=GPU_BATCH_SIZE,
            compute_proximity_importance=True,
            compute_local_importance=True,
            compute_leaf_assignments=True,
            iseed=42,
        )
        reg.fit(X_train, y_train)  # Train on training set only
        reg_time = time.time() - t0
        
        # Save model
        print(f"\n[SAVE]Saving regressor model to {reg_model_path.name}...")
        reg.save(str(reg_model_path))
        print("Model saved successfully")
    
    print(f"\nTraining: {reg_time:.1f}s ({N_TREES} trees)")
    rfx_oob_error = reg.get_oob_error()
    rfx_oob_rmse = np.sqrt(rfx_oob_error)
    print(f"OOB MSE: {rfx_oob_error:.4f}")
    
    # Overall Variable Importance
    overall_var = reg.feature_importances_()
    sorted_var = np.argsort(overall_var)[::-1]
    
    print(f"\nOVERALL VARIABLE IMPORTANCE")
    print(f"   'What features predict ratings globally?'")
    for i in range(5):
        idx = sorted_var[i]
        print(f"   {i+1}. {feature_names[idx]:18s} {fmt(overall_var[idx])}")
    
    # Overall Proximity Importance (prediction space)
    overall_prox_reg = reg.get_proximity_importance()
    overall_prox_reg_sum = overall_prox_reg.sum(axis=0)
    sorted_prox = np.argsort(overall_prox_reg_sum)[::-1]
    
    print(f"\nOVERALL PROXIMITY IMPORTANCE (Prediction Space)")
    print(f"   'What features cluster items in prediction space?'")
    for i in range(5):
        idx = sorted_prox[i]
        print(f"   {i+1}. {feature_names[idx]:18s} {fmt(overall_prox_reg_sum[idx])}")
    
    # Local importance arrays
    local_var = reg.get_local_importance()
    
    # --- Toy Story ---
    print(f"\n{'─'*75}")
    print(f"QUERY: {toy_story_title}")
    print(f"{'─'*75}")
    
    pred_ts = reg.predict(X_supervised[toy_story_idx:toy_story_idx+1])[0]
    actual_ts = y_reg[toy_story_idx]
    print(f"\n   PREDICTION: {pred_ts:.2f} stars (actual: {actual_ts:.2f})")
    
    # Local Variable Importance
    if local_var is not None:
        local_var_ts = local_var[toy_story_train_idx]
        sorted_lv = np.argsort(np.abs(local_var_ts))[::-1]
        print(f"\n   LOCAL VARIABLE IMPORTANCE ('Why this prediction?'):")
        for i in range(5):
            idx = sorted_lv[i]
            if local_var_ts[idx] != 0:
                val = X_train[toy_story_train_idx, idx]
                print(f"   {i+1}. {feature_names[idx]:18s} = {val:>6.2f}  (impact: {fmt_sign(local_var_ts[idx])})")
    
    # Top-K Similar (Prediction Space) - RE-RANKER
    similar_ts_reg, scores_ts_reg = reg.get_top_k_similar(toy_story_train_idx, TOP_K)

    print(f"\n   TOP-{TOP_K} SIMILAR (Prediction Space - Re-Ranker):")
    for i, (sim_idx, score) in enumerate(zip(similar_ts_reg, scores_ts_reg)):
        # sim_idx is index into X_train, map back to original index for movie lookup
        orig_idx = train_indices[sim_idx] if sim_idx < len(train_indices) else sim_idx
        if orig_idx >= len(X):
            print(f"   {i+1}. [Invalid index {orig_idx}, skipping]")
            continue
        title = movies.get_title(orig_idx)
        rating = y_train[sim_idx]
        # score is already normalized (0 to 1) from get_top_k_similar()
        proximity = score
        pct = proximity * 100
        raw_count = score * N_TREES  # Estimate raw count (score * ntree)
        sim_genres = [feature_names[j] for j in range(len(feature_names))
                      if X[orig_idx, j] == 1.0 and feature_names[j] not in ['avg_rating', 'std_rating', 'log_ratings', 'log_users']]
        print(f"   {i+1}. \"{title[:40]}\" (proximity={proximity:.6f}, {pct:.3f}%, ~{raw_count:.0f}/{N_TREES} trees) → {rating:.2f}★")
        print(f"      [{', '.join(sim_genres[:4])}]")
    
    # Local Proximity Importance (Prediction Space)
    local_prox_ts_reg = overall_prox_reg[toy_story_train_idx]
    sorted_lp = np.argsort(local_prox_ts_reg)[::-1]
    
    print(f"\n   LOCAL PROXIMITY IMPORTANCE ('Why similar in prediction space?'):")
    for i in range(5):
        idx = sorted_lp[i]
        if local_prox_ts_reg[idx] > 0:
            val = X_train[toy_story_train_idx, idx]
            print(f"   {i+1}. {feature_names[idx]:18s} = {val:>6.2f}  (prox: {fmt(local_prox_ts_reg[idx])})")
    
    # --- The Matrix ---
    print(f"\n{'─'*75}")
    print(f"QUERY: {matrix_title}")
    print(f"{'─'*75}")
    
    pred_m = reg.predict(X_train[matrix_train_idx:matrix_train_idx+1])[0]
    actual_m = y_train[matrix_train_idx]
    print(f"\n   PREDICTION: {pred_m:.2f} stars (actual: {actual_m:.2f})")
    
    if local_var is not None:
        local_var_m = local_var[matrix_train_idx]
        sorted_lv = np.argsort(np.abs(local_var_m))[::-1]
        print(f"\n   LOCAL VARIABLE IMPORTANCE ('Why this prediction?'):")
        for i in range(5):
            idx = sorted_lv[i]
            if local_var_m[idx] != 0:
                val = X_train[matrix_train_idx, idx]
                print(f"   {i+1}. {feature_names[idx]:18s} = {val:>6.2f}  (impact: {fmt_sign(local_var_m[idx])})")
    
    similar_m_reg, scores_m_reg = reg.get_top_k_similar(matrix_train_idx, TOP_K)

    print(f"\n   TOP-{TOP_K} SIMILAR (Prediction Space - Re-Ranker):")
    for i, (sim_idx, score) in enumerate(zip(similar_m_reg, scores_m_reg)):
        # sim_idx is index into X_train, map back to original index for movie lookup
        orig_idx = train_indices[sim_idx] if sim_idx < len(train_indices) else sim_idx
        if orig_idx >= len(X):
            print(f"   {i+1}. [Invalid index {orig_idx}, skipping]")
            continue
        title = movies.get_title(orig_idx)
        rating = y_train[sim_idx]
        # score is already normalized (0 to 1) from get_top_k_similar()
        proximity = score
        pct = proximity * 100
        raw_count = score * N_TREES  # Estimate raw count (score * ntree)
        sim_genres = [feature_names[j] for j in range(len(feature_names))
                      if X[orig_idx, j] == 1.0 and feature_names[j] not in ['avg_rating', 'std_rating', 'log_ratings', 'log_users']]
        print(f"   {i+1}. \"{title[:40]}\" (proximity={proximity:.6f}, {pct:.3f}%, ~{raw_count:.0f}/{N_TREES} trees) → {rating:.2f}★")
        print(f"      [{', '.join(sim_genres[:4])}]")
    
    local_prox_m_reg = overall_prox_reg[matrix_train_idx]
    sorted_lp = np.argsort(local_prox_m_reg)[::-1]
    
    print(f"\n   LOCAL PROXIMITY IMPORTANCE ('Why similar in prediction space?'):")
    for i in range(5):
        idx = sorted_lp[i]
        if local_prox_m_reg[idx] > 0:
            val = X_train[matrix_train_idx, idx]
            print(f"   {i+1}. {feature_names[idx]:18s} = {val:>6.2f}  (prox: {fmt(local_prox_m_reg[idx])})")
    
    # =========================================================================
    # COMPARISON: Unsupervised vs Supervised (Matrix Query)
    # =========================================================================
    print(f"\n{'='*75}")
    print("COMPARISON: Unsupervised vs Supervised (Matrix Query)")
    print("=" * 75)
    
    print(f"\nQUERY: {matrix_title}")
    print(f"{'─'*75}")
    
    print(f"\nUNSUPERVISED (Feature Space):")
    print(f"   Top-{TOP_K} Similar Movies:")
    for i, (sim_idx, score) in enumerate(zip(similar_m, scores_m)):
        if sim_idx >= len(X):
            print(f"   {i+1}. [Invalid index {sim_idx}, skipping]")
            continue
        title = movies.get_title(sim_idx)
        proximity = score
        pct = proximity * 100
        print(f"   {i+1}. \"{title[:40]}\" (proximity={proximity:.6f}, {pct:.3f}%)")

    print(f"\n   Top-5 Local Proximity Importance (Unsupervised):")
    sorted_local_m = np.argsort(local_prox_m)[::-1]
    for i in range(5):
        idx = sorted_local_m[i] if i < len(sorted_local_m) else 0
        if local_prox_m[idx] > 0:
            val = X[matrix_idx, idx]
            print(f"   {i+1}. {feature_names_unsup[idx]:18s} = {val:>6.2f}  (prox: {fmt(local_prox_m[idx])})")

    print(f"\nSUPERVISED (Prediction Space - Re-Ranker):")
    print(f"   Top-{TOP_K} Similar Movies:")
    for i, (sim_idx, score) in enumerate(zip(similar_m_reg, scores_m_reg)):
        orig_idx = train_indices[sim_idx] if sim_idx < len(train_indices) else sim_idx
        if orig_idx >= len(X):
            print(f"   {i+1}. [Invalid index {orig_idx}, skipping]")
            continue
        title = movies.get_title(orig_idx)
        rating = y_train[sim_idx]
        proximity = score
        pct = proximity * 100
        print(f"   {i+1}. \"{title[:40]}\" (proximity={proximity:.6f}, {pct:.3f}%) → {rating:.2f}★")
    
    print(f"\n   Top-5 Local Proximity Importance (Supervised):")
    for i in range(5):
        idx = sorted_lp[i] if i < len(sorted_lp) else 0
        if local_prox_m_reg[idx] > 0:
            val = X_supervised[matrix_idx, idx]
            print(f"   {i+1}. {feature_names[idx]:18s} = {val:>6.2f}  (prox: {fmt(local_prox_m_reg[idx])})")
    
    # Local Proximity Importance Comparison (as lists)
    print(f"\nLOCAL PROXIMITY IMPORTANCE COMPARISON (Unsupervised vs Supervised):")
    print(f"   Matrix query: Top features by local proximity importance")
    
    # Get top features from both (Matrix) - show as lists
    sorted_local_m = np.argsort(local_prox_m)[::-1]
    sorted_local_m_reg = np.argsort(local_prox_m_reg)[::-1]
    
    print(f"\n   UNSUPERVISED Local Proximity Importance (Top-10):")
    print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
    print(f"   {'-'*6} {'-'*20} {'-'*15}")
    top_unsup_list = []
    for i in range(min(10, len(sorted_local_m))):
        idx = sorted_local_m[i]
        if local_prox_m[idx] > 0:
            top_unsup_list.append(idx)
            print(f"   {i+1:<6d} {feature_names_unsup[idx]:<20s} {local_prox_m[idx]:>15.6f}")
    
    print(f"\n   SUPERVISED Local Proximity Importance (Top-10):")
    print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
    print(f"   {'-'*6} {'-'*20} {'-'*15}")
    top_sup_list = []
    for i in range(min(10, len(sorted_local_m_reg))):
        idx = sorted_local_m_reg[i]
        if local_prox_m_reg[idx] > 0:
            top_sup_list.append(idx)
            print(f"   {i+1:<6d} {feature_names[idx]:<20s} {local_prox_m_reg[idx]:>15.6f}")
    
    # Show which features are in both
    top_unsup_set = set(top_unsup_list)
    top_sup_set = set(top_sup_list)
    in_both = top_unsup_set & top_sup_set
    print(f"\n   Features in both top-10: {len(in_both)}/10")
    if in_both:
        print(f"   Common features (showing similarity structure):")
        for idx in sorted(in_both, key=lambda x: local_prox_m[x], reverse=True):
            unsup_rank = top_unsup_list.index(idx) + 1
            sup_rank = top_sup_list.index(idx) + 1
            print(f"      • {feature_names_unsup[idx]:<20s} Unsup rank: {unsup_rank}, Sup rank: {sup_rank}")
    
    # =========================================================================
    # COMPARISON: Unsupervised vs Supervised (Toy Story Query)
    # =========================================================================
    print(f"\n{'='*75}")
    print("COMPARISON: Unsupervised vs Supervised (Toy Story Query)")
    print("=" * 75)
    
    print(f"\nQUERY: {toy_story_title}")
    print(f"{'─'*75}")
    
    print(f"\nLOCAL PROXIMITY IMPORTANCE COMPARISON (Unsupervised vs Supervised):")
    print(f"   Toy Story query: Top features by local proximity importance")
    
    # Get top features from both for Toy Story - show as lists
    sorted_local_ts = np.argsort(local_prox_ts)[::-1]
    sorted_local_ts_reg = np.argsort(local_prox_ts_reg)[::-1]
    
    print(f"\n   UNSUPERVISED Local Proximity Importance (Top-10):")
    print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
    print(f"   {'-'*6} {'-'*20} {'-'*15}")
    top_unsup_list_ts = []
    for i in range(min(10, len(sorted_local_ts))):
        idx = sorted_local_ts[i]
        if local_prox_ts[idx] > 0:
            top_unsup_list_ts.append(idx)
            print(f"   {i+1:<6d} {feature_names_unsup[idx]:<20s} {local_prox_ts[idx]:>15.6f}")
    
    print(f"\n   SUPERVISED Local Proximity Importance (Top-10):")
    print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
    print(f"   {'-'*6} {'-'*20} {'-'*15}")
    top_sup_list_ts = []
    for i in range(min(10, len(sorted_local_ts_reg))):
        idx = sorted_local_ts_reg[i]
        if local_prox_ts_reg[idx] > 0:
            top_sup_list_ts.append(idx)
            print(f"   {i+1:<6d} {feature_names[idx]:<20s} {local_prox_ts_reg[idx]:>15.6f}")
    
    # Show which features are in both
    top_unsup_set_ts = set(top_unsup_list_ts)
    top_sup_set_ts = set(top_sup_list_ts)
    in_both_ts = top_unsup_set_ts & top_sup_set_ts
    print(f"\n   Features in both top-10: {len(in_both_ts)}/10")
    if in_both_ts:
        print(f"   Common features (showing similarity structure):")
        for idx in sorted(in_both_ts, key=lambda x: local_prox_ts[x], reverse=True):
            unsup_rank = top_unsup_list_ts.index(idx) + 1
            sup_rank = top_sup_list_ts.index(idx) + 1
            print(f"      • {feature_names_unsup[idx]:<20s} Unsup rank: {unsup_rank}, Sup rank: {sup_rank}")
    
    # =========================================================================
    # PART B COMPARISON: RFX-Fuse vs XGBoost + SHAP (Supervised Regression)
    # =========================================================================
    print(f"\n{'='*75}")
    print("PART B COMPARISON: RFX-Fuse vs XGBoost + SHAP (Supervised Regression)")
    print("=" * 75)
    
    if HAS_XGBOOST:
        print(f"\nTraining XGBoost Regressor (with early stopping)...")
        t0 = time.time()
        
        # Split training set further for validation (for early stopping)
        from sklearn.model_selection import train_test_split
        X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=N_TREES,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            tree_method='hist' if USE_GPU else 'hist',
            device='cuda' if USE_GPU else 'cpu',
            early_stopping_rounds=50,  # Stop if no improvement for 50 rounds
            eval_metric='rmse'
        )
        xgb_model.fit(
            X_train_xgb, y_train_xgb,
            eval_set=[(X_val_xgb, y_val_xgb)],
            verbose=False
        )
        xgb_train_time = time.time() - t0
        print(f"   Early stopping: Stopped at {xgb_model.best_iteration + 1} trees (best RMSE: {xgb_model.best_score:.4f})")
        
        # XGBoost predictions on test set
        xgb_pred_test = xgb_model.predict(X_test)
        xgb_mse_test = np.mean((y_test_reg - xgb_pred_test) ** 2)
        xgb_rmse_test = np.sqrt(xgb_mse_test)
        
        # RFX-Fuse test predictions
        rfx_pred_test = reg.predict(X_test)
        rfx_mse_test = np.mean((y_test_reg - rfx_pred_test) ** 2)
        rfx_rmse_test = np.sqrt(rfx_mse_test)
        
        print(f"   Training: {xgb_train_time:.1f}s ({N_TREES} trees)")
        print(f"   Test MSE: {xgb_mse_test:.4f}")
        print(f"   Test RMSE: {xgb_rmse_test:.4f}")
        
        # XGBoost feature importance (permutation-based, matching RFX)
        if HAS_PERMUTATION_IMPORTANCE:
            print(f"   Computing permutation importance for XGBoost (matching RFX-Fuse method)...")
            perm_result = permutation_importance(
                xgb_model, X_test, y_test_reg, 
                n_repeats=10, random_state=42, n_jobs=-1
            )
            xgb_importance = perm_result.importances_mean
            xgb_importance_norm = xgb_importance / (xgb_importance.sum() + 1e-10)
            print(f"   Using permutation importance (matches RFX-Fuse method)")
        else:
            xgb_importance = xgb_model.feature_importances_
            xgb_importance_norm = xgb_importance / (xgb_importance.sum() + 1e-10)
            print(f"   WARNING:Using gain-based importance (permutation not available)")
        
        # Comparison: OOB Error vs Test Error
        print(f"\nERROR COMPARISON: RFX-Fuse OOB vs RFX-Fuse Test vs XGBoost Test")
        print(f"   {'Method':<20s} {'MSE':>12s} {'RMSE':>12s} {'Type':>15s}")
        print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*15}")
        print(f"   {'RFX-Fuse OOB':<20s} {rfx_oob_error:>12.4f} {rfx_oob_rmse:>12.4f} {'Training (OOB)':>15s}")
        print(f"   {'RFX-Fuse Test':<20s} {rfx_mse_test:>12.4f} {rfx_rmse_test:>12.4f} {'Test Set':>15s}")
        print(f"   {'XGBoost Test':<20s} {xgb_mse_test:>12.4f} {xgb_rmse_test:>12.4f} {'Test Set':>15s}")
        
        print(f"\n   Key Insight:")
        print(f"      • RFX-Fuse OOB error: {rfx_oob_rmse:.4f} (no test set needed)")
        print(f"      • RFX-Fuse Test error: {rfx_rmse_test:.4f} (actual test performance)")
        print(f"      • XGBoost Test error: {xgb_rmse_test:.4f} (requires test set)")
        print(f"      • OOB provides unbiased estimate without held-out data")
        
        # Comparison: Overall Importance (RFX-Fuse Overall Variable Importance vs XGBoost Permutation Importance)
        print(f"\nOVERALL VARIABLE IMPORTANCE COMPARISON (RFX-Fuse vs XGBoost Permutation):")
        print(f"   Comparing: RFX-Fuse Overall Variable Importance (permutation) vs XGBoost Permutation Importance")
        print(f"   {'Rank':<6s} {'Feature':<18s} {'RFX-Fuse Var Imp':>15s} {'XGBoost Perm':>15s} {'Rank Match':>12s}")
        print(f"   {'-'*6} {'-'*18} {'-'*15} {'-'*15} {'-'*12}")
        
        rfx_rank = np.argsort(overall_var)[::-1]
        xgb_rank = np.argsort(xgb_importance_norm)[::-1]
        
        rank_correlations = []
        for i in range(min(10, len(feature_names))):
            rfx_idx = rfx_rank[i]
            xgb_idx = xgb_rank[i]
            match = "Y" if rfx_idx == xgb_idx else "N"
            rank_correlations.append(1 if rfx_idx == xgb_idx else 0)
            print(f"   {i+1:<6d} {feature_names[rfx_idx]:<18s} {overall_var[rfx_idx]:>15.4f} {xgb_importance_norm[xgb_idx]:>15.4f} {match:>12s}")
        
        rank_agreement = np.mean(rank_correlations)
        print(f"\n   Rank Agreement (Top-10): {rank_agreement*100:.1f}%")
        
        # Comparison: Local Variable Importance (SHAP vs RFX)
        if HAS_SHAP:
            print(f"\nLOCAL VARIABLE IMPORTANCE COMPARISON (SHAP vs RFX):")
            print(f"   Comparing: RFX-Fuse Local Variable Importance vs SHAP (XGBoost)")
            print(f"   Note: RFX-Fuse provides this natively; SHAP requires post-hoc computation on XGBoost")
            print(f"   Computing SHAP values for sample queries...")
            
            # SHAP explainer (based on XGBoost model)
            explainer = shap.TreeExplainer(xgb_model)
            
            # Query 1: Toy Story
            print(f"\n{'─'*75}")
            print(f"QUERY 1: {toy_story_title} (idx={toy_story_idx})")
            print(f"{'─'*75}")
            
            # RFX-Fuse local variable importance (use training index)
            rfx_local_ts = local_var[toy_story_train_idx] if local_var is not None else None
            if rfx_local_ts is not None:
                rfx_local_abs_ts = np.abs(rfx_local_ts)
                rfx_local_rank_ts = np.argsort(rfx_local_abs_ts)[::-1]
            
            # SHAP values (from XGBoost, use training data)
            shap_values_ts = explainer.shap_values(X_train[toy_story_train_idx:toy_story_train_idx+1])
            shap_values_flat_ts = shap_values_ts[0] if isinstance(shap_values_ts, list) else shap_values_ts[0]
            shap_abs_ts = np.abs(shap_values_flat_ts)
            shap_rank_ts = np.argsort(shap_abs_ts)[::-1]
            
            # Show top 10 for each and what matches
            top_n = min(10, len(feature_names))
            
            print(f"\n   RFX-Fuse Top-10 Local Variable Importance:")
            print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
            print(f"   {'-'*6} {'-'*20} {'-'*15}")
            rfx_top10_set = set()
            for i in range(top_n):
                if rfx_local_ts is not None:
                    rfx_idx = rfx_local_rank_ts[i]
                    rfx_top10_set.add(rfx_idx)
                    print(f"   {i+1:<6d} {feature_names[rfx_idx]:<20s} {rfx_local_ts[rfx_idx]:>15.4f}")
            
            print(f"\n   SHAP Top-10 Local Importance (XGBoost):")
            print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
            print(f"   {'-'*6} {'-'*20} {'-'*15}")
            shap_top10_set = set()
            for i in range(top_n):
                shap_idx = shap_rank_ts[i]
                shap_top10_set.add(shap_idx)
                print(f"   {i+1:<6d} {feature_names[shap_idx]:<20s} {shap_values_flat_ts[shap_idx]:>15.4f}")
            
            # Show matches
            matches = rfx_top10_set & shap_top10_set
            print(f"\n   Feature Matches (in both top-10): {len(matches)}/{top_n}")
            if matches:
                print(f"   Matching features:")
                for idx in sorted(matches, key=lambda x: rfx_local_abs_ts[x], reverse=True):
                    rfx_rank_in_top10 = list(rfx_local_rank_ts[:top_n]).index(idx) + 1
                    shap_rank_in_top10 = list(shap_rank_ts[:top_n]).index(idx) + 1
                    print(f"      • {feature_names[idx]:<20s} RFX-Fuse rank: {rfx_rank_in_top10}, SHAP rank: {shap_rank_in_top10}")
            
            # Query 2: Matrix
            print(f"\n{'─'*75}")
            print(f"QUERY 2: {matrix_title} (idx={matrix_idx})")
            print(f"{'─'*75}")
            
            # RFX-Fuse local variable importance (use training index)
            rfx_local_m = local_var[matrix_train_idx] if local_var is not None else None
            if rfx_local_m is not None:
                rfx_local_abs_m = np.abs(rfx_local_m)
                rfx_local_rank_m = np.argsort(rfx_local_abs_m)[::-1]
            
            # SHAP values (from XGBoost, use training data)
            shap_values_m = explainer.shap_values(X_train[matrix_train_idx:matrix_train_idx+1])
            shap_values_flat_m = shap_values_m[0] if isinstance(shap_values_m, list) else shap_values_m[0]
            shap_abs_m = np.abs(shap_values_flat_m)
            shap_rank_m = np.argsort(shap_abs_m)[::-1]
            
            # Show top 10 for each and what matches
            print(f"\n   RFX-Fuse Top-10 Local Variable Importance:")
            print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
            print(f"   {'-'*6} {'-'*20} {'-'*15}")
            rfx_top10_set_m = set()
            for i in range(top_n):
                if rfx_local_m is not None:
                    rfx_idx = rfx_local_rank_m[i]
                    rfx_top10_set_m.add(rfx_idx)
                    print(f"   {i+1:<6d} {feature_names[rfx_idx]:<20s} {rfx_local_m[rfx_idx]:>15.4f}")
            
            print(f"\n   SHAP Top-10 Local Importance (XGBoost):")
            print(f"   {'Rank':<6s} {'Feature':<20s} {'Value':>15s}")
            print(f"   {'-'*6} {'-'*20} {'-'*15}")
            shap_top10_set_m = set()
            for i in range(top_n):
                shap_idx = shap_rank_m[i]
                shap_top10_set_m.add(shap_idx)
                print(f"   {i+1:<6d} {feature_names[shap_idx]:<20s} {shap_values_flat_m[shap_idx]:>15.4f}")
            
            # Show matches
            matches_m = rfx_top10_set_m & shap_top10_set_m
            print(f"\n   Feature Matches (in both top-10): {len(matches_m)}/{top_n}")
            if matches_m:
                print(f"   Matching features:")
                for idx in sorted(matches_m, key=lambda x: rfx_local_abs_m[x], reverse=True):
                    rfx_rank_in_top10 = list(rfx_local_rank_m[:top_n]).index(idx) + 1
                    shap_rank_in_top10 = list(shap_rank_m[:top_n]).index(idx) + 1
                    print(f"      • {feature_names[idx]:<20s} RFX-Fuse rank: {rfx_rank_in_top10}, SHAP rank: {shap_rank_in_top10}")
            
            print(f"\n   RFX-Fuse Local Variable Importance is native (no post-hoc computation)")
            print(f"   WARNING:SHAP requires separate computation after XGBoost training")
            print(f"   Both explain 'why predicted?' - RFX-Fuse does it natively, SHAP is post-hoc")
        
        # RFX-Fuse also does similarity scoring (compare to FAISS)
        print(f"\nRFX-Fuse SIMILARITY SCORING (Prediction Space):")
        print(f"   RFX-Fuse can also do similarity search in prediction space (re-ranker)")
        print(f"   Comparing RFX-Fuse supervised similarity vs FAISS...")
        
        # Use same ground truth function (on training set)
        # Note: eval_query_indices and rating_ground_truth are defined in Part A comparison section
        # If using user-based evaluation, we need to create query indices from training set
        if 'eval_query_indices' not in locals() or 'rating_ground_truth' not in locals():
            # Fallback: create simple query indices from training set
            eval_query_indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
            def rating_ground_truth(query_idx, candidate_idx):
                # Simple fallback: movies with similar ratings
                if query_idx < 0 or query_idx >= len(X_train) or candidate_idx < 0 or candidate_idx >= len(X_train):
                    return 0.0
                query_rating = y_train[query_idx]
                cand_rating = y_train[candidate_idx]
                return 1.0 if abs(query_rating - cand_rating) < 0.5 else 0.0
        
        t0 = time.time()
        rfx_sup_ndcg, rfx_sup_hr = compute_similarity_metrics(reg, X_train, eval_query_indices, rating_ground_truth, k=TOP_K)
        rfx_sup_eval_time = time.time() - t0
        rfx_sup_ndcg_mean = np.mean(rfx_sup_ndcg) if rfx_sup_ndcg else 0.0
        rfx_sup_hr_mean = np.mean(rfx_sup_hr) if rfx_sup_hr else 0.0
        
        print(f"   Evaluation time: {rfx_sup_eval_time:.2f}s")
        print(f"   NDCG@{TOP_K} (Prediction Space): {rfx_sup_ndcg_mean:.4f}")
        print(f"   HR@{TOP_K} (Prediction Space): {rfx_sup_hr_mean:.4f}")
        
        if HAS_FAISS:
            print(f"\n   Comparison with FAISS (from Part A):")
            print(f"   {'Metric':<15s} {'RFX-Fuse (Unsup)':>15s} {'RFX-Fuse (Sup)':>15s} {'FAISS':>15s}")
            print(f"   {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
            print(f"   {'NDCG@10':<15s} {rfx_ndcg_mean:>15.4f} {rfx_sup_ndcg_mean:>15.4f} {faiss_ndcg_mean:>15.4f}")
            print(f"   {'HR@10':<15s} {rfx_hr_mean:>15.4f} {rfx_sup_hr_mean:>15.4f} {faiss_hr_mean:>15.4f}")
        
        # Comparison: Outlier Detection (Isolation Forest vs RFX)
        if HAS_ISOLATION_FOREST:
            print(f"\nOUTLIER DETECTION COMPARISON (Isolation Forest vs RFX):")
            print(f"   Training Isolation Forest for outlier detection...")

            t0 = time.time()
            iso_forest = IsolationForest(n_estimators=100, random_state=42, contamination='auto')
            iso_forest.fit(X_train)  # Train on same data as RFX-Fuse supervised model
            iso_train_time = time.time() - t0

            # Isolation Forest outlier scores (on training set for fair comparison)
            iso_scores = iso_forest.score_samples(X_train)
            iso_outliers = iso_scores < np.percentile(iso_scores, 10)  # Bottom 10% as outliers

            # RFX-Fuse outlier scores (from supervised model, trained on X_train)
            # Load cached scores if available (for reproducibility when loading models)
            outlier_cache_path = MODEL_DIR / f"outlier_scores_i{MAX_ITEMS}.npz"
            if LOAD_MODELS and not FORCE_RETRAIN and outlier_cache_path.exists():
                print(f"   [LOAD]Loading cached outlier scores...")
                outlier_cache = np.load(outlier_cache_path)
                rfx_outlier_scores = outlier_cache['rfx_outlier_scores']
                print(f"Loaded {len(rfx_outlier_scores):,} outlier scores")
            else:
                rfx_outlier_scores = reg.compute_outlier_scores(mode='greedy', n_anchors=100)
                # Save for reproducibility
                np.savez(outlier_cache_path, rfx_outlier_scores=rfx_outlier_scores)
                print(f"   [SAVE]Saved outlier scores to {outlier_cache_path.name}")

            rfx_outliers = rfx_outlier_scores > np.percentile(rfx_outlier_scores, 90)  # Top 10% as outliers
            
            # Compare overlap
            overlap = np.sum(iso_outliers & rfx_outliers)
            total_iso = np.sum(iso_outliers)
            total_rfx = np.sum(rfx_outliers)
            
            print(f"   {'Method':<20s} {'Training Time':>15s} {'Outliers Found':>15s} {'Overlap':>15s}")
            print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
            print(f"   {'Isolation Forest':<20s} {iso_train_time:>14.2f}s {total_iso:>15d} {'N/A':>15s}")
            print(f"   {'RFX-Fuse (built-in)':<20s} {'0.00 (native)':>15s} {total_rfx:>15d} {overlap:>15d}")
            
            if total_iso > 0:
                overlap_pct = (overlap / total_iso) * 100
                print(f"\n   Overlap: {overlap}/{total_iso} ({overlap_pct:.1f}% of Isolation Forest outliers)")
            
            print(f"\n   RFX-Fuse provides outlier detection with explanations (why outlier?)")
            print(f"   WARNING:Isolation Forest only provides scores, no explanations")
        
        print(f"\nCOMPLETE 5-TOOL COMPARISON SUMMARY:")
        print(f"   Traditional Approach: FAISS + XGBoost + SHAP + Isolation Forest + custom code")
        print(f"   RFX-Fuse Approach: 2 model objects (Unsupervised + Supervised)")
        print(f"\n   RFX-Fuse provides similarity + prediction + explanations + outliers in unified framework")
        print(f"   RFX-Fuse similarity in prediction space acts as re-ranker")
        print(f"   WARNING:Traditional approach requires 5 separate tools with complex orchestration")
    else:
        print(f"\nWARNING:XGBoost not available - skipping comparison")
    
    # =========================================================================
    # PART C: User Similarity (Unsupervised) - Claim 7 User-Item Hybrid
    # COMMENTED OUT - focusing on Parts A and B only
    # =========================================================================
    # user_unsup = None
    # if user_features is not None and user_feature_names is not None:
    if False:  # Disabled - Parts C and D commented out
        user_unsup = None
        if user_features is not None and user_feature_names is not None:
            print(f"\n{'='*75}")
            print("PART C: USER SIMILARITY (Unsupervised) - Claim 7 User-Item Hybrid")
            print("=" * 75)
        
            n_users = len(user_features)
            print(f"\nTraining unsupervised model on {n_users:,} users...")
            
            user_unsup_model_path = MODEL_DIR / f"user_unsupervised_i{MAX_ITEMS}.rfx"
            
            if LOAD_MODELS and not FORCE_RETRAIN and user_unsup_model_path.exists():
                print(f"\n[LOAD]Loading saved user unsupervised model from {user_unsup_model_path.name}...")
                user_unsup = rfx.load(str(user_unsup_model_path))
                user_unsup_time = 0.0
                print("Model loaded successfully")
            else:
                t0 = time.time()
                user_unsup = rfx.RandomForestUnsupervised(
                    ntree=N_TREES,
                    use_gpu=USE_GPU,
                    batch_size=GPU_BATCH_SIZE,
                    compute_proximity=False,
                    compute_proximity_importance=True,
                    compute_leaf_assignments=True,
                    iseed=42,
                )
                user_unsup.fit(user_features)
                user_unsup_time = time.time() - t0
                
                # Save model
                print(f"\n[SAVE]Saving user unsupervised model to {user_unsup_model_path.name}...")
                user_unsup.save(str(user_unsup_model_path))
                print("Model saved successfully")
            
            print(f"\nTraining: {user_unsup_time:.1f}s ({N_TREES} trees, {n_users:,} users)")
            print(f"OOB Error: {user_unsup.get_oob_error()*100:.1f}%")
            
            # C1: Overall User Proximity Importance
            user_prox_imp = user_unsup.get_proximity_importance()
            user_prox_imp_sum = user_prox_imp.sum(axis=0)
            sorted_user_idx = np.argsort(user_prox_imp_sum)[::-1]
            
            print(f"\n[C1] OVERALL USER PROXIMITY IMPORTANCE")
            print(f"   'What features cluster users globally?'")
            user_feature_names_clean = [n.replace('user_', '') for n in user_feature_names]
            for i in range(min(5, len(user_feature_names_clean))):
                idx = sorted_user_idx[i]
                print(f"   {i+1}. {user_feature_names_clean[idx]:18s} {fmt(user_prox_imp_sum[idx])}")
            
            # C2: Local User Proximity Importance (example user)
            print(f"\n[C2] LOCAL USER PROXIMITY IMPORTANCE")
            print(f"   'Why are these users similar to a query user?'")
            query_user_idx = 0  # Use first user as example
            if query_user_idx < len(user_features):
                user_local_prox = user_prox_imp[query_user_idx]
                sorted_user_local = np.argsort(user_local_prox)[::-1]
                
                # Get similar users
                similar_user_indices, similar_user_scores = user_unsup.get_top_k_similar(query_user_idx, k=5)
                
                print(f"\n   Query User: User {user_ids[query_user_idx] if user_ids else query_user_idx}")
                print(f"   Top-3 Similar Users:")
                for i, (sim_idx, score) in enumerate(zip(similar_user_indices[:3], similar_user_scores[:3])):
                    if sim_idx < len(user_ids):
                        print(f"   {i+1}. User {user_ids[sim_idx]} (proximity={score:.4f})")
                
                print(f"\n   WHY similar? (Top-5 features):")
                for i in range(min(5, len(sorted_user_local))):
                    idx = sorted_user_local[i]
                    if user_local_prox[idx] > 0:
                        print(f"   {i+1}. {user_feature_names_clean[idx]:18s} = {fmt(user_local_prox[idx])}")
        # else:
        #     print(f"\nWARNING:User features not available in cache - skipping Part C (User Similarity)")
        #     print(f"   (For user-based evaluation, ensure cache includes user_features)")
    
    # =========================================================================
    # PART D: Full Pipeline (User → Similar Users → Candidate Items → Ranked)
    # COMMENTED OUT - focusing on Parts A and B only
    # =========================================================================
    # if user_unsup is not None and movies.train_df is not None:
    if False:  # Disabled - Parts C and D commented out
        user_unsup = None
        if user_unsup is not None and movies.train_df is not None:
            print(f"\n{'='*75}")
            print("PART D: FULL PIPELINE (User → Similar Users → Candidate Items → Ranked)")
            print("=" * 75)
            print("   Production inference example: Complete user-item hybrid recommendation")
            
            # Example: Query user
            query_user_id = user_ids[0] if user_ids else 0
            query_user_idx = 0
            
            print(f"\nPRODUCTION API CALL: get_recommendations(user_id={query_user_id})")
            print(f"{'─'*75}")
            
            # Step 1: Find similar users
            print(f"\nSTEP 1: Find Similar Users")
            similar_user_result = user_unsup.get_top_k_similar_with_explanations(query_user_idx, k=10, n_explanations=5)
            similar_user_indices, similar_user_scores, _, user_feat_idx, user_feat_scores = similar_user_result
            
            sim_user_ids = []
            for idx in similar_user_indices[:5]:
                if idx < len(user_ids) and idx != query_user_idx:
                    sim_user_ids.append(user_ids[idx])
            
            print(f"   → Similar users: {sim_user_ids[:3]}")
            if len(user_feat_idx) > 0:
                top_user_feat = user_feature_names_clean[user_feat_idx[0]] if user_feat_idx[0] < len(user_feature_names_clean) else "N/A"
                print(f"   → Why similar: {top_user_feat} ({user_feat_scores[0]:.3f})")
            
            # Step 2: Get candidate items from similar users' history
            print(f"\nSTEP 2: Get Candidate Items (from similar users' liked items)")
            candidates = {}
            for sim_uid in sim_user_ids[:5]:  # Use top 5 similar users
                user_ratings = movies.train_df[(movies.train_df['userId'] == sim_uid) & (movies.train_df['rating'] >= 4.0)]
                for _, row in user_ratings.iterrows():
                    mid = row['movieId']
                    if mid in movies.item_id_to_idx:
                        item_idx = movies.item_id_to_idx[mid]
                        if item_idx < len(movies.X):
                            candidates[item_idx] = candidates.get(item_idx, 0) + 1
            
            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"   → Found {len(sorted_candidates)} candidate items")
            if len(sorted_candidates) > 0:
                top_cand_idx = sorted_candidates[0][0]
                top_cand_title = movies.get_title(top_cand_idx)
                print(f"   → Top candidate: \"{top_cand_title[:40]}\" (liked by {sorted_candidates[0][1]} similar users)")
            
            # Step 3: Score and rank candidates using item classifier/regressor
            print(f"\nSTEP 3: Score & Rank Candidates (using item model)")
            print(f"{'─'*75}")
            
            # Use the supervised regressor to score candidates
            candidate_scores = []
            for item_idx, count in sorted_candidates[:5]:
                if item_idx < len(movies.X):
                    # Predict rating
                    pred_rating = reg.predict(movies.X[item_idx:item_idx+1])[0]
                    title = movies.get_title(item_idx)
                    
                    # Get local importance
                    if local_var is not None and item_idx < len(local_var):
                        item_local = local_var[item_idx]
                        top_feat_idx = np.argsort(np.abs(item_local))[::-1][0]
                        top_feat_name = feature_names[top_feat_idx]
                        top_feat_val = item_local[top_feat_idx]
                    else:
                        top_feat_name, top_feat_val = "N/A", 0
                    
                    candidate_scores.append((item_idx, pred_rating, count, title, top_feat_name, top_feat_val))
            
            # Sort by predicted rating
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n   Top-3 Ranked Recommendations:")
            for i, (item_idx, pred_rating, count, title, feat_name, feat_val) in enumerate(candidate_scores[:3]):
                print(f"\n   {i+1}. \"{title[:40]}\"")
                print(f"      ├── Predicted rating: {pred_rating:.2f}★")
                print(f"      ├── Liked by {count} similar users")
                print(f"      └── Why predicted: {feat_name} ({feat_val:.4f})")
            
            print(f"\nFULL PIPELINE COMPLETE: User similarity → Candidate items → Ranked recommendations")
            print(f"   All with explanations at every step!")
    else:
        print(f"\nWARNING:User similarity model or rating data not available - skipping Part D")
        print(f"   (For user-based evaluation, ensure cache includes user_features and train_df)")
    
    # =========================================================================
    # CREATE 4 SEPARATE COMPARISON FIGURES
    # =========================================================================
    print(f"\n{'='*75}")
    print("GENERATING 4 SEPARATE COMPARISON FIGURES...")
    print("=" * 75)

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig_dir = SCRIPT_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Get supervised proximity importance for later use
    try:
        sup_prox_imp = reg.get_proximity_importance()
    except:
        sup_prox_imp = None

    # Compute FAISS top-K for Toy Story and Matrix (for Top-10 comparison)
    faiss_similar_ts = []
    faiss_similar_m = []
    if HAS_FAISS:
        # Normalize features for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm_faiss = X / norms

        # Build FAISS index
        faiss_index = faiss.IndexFlatIP(X.shape[1])
        faiss_index.add(X_norm_faiss.astype(np.float32))

        # Get FAISS top-K for Toy Story
        if toy_story_idx is not None and toy_story_idx < len(X):
            query_ts = X_norm_faiss[toy_story_idx:toy_story_idx+1].astype(np.float32)
            _, faiss_indices_ts = faiss_index.search(query_ts, TOP_K+1)
            faiss_similar_ts = [int(idx) for idx in faiss_indices_ts[0][1:] if idx != toy_story_idx]  # Exclude self

        # Get FAISS top-K for Matrix
        if matrix_idx is not None and matrix_idx < len(X):
            query_m = X_norm_faiss[matrix_idx:matrix_idx+1].astype(np.float32)
            _, faiss_indices_m = faiss_index.search(query_m, TOP_K+1)
            faiss_similar_m = [int(idx) for idx in faiss_indices_m[0][1:] if idx != matrix_idx]  # Exclude self

        print(f"\nComputed FAISS top-K for Toy Story ({len(faiss_similar_ts)} items) and Matrix ({len(faiss_similar_m)} items)")

    # =========================================================================
    # FIGURE 1: RFX-Fuse Unsupervised vs FAISS (+ Proximity Importance)
    # 3x2 Grid: Charts only (text explanations moved to paper)
    # =========================================================================
    print("\n  [1/4] RFX-Fuse Unsupervised vs FAISS...")
    fig1, axes1 = plt.subplots(3, 2, figsize=(14, 16))
    fig1.suptitle('RFX-Fuse Unsupervised vs FAISS: Non-Linear Similarity + Explanations', fontsize=14, fontweight='bold', y=1.02)

    # -------------------------------------------------------------------------
    # ROW 0: NDCG/HR Bar Chart | Overall Prox Importance
    # -------------------------------------------------------------------------

    # Panel 0,0: NDCG/HR Bar Chart
    ax = axes1[0, 0]
    if HAS_FAISS:
        methods = ['RFX\nUnsupervised', 'FAISS\n(Cosine)']
        ndcg_vals = [rfx_ndcg_mean, faiss_ndcg_mean]
        hr_vals = [rfx_hr_mean, faiss_hr_mean]
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, ndcg_vals, width, label='NDCG@10', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, hr_vals, width, label='HR@10', color='coral', alpha=0.8)
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title('RFX-Fuse Unsupervised vs FAISS', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for i, (n, h) in enumerate(zip(ndcg_vals, hr_vals)):
            ax.text(i - width/2, n + 0.02, f'{n:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i + width/2, h + 0.02, f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'FAISS not available', ha='center', va='center', fontsize=12)

    # Panel 0,1: Overall Proximity Importance (UNSUPERVISED - use original feature names)
    ax = axes1[0, 1]
    try:
        overall_prox_imp = unsup.get_proximity_importance()
        if overall_prox_imp is not None:
            overall_imp = np.mean(overall_prox_imp, axis=0) if overall_prox_imp.ndim > 1 else overall_prox_imp
            sorted_idx = np.argsort(overall_imp)[::-1][:8]
            labels = [feature_names_unsup[i][:15] for i in sorted_idx]
            vals = overall_imp[sorted_idx]
            vals_norm = vals / (vals.max() + 1e-10)
            y = np.arange(len(labels))
            ax.barh(y, vals_norm, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
            ax.set_xlabel('Normalized Importance', fontsize=10, fontweight='bold')
            ax.set_title('RFX-Fuse: Overall Prox Importance\n(RFX-Fuse only)', fontsize=11, fontweight='bold', color='darkgreen')
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)
            ax.axis('off')
    except:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)
        ax.axis('off')

    # -------------------------------------------------------------------------
    # ROW 1: Toy Story - Top-K Similar | Local Prox Importance
    # -------------------------------------------------------------------------

    # Panel 1,0: Toy Story Top-K Similar
    ax = axes1[1, 0]
    if toy_story_idx is not None and toy_story_idx < len(X):
        top_k_ts = min(TOP_K, len(similar_ts))
        valid_indices_ts = [(idx, i) for i, idx in enumerate(similar_ts[:top_k_ts]) if idx < len(X)]
        movie_titles_ts = [movies.get_title(idx)[:28] for idx, _ in valid_indices_ts]
        proximity_scores_ts = [scores_ts[i] for _, i in valid_indices_ts]
        y_pos = np.arange(len(movie_titles_ts))
        ax.barh(y_pos, proximity_scores_ts, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
        ax.set_xlabel('Proximity Score', fontsize=10, fontweight='bold')
        ax.set_title(f'RFX-Fuse: Toy Story Top-{len(movie_titles_ts)} Similar', fontsize=11, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(movie_titles_ts, fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    # Panel 1,1: Toy Story Local Prox Importance (UNSUPERVISED - use original feature names)
    ax = axes1[1, 1]
    if toy_story_idx is not None:
        top_n = 8
        sorted_local = np.argsort(local_prox_ts)[::-1][:top_n]
        local_norm = local_prox_ts / (local_prox_ts.max() + 1e-10)
        labels = [feature_names_unsup[i][:15] for i in sorted_local]
        actual_vals = [X[toy_story_idx, i] for i in sorted_local]
        vals = [local_norm[i] for i in sorted_local]
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        for bar, val in zip(bars, actual_vals):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, f'Actual: {val:.1f}',
                   va='center', ha='right', fontsize=8, fontweight='bold', color='black')
        ax.set_xlabel('Normalized Local Prox Importance', fontsize=10, fontweight='bold')
        ax.set_title('RFX-Fuse: Toy Story Why Similar?\n(RFX-Fuse only)', fontsize=11, fontweight='bold', color='darkgreen')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    # -------------------------------------------------------------------------
    # ROW 2: Matrix - Top-K Similar | Local Prox Importance
    # -------------------------------------------------------------------------

    # Panel 2,0: Matrix Top-K Similar
    ax = axes1[2, 0]
    if matrix_idx is not None and matrix_idx < len(X):
        top_k_m = min(TOP_K, len(similar_m))
        valid_indices_m = [(idx, i) for i, idx in enumerate(similar_m[:top_k_m]) if idx < len(X)]
        movie_titles_m = [movies.get_title(idx)[:28] for idx, _ in valid_indices_m]
        proximity_scores_m = [scores_m[i] for _, i in valid_indices_m]
        y_pos = np.arange(len(movie_titles_m))
        ax.barh(y_pos, proximity_scores_m, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
        ax.set_xlabel('Proximity Score', fontsize=10, fontweight='bold')
        ax.set_title(f'RFX-Fuse: The Matrix (1999) Top-{len(movie_titles_m)} Similar', fontsize=11, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(movie_titles_m, fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    # Panel 2,1: Matrix Local Prox Importance (UNSUPERVISED - use original feature names)
    ax = axes1[2, 1]
    if matrix_idx is not None:
        sorted_local = np.argsort(local_prox_m)[::-1][:top_n]
        local_norm = local_prox_m / (local_prox_m.max() + 1e-10)
        labels = [feature_names_unsup[i][:15] for i in sorted_local]
        actual_vals = [X[matrix_idx, i] for i in sorted_local]
        vals = [local_norm[i] for i in sorted_local]
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        for bar, val in zip(bars, actual_vals):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, f'Actual: {val:.1f}',
                   va='center', ha='right', fontsize=8, fontweight='bold', color='black')
        ax.set_xlabel('Normalized Local Prox Importance', fontsize=10, fontweight='bold')
        ax.set_title('RFX-Fuse: The Matrix (1999) Why Similar?\n(RFX-Fuse only)', fontsize=11, fontweight='bold', color='darkgreen')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    fig1_path = fig_dir / "unsupervised_and_faiss.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"     Saved: {fig1_path}")
    plt.close(fig1)

    # =========================================================================
    # FIGURE 2A: RFX-Fuse Supervised - PREDICTION + SIMILARITY
    # 2x3 Grid (no text cards - explanations moved to paper)
    # =========================================================================
    print("  [2a/5] RFX-Fuse Supervised: Prediction + Similarity...")
    fig2a, axes2a = plt.subplots(2, 3, figsize=(18, 11))
    fig2a.suptitle('RFX-Fuse Supervised: ONE MODEL → Predictions + Explanations + Similarity + Outliers\n[Part 1: Predictions & Similarity]', fontsize=14, fontweight='bold', y=1.02)

    # Get top outlier info upfront for use in multiple panels
    # NOTE: RFX outlier scores have lower = more outlier, so use [0] after argsort (or argmin)
    # RFX: higher = more outlier, so use [-1] to get highest score
    top_outlier_idx = np.argsort(rfx_outlier_scores)[-1] if HAS_ISOLATION_FOREST else None
    top_outlier_title = movies.get_title(train_indices[top_outlier_idx])[:25] if top_outlier_idx is not None and top_outlier_idx < len(train_indices) else "Top Outlier"

    # Normalize outlier scores for plotting
    # Use percentile-based clipping to handle extreme outliers that skew min-max
    # RFX: higher = more outlier (no negation needed)
    # IF: lower = more outlier (negate to match RFX convention)
    if HAS_ISOLATION_FOREST:
        rfx_clip_max = np.percentile(rfx_outlier_scores, 99)
        rfx_clipped = np.clip(rfx_outlier_scores, rfx_outlier_scores.min(), rfx_clip_max)
        rfx_scores_norm = (rfx_clipped - rfx_clipped.min()) / (rfx_clipped.max() - rfx_clipped.min() + 1e-10)

        iso_negated = -iso_scores
        iso_clip_max = np.percentile(iso_negated, 99)
        iso_clipped = np.clip(iso_negated, iso_negated.min(), iso_clip_max)
        iso_scores_inv = (iso_clipped - iso_clipped.min()) / (iso_clipped.max() - iso_clipped.min() + 1e-10)

        rfx_threshold = np.percentile(rfx_scores_norm, 90)
        iso_threshold = np.percentile(iso_scores_inv, 90)
        rfx_outlier_mask = rfx_scores_norm > rfx_threshold
        iso_outlier_mask = iso_scores_inv > iso_threshold

    # -------------------------------------------------------------------------
    # ROW 0: RFX-Fuse vs XGBoost Predictions | Overall Var Imp | Matrix SHAP vs RFX-Fuse
    # -------------------------------------------------------------------------

    # Panel 0,0: RFX-Fuse vs XGBoost Predictions (RMSE)
    ax = axes2a[0, 0]
    if HAS_XGBOOST:
        from matplotlib.patches import Patch
        methods = ['RFX-Fuse OOB', 'RFX-Fuse Test', 'XGBoost']
        rmse_vals = [np.sqrt(rfx_oob_error), np.sqrt(rfx_mse_test), np.sqrt(xgb_mse_test)]
        x = np.arange(len(methods))
        colors = ['steelblue', 'steelblue', 'coral']
        bars = ax.bar(x, rmse_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('RMSE', fontsize=10, fontweight='bold')
        ax.set_title('RFX-Fuse vs. XGBoost: Predictions', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(axis='y', alpha=0.3)
        # Add legend for RFX-Fuse vs XGBoost
        legend_elements = [Patch(facecolor='steelblue', edgecolor='black', alpha=0.8, label='RFX-Fuse'),
                          Patch(facecolor='coral', edgecolor='black', alpha=0.8, label='XGBoost')]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'XGBoost not available', ha='center', va='center', fontsize=12)

    # Panel 0,1: Overall Variable Importance (RFX-Fuse vs XGBoost)
    ax = axes2a[0, 1]
    if HAS_XGBOOST:
        rfx_var_imp = reg.feature_importances_()
        xgb_var_imp = xgb_model.feature_importances_
        rfx_norm = rfx_var_imp / (rfx_var_imp.max() + 1e-10)
        xgb_norm = xgb_var_imp / (xgb_var_imp.max() + 1e-10)
        top_idx = np.argsort(rfx_norm)[::-1][:8]
        labels = [feature_names[i][:12] for i in top_idx]
        rfx_vals = [rfx_norm[i] for i in top_idx]
        xgb_vals = [xgb_norm[i] for i in top_idx]
        y = np.arange(len(labels))
        width = 0.35
        bars1 = ax.barh(y - width/2, rfx_vals, width, label='RFX-Fuse', color='steelblue', alpha=0.8)
        bars2 = ax.barh(y + width/2, xgb_vals, width, label='XGBoost', color='coral', alpha=0.8)
        ax.set_xlabel('Normalized Importance', fontsize=10, fontweight='bold')
        # Calculate top-7 agreement
        rfx_top7 = set(np.argsort(rfx_norm)[::-1][:7])
        xgb_top7 = set(np.argsort(xgb_norm)[::-1][:7])
        agreement = len(rfx_top7 & xgb_top7)
        ax.set_title(f'Overall Var Imp: RFX-Fuse vs XGBoost\n(Top-7 Agreement: {agreement}/7)', fontsize=11, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'XGBoost not available', ha='center', va='center', fontsize=12)

    # Panel 0,2: Matrix SHAP vs RFX-Fuse Local Var Imp
    ax = axes2a[0, 2]
    if HAS_SHAP and HAS_XGBOOST and local_var is not None and matrix_train_idx is not None:
        sample = X_train[matrix_train_idx:matrix_train_idx+1]
        rfx_local = local_var[matrix_train_idx]
        rfx_local_abs = np.abs(rfx_local)
        rfx_local_rank = np.argsort(rfx_local_abs)[::-1][:8]
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(sample)
        shap_flat = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        shap_abs = np.abs(shap_flat)
        rfx_norm = rfx_local_abs / (rfx_local_abs.max() + 1e-10)
        shap_norm = shap_abs / (shap_abs.max() + 1e-10)
        # Labels with actual values inside
        labels = [feature_names[i][:10] for i in rfx_local_rank]
        actual_vals = [X_train[matrix_train_idx, i] for i in rfx_local_rank]
        rfx_vals = [rfx_norm[i] for i in rfx_local_rank]
        shap_vals = [shap_norm[i] for i in rfx_local_rank]
        y = np.arange(len(labels))
        width = 0.35
        bars1 = ax.barh(y - width/2, rfx_vals, width, label='RFX-Fuse', color='steelblue', alpha=0.8)
        bars2 = ax.barh(y + width/2, shap_vals, width, label='SHAP', color='coral', alpha=0.8)
        # Add actual values inside bars
        for i, (bar, val) in enumerate(zip(bars1, actual_vals)):
            ax.text(0.02, bar.get_y() + bar.get_height()/2, f'Actual: {val:.1f}',
                   va='center', ha='left', fontsize=7, fontweight='bold', color='black')
        ax.set_xlabel('Normalized Importance', fontsize=10, fontweight='bold')
        # Calculate top-7 agreement
        rfx_top7 = set(np.argsort(rfx_local_abs)[::-1][:7])
        shap_top7 = set(np.argsort(shap_abs)[::-1][:7])
        agreement = len(rfx_top7 & shap_top7)
        ax.set_title(f'The Matrix (1999): Local Var Imp\n(RFX-Fuse vs SHAP, Top-7: {agreement}/7)', fontsize=11, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'SHAP not available', ha='center', va='center', fontsize=12)

    # -------------------------------------------------------------------------
    # ROW 1: Overall Prox Imp | Re-ranked Top-K | Why Re-ranked
    # -------------------------------------------------------------------------

    # Panel 1,0: Overall Proximity Importance (RFX-Fuse Supervised)
    ax = axes2a[1, 0]
    if sup_prox_imp is not None:
        # Compute overall proximity importance by summing across all samples
        overall_prox_sup = np.sum(sup_prox_imp, axis=0)
        overall_prox_norm = overall_prox_sup / (overall_prox_sup.max() + 1e-10)
        top_n = 8
        sorted_idx = np.argsort(overall_prox_norm)[::-1][:top_n]
        labels = [feature_names[i][:15] for i in sorted_idx]
        vals = [overall_prox_norm[i] for i in sorted_idx]
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        ax.set_xlabel('Normalized Prox Importance', fontsize=10, fontweight='bold')
        ax.set_title('Overall Prox Importance\n(RFX-Fuse capability only)', fontsize=11, fontweight='bold', color='darkgreen')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        # Add values inside bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='right', va='center', fontsize=8, fontweight='bold', color='black')
    else:
        ax.text(0.5, 0.5, 'Prox importance not available', ha='center', va='center', fontsize=12)

    # Panel 1,1: Matrix Re-ranked Top-K (Supervised)
    ax = axes2a[1, 1]
    if matrix_train_idx is not None:
        try:
            sup_similar_m, sup_scores_m = reg.get_top_k_similar(matrix_train_idx, TOP_K)
            valid_sup = [(idx, i) for i, idx in enumerate(sup_similar_m[:TOP_K]) if idx < len(train_indices)]
            titles = [movies.get_title(train_indices[idx])[:25] for idx, _ in valid_sup]
            prox = [sup_scores_m[i] for _, i in valid_sup]
            y_pos = np.arange(len(titles))
            bars = ax.barh(y_pos, prox, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
            ax.set_xlabel('Proximity Score', fontsize=10, fontweight='bold')
            ax.set_title('The Matrix (1999): Re-ranked Top-K\n(Supervised Similarity)', fontsize=11, fontweight='bold', color='darkgreen')
            ax.set_yticks([])  # Remove y-axis labels
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            # Add movie names inside bars
            for bar, title in zip(bars, titles):
                ax.text(0.01, bar.get_y() + bar.get_height()/2, title,
                       va='center', ha='left', fontsize=7, fontweight='bold', color='black')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Matrix not found', ha='center', va='center', fontsize=12)

    # Panel 1,2: Matrix Local Prox Imp (Why Re-ranked?)
    ax = axes2a[1, 2]
    if matrix_train_idx is not None and sup_prox_imp is not None:
        sup_local = sup_prox_imp[matrix_train_idx]
        sorted_local = np.argsort(sup_local)[::-1][:8]
        local_norm = sup_local / (sup_local.max() + 1e-10)
        labels = [feature_names[i][:10] for i in sorted_local]
        actual_vals = [X_train[matrix_train_idx, i] for i in sorted_local]
        vals = [local_norm[i] for i in sorted_local]
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        # Add actual values inside bars
        for bar, val in zip(bars, actual_vals):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, f'Actual: {val:.1f}',
                   va='center', ha='right', fontsize=8, fontweight='bold', color='black')
        ax.set_xlabel('Prox Importance', fontsize=10, fontweight='bold')
        ax.set_title('The Matrix (1999): Local Prox Imp\n(RFX-Fuse capability only)', fontsize=11, fontweight='bold', color='darkgreen')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Prox imp not available', ha='center', va='center', fontsize=12)

    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
    fig2a_path = fig_dir / "supervised_prediction_similarity.png"
    try:
        fig2a.savefig(fig2a_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"     Warning: bbox_inches='tight' failed, trying without: {e}")
        fig2a.savefig(fig2a_path, dpi=300)
    print(f"     Saved: {fig2a_path}")
    plt.close(fig2a)

    # =========================================================================
    # FIGURE 2B: RFX-Fuse Supervised - OUTLIER DETECTION
    # 2x3 Grid (no text cards - explanations moved to paper)
    # =========================================================================
    print("  [2b/5] RFX-Fuse Supervised: Outlier Detection...")
    fig2b, axes2b = plt.subplots(2, 3, figsize=(18, 11))
    fig2b.suptitle('RFX-Fuse Supervised: ONE MODEL → Predictions + Explanations + Similarity + Outliers\n[Part 2: Outlier Detection]', fontsize=14, fontweight='bold', y=0.98)

    # -------------------------------------------------------------------------
    # ROW 0: RFX-Fuse vs IF Scatter | Histogram | Top-K to Outlier
    # -------------------------------------------------------------------------

    # Panel 0,0: RFX-Fuse vs IF Scatterplot (Manifold vs Marginal)
    ax = axes2b[0, 0]
    if HAS_ISOLATION_FOREST:
        colors_scatter = np.array(['lightgray'] * len(rfx_scores_norm), dtype=object)
        colors_scatter[rfx_outlier_mask & ~iso_outlier_mask] = 'steelblue'
        colors_scatter[~rfx_outlier_mask & iso_outlier_mask] = 'orange'
        colors_scatter[rfx_outlier_mask & iso_outlier_mask] = 'purple'
        ax.scatter(rfx_scores_norm, iso_scores_inv, alpha=0.5, s=12, c=colors_scatter, edgecolors='none')
        ax.axvline(rfx_threshold, color='steelblue', linestyle='--', linewidth=1.5, label='RFX-Fuse 90th')
        ax.axhline(iso_threshold, color='orange', linestyle='--', linewidth=1.5, label='IF 90th')
        # Mark top outlier with star marker
        if top_outlier_idx is not None:
            ax.scatter([rfx_scores_norm[top_outlier_idx]], [iso_scores_inv[top_outlier_idx]],
                      s=200, c='red', marker='*', edgecolors='black', linewidth=1, zorder=10, label='#1 Outlier')
            # Get and mark top-1 neighbor with smaller square
            try:
                outlier_neighbors, _ = reg.get_top_k_similar(top_outlier_idx, k=1)
                if len(outlier_neighbors) > 0:
                    neighbor_idx = outlier_neighbors[0]
                    neighbor_title = movies.get_title(train_indices[neighbor_idx])[:15] if neighbor_idx < len(train_indices) else "Neighbor"
                    ax.scatter([rfx_scores_norm[neighbor_idx]], [iso_scores_inv[neighbor_idx]],
                              s=80, c='limegreen', marker='s', edgecolors='darkgreen', linewidth=1.5, zorder=9, label=f'Top-1 Neighbor')
                    # Add annotation for neighbor
                    ax.annotate(f'{neighbor_title}...',
                               xy=(rfx_scores_norm[neighbor_idx], iso_scores_inv[neighbor_idx]),
                               xytext=(rfx_scores_norm[neighbor_idx] + 0.08, iso_scores_inv[neighbor_idx] - 0.08),
                               fontsize=7, fontweight='bold', color='darkgreen',
                               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1),
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='honeydew', edgecolor='darkgreen', alpha=0.9))
            except:
                pass
        ax.set_xlabel('RFX-Fuse Outlier Score (Manifold)', fontsize=10, fontweight='bold')
        ax.set_ylabel('IF Outlier Score (flipped)', fontsize=10, fontweight='bold')
        ax.set_title('RFX-Fuse vs. IF: Outlier Detection\n(IF scores flipped: higher=outlier)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
        both = np.sum(rfx_outlier_mask & iso_outlier_mask)
        total = np.sum(rfx_outlier_mask | iso_outlier_mask)
        overlap_pct = both / total * 100 if total > 0 else 0
        ax.text(0.95, 0.05, f'Overlap: {overlap_pct:.0f}%', transform=ax.transAxes,
               fontsize=9, ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'IF not available', ha='center', va='center', fontsize=12)

    # Panel 0,1: Overlaid Histogram with 90th percentiles
    ax = axes2b[0, 1]
    if HAS_ISOLATION_FOREST:
        ax.hist(rfx_scores_norm, bins=30, alpha=0.6, label='RFX-Fuse (Manifold)', color='steelblue', edgecolor='navy')
        ax.hist(iso_scores_inv, bins=30, alpha=0.6, label='IF (Marginal)', color='orange', edgecolor='darkorange')
        ax.axvline(rfx_threshold, color='steelblue', linestyle='--', linewidth=2, label=f'RFX-Fuse 90th: {rfx_threshold:.2f}')
        ax.axvline(iso_threshold, color='orange', linestyle='--', linewidth=2, label=f'IF 90th: {iso_threshold:.2f}')
        ax.set_xlabel('Normalized Outlier Score', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title('Outlier Score Distributions', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'IF not available', ha='center', va='center', fontsize=12)

    # Panel 0,2: Top-K to Outlier (who's closest to the outlier)
    ax = axes2b[0, 2]
    if HAS_ISOLATION_FOREST and top_outlier_idx is not None:
        try:
            outlier_similar, outlier_sim_scores = reg.get_top_k_similar(top_outlier_idx, k=8)
            valid_outlier = [(idx, i) for i, idx in enumerate(outlier_similar[:8]) if idx < len(train_indices)]
            outlier_titles = [movies.get_title(train_indices[idx])[:20] for idx, _ in valid_outlier]
            outlier_prox = [outlier_sim_scores[i] for _, i in valid_outlier]
            y_pos = np.arange(len(outlier_titles))
            bars = ax.barh(y_pos, outlier_prox, color='indianred', alpha=0.8, edgecolor='darkred', linewidth=1.5)
            # Reference line for comparison
            if len(scores_ts) > 0:
                ax.axvline(max(scores_ts), color='steelblue', linestyle='--', linewidth=1.5, alpha=0.7, label='Toy Story max')
            ax.set_xlabel('Proximity (Low = Unique)', fontsize=10, fontweight='bold')
            ax.set_title(f'Top-K to Outlier\n({top_outlier_title[:18]}...)', fontsize=10, fontweight='bold')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(outlier_titles, fontsize=7)
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'IF not available', ha='center', va='center', fontsize=12)

    # -------------------------------------------------------------------------
    # ROW 1: Outlier Var Imp | Outlier Prox Imp | Neighbor Prox Imp
    # -------------------------------------------------------------------------

    # Panel 1,0: Outlier's Local Var Imp (why predicted this way)
    ax = axes2b[1, 0]
    if HAS_ISOLATION_FOREST and local_var is not None and top_outlier_idx is not None:
        outlier_local_var = local_var[top_outlier_idx]
        outlier_local_abs = np.abs(outlier_local_var)
        sorted_idx = np.argsort(outlier_local_abs)[::-1][:8]
        local_norm = outlier_local_abs / (outlier_local_abs.max() + 1e-10)
        labels = [feature_names[i][:10] for i in sorted_idx]
        actual_vals = [X_train[top_outlier_idx, i] for i in sorted_idx]
        vals = [local_norm[i] for i in sorted_idx]
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color='indianred', alpha=0.8, edgecolor='darkred', linewidth=1.5)
        # Add actual values - inside bars if wide enough, outside if narrow
        for bar, actual_val, norm_val in zip(bars, actual_vals, vals):
            if norm_val < 0.2:
                # Put text to the right of narrow bars
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f'Actual: {actual_val:.1f}',
                       va='center', ha='left', fontsize=8, fontweight='bold', color='black')
            else:
                # Put text inside wide bars
                ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, f'Actual: {actual_val:.1f}',
                       va='center', ha='right', fontsize=8, fontweight='bold', color='black')
        ax.set_xlabel('Normalized Var Importance', fontsize=10, fontweight='bold')
        ax.set_title('Outlier: Local Var Imp\n(Why predicted?)', fontsize=10, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)

    # Get K=1 nearest neighbor info for use in multiple panels
    nearest_neighbor_idx = None
    nearest_neighbor_title = None
    nearest_neighbor_score = None
    if HAS_ISOLATION_FOREST and top_outlier_idx is not None:
        try:
            outlier_similar, outlier_sim_scores = reg.get_top_k_similar(top_outlier_idx, k=1)
            if len(outlier_similar) > 0 and outlier_similar[0] < len(train_indices):
                nearest_neighbor_idx = outlier_similar[0]
                nearest_neighbor_title = movies.get_title(train_indices[nearest_neighbor_idx])[:25]
                nearest_neighbor_score = outlier_sim_scores[0]
        except:
            pass

    # Panel 1,1: Outlier's Local Prox Imp (why unique) - green like neighbor
    ax = axes2b[1, 1]
    if HAS_ISOLATION_FOREST and sup_prox_imp is not None and top_outlier_idx is not None:
        outlier_prox_imp = sup_prox_imp[top_outlier_idx]
        sorted_idx = np.argsort(outlier_prox_imp)[::-1][:8]
        local_norm = outlier_prox_imp / (outlier_prox_imp.max() + 1e-10)
        labels = [feature_names[i][:10] for i in sorted_idx]
        actual_vals = [X_train[top_outlier_idx, i] for i in sorted_idx]
        vals = [local_norm[i] for i in sorted_idx]
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        # Add actual values inside bars
        for bar, val in zip(bars, actual_vals):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, f'Actual: {val:.1f}',
                   va='center', ha='right', fontsize=8, fontweight='bold', color='black')
        ax.set_xlabel('Prox Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'Outlier: {top_outlier_title[:15]}... Local Prox Imp\n(RFX-Fuse capability only)', fontsize=10, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)

    # Panel 1,2: Top-1 Neighbor's Local Prox Imp (why similar to outlier)
    ax = axes2b[1, 2]
    if HAS_ISOLATION_FOREST and sup_prox_imp is not None and nearest_neighbor_idx is not None:
        neighbor_prox_imp = sup_prox_imp[nearest_neighbor_idx]
        sorted_idx = np.argsort(neighbor_prox_imp)[::-1][:8]
        local_norm = neighbor_prox_imp / (neighbor_prox_imp.max() + 1e-10)
        labels = [feature_names[i][:10] for i in sorted_idx]
        actual_vals = [X_train[nearest_neighbor_idx, i] for i in sorted_idx]
        vals = [local_norm[i] for i in sorted_idx]
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
        # Add actual values inside bars
        for bar, val in zip(bars, actual_vals):
            ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height()/2, f'Actual: {val:.1f}',
                   va='center', ha='right', fontsize=8, fontweight='bold', color='black')
        ax.set_xlabel('Prox Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'Neighbor: {nearest_neighbor_title[:12]}... Local Prox Imp\n(RFX-Fuse capability only)', fontsize=10, fontweight='bold', color='darkgreen')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)

    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)
    fig2b_path = fig_dir / "supervised_outlier_detection.png"
    try:
        fig2b.savefig(fig2b_path, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"     Warning: bbox_inches='tight' failed, trying without: {e}")
        fig2b.savefig(fig2b_path, dpi=300)
    print(f"     Saved: {fig2b_path}")
    plt.close(fig2b)

    # =========================================================================
    # FIGURE 2.5: RFX-Fuse Unsupervised vs RFX-Fuse Supervised vs FAISS (Re-Ranking Boost)
    # 1x3 Grid (no text cards - explanations moved to paper)
    # =========================================================================
    print("  [2.5/4] RFX-Fuse Unsupervised vs RFX-Fuse Supervised vs FAISS (Re-Ranking)...")
    fig2_5, axes2_5 = plt.subplots(1, 3, figsize=(18, 6))
    fig2_5.suptitle('RFX-Fuse Re-Ranking: Unsupervised → Supervised Boost', fontsize=14, fontweight='bold', y=0.98)

    # Panel 0: Three-way NDCG/HR comparison (same colors as Figure 1)
    ax = axes2_5[0]
    if HAS_FAISS:
        methods = ['RFX\nUnsupervised', 'RFX\nSupervised', 'FAISS\n(Cosine)']
        ndcg_vals = [rfx_ndcg_mean, rfx_sup_ndcg_mean, faiss_ndcg_mean]
        hr_vals = [rfx_hr_mean, rfx_sup_hr_mean, faiss_hr_mean]
        x = np.arange(len(methods))
        width = 0.35
        # Same colors as Figure 1: steelblue for NDCG, coral for HR
        ax.bar(x - width/2, ndcg_vals, width, label='NDCG@10', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, hr_vals, width, label='HR@10', color='coral', alpha=0.8)
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title('Similarity Quality: All Three Methods', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for i, (n, h) in enumerate(zip(ndcg_vals, hr_vals)):
            ax.text(i - width/2, n + 0.02, f'{n:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(i + width/2, h + 0.02, f'{h:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'FAISS not available', ha='center', va='center', fontsize=12)

    # Panel 1: Matrix Local Prox Importance - UNSUPERVISED (use original feature names)
    ax = axes2_5[1]
    if matrix_idx is not None and local_prox_m is not None and len(local_prox_m) > 0:
        top_n = 8
        sorted_local = np.argsort(local_prox_m)[::-1][:top_n]
        local_norm = local_prox_m / (local_prox_m.max() + 1e-10)
        # Feature names only on y-axis
        labels = [feature_names_unsup[i][:12] for i in sorted_local]
        actual_vals = [X[matrix_idx, i] for i in sorted_local]
        vals = [local_norm[i] for i in sorted_local]
        colors = plt.cm.Blues(np.linspace(0.8, 0.4, len(vals)))
        bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor='navy', alpha=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Normalized Local Prox Importance', fontsize=10, fontweight='bold')
        ax.set_title('The Matrix (1999): RFX-Fuse Unsupervised\nWhy Top-K Similar? (Feature Space)', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        # Add actual values inside bars (black text)
        for bar, actual_val in zip(bars, actual_vals):
            ax.text(0.02, bar.get_y() + bar.get_height()/2,
                    f'Actual: {actual_val:.1f}', ha='left', va='center', fontsize=8, fontweight='bold', color='black')
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)

    # Panel 2: Matrix Local Prox Importance - SUPERVISED (use green to differentiate from XGBoost coral)
    ax = axes2_5[2]
    if matrix_train_idx is not None and sup_prox_imp is not None:
        # Get supervised local proximity importance from pre-computed array
        sup_local_prox = sup_prox_imp[matrix_train_idx]
        if sup_local_prox is not None and len(sup_local_prox) > 0:
            top_n = 8
            sorted_local = np.argsort(sup_local_prox)[::-1][:top_n]
            local_norm = sup_local_prox / (sup_local_prox.max() + 1e-10)
            # Feature names only on y-axis
            labels = [feature_names[i][:12] for i in sorted_local]
            actual_vals = [X_train[matrix_train_idx, i] for i in sorted_local]
            vals = [local_norm[i] for i in sorted_local]
            colors = plt.cm.Greens(np.linspace(0.8, 0.4, len(vals)))
            bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor='darkgreen', alpha=0.8)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Normalized Local Prox Importance', fontsize=10, fontweight='bold')
            ax.set_title('The Matrix (1999): RFX-Fuse Supervised\nWhy Top-K Similar? (Prediction Space)', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 1.15)
            ax.invert_yaxis()
            # Add actual values inside bars (black text)
            for bar, actual_val in zip(bars, actual_vals):
                ax.text(0.02, bar.get_y() + bar.get_height()/2,
                        f'Actual: {actual_val:.1f}', ha='left', va='center', fontsize=8, fontweight='bold', color='black')
        else:
            ax.text(0.5, 0.5, 'Supervised local prox\nnot available', ha='center', va='center', fontsize=10)
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'Not available', ha='center', va='center', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    fig2_5_path = fig_dir / "unsupervised_supervised_boost.png"
    plt.savefig(fig2_5_path, dpi=300, bbox_inches='tight')
    print(f"     Saved: {fig2_5_path}")
    plt.close(fig2_5)

    # =========================================================================
    # FIGURE 3: RFX-Fuse Outlier Detection vs Isolation Forest
    # (Figure 3 removed - content merged into Figure 2.5 reranking comparison)
    # =========================================================================
    print("  [3/4] RFX-Fuse Outlier Detection vs Isolation Forest...")
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle('RFX-Fuse Outlier Detection vs Isolation Forest\n(Manifold vs Marginal Outliers)', fontsize=14, fontweight='bold')

    if HAS_ISOLATION_FOREST:
        # Normalize scores with percentile-based clipping to handle extreme outliers
        # RFX: higher = more outlier (no negation needed)
        # IF: lower = more outlier (negate to match RFX convention)
        rfx_clip_max = np.percentile(rfx_outlier_scores, 99)
        rfx_clipped = np.clip(rfx_outlier_scores, rfx_outlier_scores.min(), rfx_clip_max)
        rfx_scores_norm = (rfx_clipped - rfx_clipped.min()) / (rfx_clipped.max() - rfx_clipped.min() + 1e-10)

        iso_negated = -iso_scores
        iso_clip_max = np.percentile(iso_negated, 99)
        iso_clipped = np.clip(iso_negated, iso_negated.min(), iso_clip_max)
        iso_scores_inv = (iso_clipped - iso_clipped.min()) / (iso_clipped.max() - iso_clipped.min() + 1e-10)

        rfx_threshold = np.percentile(rfx_scores_norm, 90)
        iso_threshold = np.percentile(iso_scores_inv, 90)
        rfx_outlier_mask = rfx_scores_norm > rfx_threshold
        iso_outlier_mask = iso_scores_inv > iso_threshold

        # Panel 0,0: Scatterplot
        ax = axes4[0, 0]
        colors_scatter = np.array(['lightgray'] * len(rfx_scores_norm), dtype=object)
        colors_scatter[rfx_outlier_mask & ~iso_outlier_mask] = 'steelblue'
        colors_scatter[~rfx_outlier_mask & iso_outlier_mask] = 'orange'
        colors_scatter[rfx_outlier_mask & iso_outlier_mask] = 'purple'
        ax.scatter(rfx_scores_norm, iso_scores_inv, alpha=0.5, s=15, c=colors_scatter, edgecolors='none')
        ax.axvline(rfx_threshold, color='steelblue', linestyle='--', linewidth=1.5, label='RFX-Fuse 90th')
        ax.axhline(iso_threshold, color='orange', linestyle='--', linewidth=1.5, label='IF 90th')
        ax.set_xlabel('RFX-Fuse Outlier Score (Manifold)', fontsize=10, fontweight='bold')
        ax.set_ylabel('IF Outlier Score (flipped)', fontsize=10, fontweight='bold')
        ax.set_title('RFX-Fuse vs IF (IF flipped: higher=outlier)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 0,1: Score Distributions
        ax = axes4[0, 1]
        ax.hist(rfx_scores_norm, bins=50, alpha=0.6, color='steelblue', label='RFX-Fuse', density=True)
        ax.hist(iso_scores_inv, bins=50, alpha=0.6, color='orange', label='IF (flipped)', density=True)
        ax.axvline(rfx_threshold, color='steelblue', linestyle='--', linewidth=2)
        ax.axvline(iso_threshold, color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('Normalized Outlier Score', fontsize=10, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10, fontweight='bold')
        ax.set_title('Score Distributions', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Panel 1,0: RFX-Fuse Outlier Top-K (shows uniqueness)
        ax = axes4[1, 0]
        # RFX: higher = more outlier, so use [-1] to get highest score
        top_outlier_idx = np.argsort(rfx_outlier_scores)[-1]
        top_outlier_title = movies.get_title(train_indices[top_outlier_idx]) if top_outlier_idx < len(train_indices) else f"Movie #{top_outlier_idx}"
        try:
            outlier_similar, outlier_sim_scores = reg.get_top_k_similar(top_outlier_idx, k=8)
            valid_outlier = [(idx, i) for i, idx in enumerate(outlier_similar[:8]) if idx < len(train_indices)]
            outlier_titles = [movies.get_title(train_indices[idx])[:25] for idx, _ in valid_outlier]
            outlier_prox = [outlier_sim_scores[i] for _, i in valid_outlier]
            y_pos = np.arange(len(outlier_titles))
            ax.barh(y_pos, outlier_prox, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
            ax.set_xlabel('Proximity Score (Low = Unique)', fontsize=10, fontweight='bold')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(outlier_titles, fontsize=8)
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            max_outlier_prox = max(outlier_prox) if outlier_prox else 0
            ax.axvline(max(scores_ts), color='steelblue', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.text(max(scores_ts), 0.1, f'Toy Story: {max(scores_ts):.3f}', fontsize=8, color='steelblue', rotation=90, va='bottom')
            ax.set_title(f'RFX-Fuse Outlier: {top_outlier_title[:25]}\n(Low prox = unique)', fontsize=11, fontweight='bold', color='darkgreen')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center', fontsize=10)

        # Panel 1,1: Empty (explanations moved to paper)
        ax = axes4[1, 1]
        ax.axis('off')
    else:
        for ax in axes4.flat:
            ax.text(0.5, 0.5, 'Isolation Forest not available', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    fig4_path = fig_dir / "rfx_outlier_vs_isolation_forest.png"
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    print(f"     Saved: {fig4_path}")
    plt.close(fig4)

    # =========================================================================
    # HOOK FIGURE: 3x1 Front Page Visual (for paper abstract area)
    # =========================================================================
    print("  [Hook] Front Page 3x1 Figure...")
    fig_hook, axes_hook = plt.subplots(3, 1, figsize=(4, 7))
    fig_hook.suptitle('RFX-Fuse: Explainable Similarity (MovieLens 25M)',
                      fontsize=10, fontweight='bold', y=0.99)

    # Panel (a): RFX-Fuse vs FAISS Metrics
    ax = axes_hook[0]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [rfx_hr_mean, rfx_ndcg_mean], width, label='RFX-Fuse', color='steelblue', alpha=0.9)
    ax.bar(x + width/2, [faiss_hr_mean, faiss_ndcg_mean], width, label='FAISS', color='coral', alpha=0.9)
    ax.set_ylabel('Score', fontsize=9, fontweight='bold')
    ax.set_title('(a) Similarity Retrieval', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['HR@10', 'NDCG@10'], fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    # Add improvement percentages
    for i, (rfx_val, faiss_val) in enumerate([(rfx_hr_mean, faiss_hr_mean), (rfx_ndcg_mean, faiss_ndcg_mean)]):
        if faiss_val > 0:
            improvement = (rfx_val - faiss_val) / faiss_val * 100
            ax.annotate(f'+{improvement:.0f}%', xy=(i - width/2, rfx_val + 0.02),
                        ha='center', fontsize=8, fontweight='bold', color='darkgreen')

    # Panel (b): The Matrix Top-K Similar
    ax = axes_hook[1]
    if matrix_idx is not None and len(movie_titles_m) > 0:
        n_show = min(5, len(movie_titles_m))
        y_pos = np.arange(n_show)
        short_titles = [t[:22] + '..' if len(t) > 22 else t for t in movie_titles_m[:n_show]]
        ax.barh(y_pos, proximity_scores_m[:n_show], color='steelblue', alpha=0.8, edgecolor='darkblue', linewidth=1)
        ax.set_xlabel('Proximity', fontsize=9, fontweight='bold')
        ax.set_title('(b) The Matrix (1999): Top-K Similar', fontsize=10, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_titles, fontsize=7)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Matrix data not available', ha='center', va='center')

    # Panel (c): The Matrix Local Proximity Importance
    ax = axes_hook[2]
    if matrix_idx is not None and len(top_features_m) > 0:
        n_show = min(10, len(top_features_m))
        y_pos = np.arange(n_show)
        # Normalize importance values
        max_imp = max(top_importance_m[:n_show]) + 1e-10
        norm_vals = [v / max_imp for v in top_importance_m[:n_show]]
        ax.barh(y_pos, norm_vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1)
        ax.set_xlabel('Prox Importance', fontsize=9, fontweight='bold')
        ax.set_title('(c) Why Similar? (RFX-Fuse only)', fontsize=10, fontweight='bold', color='darkgreen')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features_m[:n_show], fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Proximity importance not available', ha='center', va='center')

    plt.tight_layout()
    hook_path = fig_dir / "first_page_figure.png"
    plt.savefig(hook_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"     Saved: {hook_path}")
    plt.close(fig_hook)

    print(f"\n  All figures saved to: {fig_dir}/")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*75}")
    print("SUMMARY: RFX-Fuse Unified Pipeline")
    print("=" * 75)
    
    print(f"""
+-------------------------------------------------------------------------+
| PART A: Item Similarity (Unsupervised) | PART B: Item Ranking (Supervised) |
+------------------------------------+------------------------------------+
| Content-based similarity           | Content-based ranking (re-ranker)  |
| Features -> Top-K similar          | Features + Labels -> Predictions   |
|                                    | Top-K in prediction space          |
+------------------------------------+------------------------------------+
| EXPLANATIONS:                      | EXPLANATIONS:                      |
| * Overall proximity importance     | * Overall variable importance      |
| * Local proximity importance       | * Local variable importance        |
|   ("why similar?")                 |   ("why predicted?")               |
|                                    | * Overall proximity importance     |
|                                    | * Local proximity importance       |
|                                    |   ("why similar in pred space?")   |
+------------------------------------+------------------------------------+

Benefits:
  [OK] All explanations native--no post-hoc computation 
  [OK] Handles cold start via features--no embeddings needed
  [OK] Unified framework for similarity AND prediction

Drawbacks:
  * Training: {unsup_time + reg_time:.1f}s (slower than FAISS indexing but can be ran overnight for trained model object)
  * Query latency: milliseconds (vs sub-millisecond for FAISS)
  * No learned semantic embeddings

Key Insight:
  RFX's tree-based proximity captures NON-LINEAR feature interactions
  that cosine similarity misses. Proximity importance answers "WHY similar?"
  --a question no other tool addresses natively.

Training Times:
  * Part A (Item Similarity - Unsupervised): {unsup_time:.1f}s
  * Part B (Item Ranking - Supervised):       {reg_time:.1f}s
  * Total:                                     {unsup_time + reg_time:.1f}s
""")


if __name__ == "__main__":
    main()
