#!/usr/bin/env python3
"""
Use Case 5: Anomaly Detection (Breiman-Cutler Original Method)

Train RFX Unsupervised on CLEAN data, then test if injected anomalies
are recognized as "synthetic" (not following real data distribution).

This implements Breiman & Cutler's original vision: samples that don't
fit the learned data manifold are outliers.

Generates: usecase5_anomaly_detection.png (2x3 grid)
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths relative to this file
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'python'))

import rfx

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def load_credit_data():
    """Load clean credit data."""
    data_dir = PROJECT_ROOT / "data" / "credit"
    kaggle_file = data_dir / "train.csv"

    if not kaggle_file.exists():
        raise FileNotFoundError(
            f"Credit data not found at {kaggle_file}\n"
            "Please download from Kaggle and place in data/credit/"
        )

    print(f"Loading credit data from {kaggle_file}...")
    df = pd.read_csv(kaggle_file, low_memory=False)

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if 'id' not in c.lower()][:10]

    # Get clean samples
    df_clean = df[numeric_cols].dropna()
    X = df_clean.values.astype(np.float32)

    print(f"  Loaded {len(X)} clean samples with {len(numeric_cols)} features")

    return X, numeric_cols


def inject_anomalies(X, n_anomalies, anomaly_type='mixed', seed=42):
    """Inject anomalies into data."""
    np.random.seed(seed)
    n_samples, n_features = X.shape

    indices = np.random.choice(n_samples, n_anomalies, replace=False)
    X_anomalies = X[indices].copy()

    if anomaly_type == 'point':
        for i in range(n_anomalies):
            n_corrupt = np.random.randint(1, 4)
            corrupt_features = np.random.choice(n_features, n_corrupt, replace=False)
            for f in corrupt_features:
                std = X[:, f].std()
                mean = X[:, f].mean()
                X_anomalies[i, f] = mean + np.random.choice([-1, 1]) * np.random.uniform(5, 10) * std

    elif anomaly_type == 'contextual':
        for i in range(n_anomalies):
            n_shuffle = np.random.randint(3, n_features)
            shuffle_features = np.random.choice(n_features, n_shuffle, replace=False)
            random_samples = np.random.choice(n_samples, n_shuffle, replace=True)
            for j, f in enumerate(shuffle_features):
                X_anomalies[i, f] = X[random_samples[j], f]

    elif anomaly_type == 'collective':
        shift = np.random.uniform(2, 4, n_features) * np.sign(np.random.randn(n_features))
        X_anomalies = X_anomalies + shift * X.std(axis=0)

    else:  # mixed
        n_per_type = n_anomalies // 3
        X_point = inject_anomalies(X, n_per_type, 'point', seed)
        X_context = inject_anomalies(X, n_per_type, 'contextual', seed+1)
        X_collective = inject_anomalies(X, n_anomalies - 2*n_per_type, 'collective', seed+2)
        X_anomalies = np.vstack([X_point, X_context, X_collective])

    return X_anomalies


def main():
    print("=" * 80)
    print("USE CASE 5: ANOMALY DETECTION (BREIMAN-CUTLER METHOD)")
    print("=" * 80)
    print()

    # =========================================================================
    # Load and Prepare Data
    # =========================================================================
    X_full, feature_names = load_credit_data()

    n_train = 15000
    n_test = 5000

    np.random.seed(77)
    indices = np.random.permutation(len(X_full))
    X_train = X_full[indices[:n_train]]
    X_test_base = X_full[indices[n_train:n_train + n_test]]

    print(f"Training set (CLEAN): {n_train} samples")
    print(f"Test base: {n_test} samples")

    # =========================================================================
    # Train RFX Unsupervised on Clean Data (with proximity & local importance)
    # =========================================================================
    print("\nTraining RFX-Fuse Unsupervised on CLEAN data...")

    # Set seed for reproducibility
    np.random.seed(77)

    model = rfx.RandomForestUnsupervised(
        ntree=100,
        use_gpu=True,
        compute_importance=True,
        compute_proximity=True,  # Enable for Top-K similar
        compute_proximity_importance=True,  # Enable for proximity importance
        compute_local_importance=True,  # Enable for local importance
        compute_leaf_assignments=True,
        show_progress=True,
        iseed=77,  # RFX internal seed
    )

    start = time.time()
    model.fit(X_train)
    train_time = time.time() - start
    print(f"Trained in {train_time:.1f}s")

    # Save model
    model_path = SCRIPT_DIR / "anomaly_detection_unsupervised.rfx"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")

    # =========================================================================
    # Create Test Set with Anomalies
    # =========================================================================
    X_clean = X_test_base[:2500]
    y_clean = np.zeros(len(X_clean))

    n_anomalies = 2500
    X_anomaly = inject_anomalies(X_test_base[2500:], n_anomalies, 'mixed')
    y_anomaly = np.ones(n_anomalies)

    X_test = np.vstack([X_clean, X_anomaly]).astype(np.float32)
    y_true = np.concatenate([y_clean, y_anomaly])

    print(f"\nTest set: {len(X_clean)} clean + {n_anomalies} anomalies")

    # =========================================================================
    # Score with RFX
    # =========================================================================
    print("Scoring with RFX-Fuse...")
    proba = model.predict_proba(X_test)
    p_synthetic = proba[:, 0] if proba.shape[1] == 2 else 1 - proba[:, 0]

    clean_scores = p_synthetic[y_true == 0]
    anomaly_scores = p_synthetic[y_true == 1]

    # =========================================================================
    # Score with Isolation Forest for comparison
    # =========================================================================
    if HAS_SKLEARN:
        print("Scoring with Isolation Forest...")
        iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)  # 'auto' for fair comparison
        iso_forest.fit(X_train)
        iso_scores_raw = -iso_forest.score_samples(X_test)
        iso_scores = (iso_scores_raw - iso_scores_raw.min()) / (iso_scores_raw.max() - iso_scores_raw.min())

    # =========================================================================
    # Get local importance for training data
    # =========================================================================
    print("Getting local importance...")
    try:
        local_var_imp = model.get_local_importance()
        print(f"  Local variable importance shape: {local_var_imp.shape}")
    except Exception as e:
        print(f"  Local variable importance not available: {e}")
        local_var_imp = None

    try:
        local_prox_imp = model.get_proximity_importance()
        print(f"  Local proximity importance shape: {local_prox_imp.shape}")
    except Exception as e:
        print(f"  Local proximity importance not available: {e}")
        local_prox_imp = None

    # =========================================================================
    # Find outlier in TRAINING data (from top 10% of outlier scores)
    # =========================================================================
    train_proba = model.predict_proba(X_train)
    train_p_synthetic = train_proba[:, 0] if train_proba.shape[1] == 2 else 1 - train_proba[:, 0]

    # Get top 10% outliers and pick one randomly (not the absolute top which may be too extreme)
    sorted_indices = np.argsort(train_p_synthetic)[::-1]  # Highest first
    top_10_pct = int(len(sorted_indices) * 0.10)
    top_outlier_candidates = sorted_indices[:top_10_pct]

    # Pick a random one from top 10% (use seed for reproducibility)
    np.random.seed(123)
    top_outlier_idx = np.random.choice(top_outlier_candidates)
    top_outlier_score = train_p_synthetic[top_outlier_idx]
    print(f"\nSelected outlier (from top 10%): Sample #{top_outlier_idx}, P(synthetic)={top_outlier_score:.4f}")

    # Debug: Print proximity and importance values for top outlier
    print(f"\nDEBUG - Selected outlier analysis:")
    print(f"  Training set size: {len(X_train)}")
    if local_var_imp is not None:
        outlier_var = local_var_imp[top_outlier_idx]
        print(f"  Local var imp for outlier: min={outlier_var.min():.6f}, max={outlier_var.max():.6f}, sum={outlier_var.sum():.6f}")
        print(f"  Local var imp values: {outlier_var}")
    if local_prox_imp is not None:
        outlier_prox = local_prox_imp[top_outlier_idx]
        print(f"  Local prox imp for outlier: min={outlier_prox.min():.6f}, max={outlier_prox.max():.6f}, sum={outlier_prox.sum():.6f}")
        print(f"  Local prox imp values: {outlier_prox}")
    try:
        # Use get_top_k_similar API (same as unified pipeline)
        # Request more to filter out synthetic samples (indices >= n_train)
        similar_idx_raw, similar_scores_raw = model.get_top_k_similar(top_outlier_idx, 20)
        # Filter to only original samples (not synthetic)
        valid_pairs = [(idx, score) for idx, score in zip(similar_idx_raw, similar_scores_raw)
                       if idx < n_train][:5]
        print(f"  Top 5 similar (via get_top_k_similar, filtered): indices={[p[0] for p in valid_pairs]}")
        print(f"  Top 5 similarity scores: {[p[1] for p in valid_pairs]}")
    except Exception as e:
        print(f"  Error getting top-k similar: {e}")

    # =========================================================================
    # Generate 2x3 Figure
    # =========================================================================
    print("\nGenerating figure...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Use Case 5: Anomaly Detection (Breiman-Cutler Method)\nTrain on CLEAN data → Anomalies look "synthetic"',
                 fontsize=14, fontweight='bold', y=1.02)

    # -------------------------------------------------------------------------
    # Panel (a): Score Distributions
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    ax.hist(clean_scores, bins=40, alpha=0.7, color='steelblue', label=f'Clean (n={len(clean_scores)})', density=True)
    ax.hist(anomaly_scores, bins=40, alpha=0.7, color='coral', label=f'Anomaly (n={len(anomaly_scores)})', density=True)
    ax.axvline(clean_scores.mean(), color='steelblue', linestyle='--', linewidth=2, label=f'Clean mean: {clean_scores.mean():.3f}')
    ax.axvline(anomaly_scores.mean(), color='coral', linestyle='--', linewidth=2, label=f'Anomaly mean: {anomaly_scores.mean():.3f}')
    ax.set_xlabel('P(synthetic) Score', fontsize=10, fontweight='bold')
    ax.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax.set_title('RFX-Fuse: Clean vs Anomaly Distributions', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel (b): ROC Curves - RFX vs IF
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    if HAS_SKLEARN:
        rfx_auc = roc_auc_score(y_true, p_synthetic)
        iso_auc = roc_auc_score(y_true, iso_scores)

        fpr_rfx, tpr_rfx, _ = roc_curve(y_true, p_synthetic)
        fpr_iso, tpr_iso, _ = roc_curve(y_true, iso_scores)

        ax.plot(fpr_rfx, tpr_rfx, color='steelblue', linewidth=2, label=f'RFX-Fuse (AUC={rfx_auc:.3f})')
        ax.plot(fpr_iso, tpr_iso, color='orange', linewidth=2, label=f'Isolation Forest (AUC={iso_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax.set_title('ROC Curve: RFX-Fuse vs Isolation Forest', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'sklearn not available', ha='center', va='center')

    # -------------------------------------------------------------------------
    # Panel (c): Overall Feature Importance for Anomaly Detection
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    importance = model.feature_importances_()
    sorted_idx = np.argsort(importance)[::-1][:8]
    labels = [feature_names[i][:15] if i < len(feature_names) else f'F{i}' for i in sorted_idx]
    vals = [importance[i] for i in sorted_idx]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax.set_title('Overall Feature Importance\n(Which features detect anomalies?)', fontsize=11, fontweight='bold', color='darkgreen')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    # -------------------------------------------------------------------------
    # Panel (d): Outlier - Local Variable Importance
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    if local_var_imp is not None:
        outlier_var_imp = local_var_imp[top_outlier_idx]
        outlier_var_abs = np.abs(outlier_var_imp)
        sorted_idx = np.argsort(outlier_var_abs)[::-1][:8]
        labels = [feature_names[i][:12] if i < len(feature_names) else f'F{i}' for i in sorted_idx]

        # Use raw values if max is too small, otherwise normalize
        max_val = outlier_var_abs.max()
        if max_val > 0.01:
            vals = [outlier_var_abs[i] / max_val for i in sorted_idx]
        else:
            vals = [outlier_var_abs[i] for i in sorted_idx]

        actual_vals = [X_train[top_outlier_idx, i] for i in sorted_idx]
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, vals, color='indianred', alpha=0.8, edgecolor='darkred', linewidth=1.5)

        # Add actual values - position based on bar width
        max_bar_width = max(vals) if max(vals) > 0 else 1
        for bar, actual_val in zip(bars, actual_vals):
            if actual_val == int(actual_val):
                label = f'Actual: {int(actual_val)}'
            else:
                label = f'Actual: {actual_val:.2f}'
            # If bar is too small, position label outside; otherwise inside
            if bar.get_width() < max_bar_width * 0.3:
                ax.text(bar.get_width() + max_bar_width * 0.02, bar.get_y() + bar.get_height()/2,
                       label, va='center', ha='left', fontsize=8, fontweight='bold', color='black')
            else:
                ax.text(bar.get_width() - max_bar_width * 0.02, bar.get_y() + bar.get_height()/2,
                       label, va='center', ha='right', fontsize=8, fontweight='bold', color='black')

        ax.set_xlabel('Normalized Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'Outlier #{top_outlier_idx}: Local Variable Importance\n(Why predicted as outlier? - RFX-Fuse unique)',
                     fontsize=11, fontweight='bold', color='darkred')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Local var imp not available', ha='center', va='center', fontsize=10)

    # -------------------------------------------------------------------------
    # Panel (e): Outlier - Top-K Similar
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    try:
        # Use get_top_k_similar() method (same as unified pipeline)
        # Request more to filter out synthetic samples (indices >= n_train)
        similar_idx_raw, similar_scores_raw = model.get_top_k_similar(top_outlier_idx, 30)

        # Filter to only original samples (not synthetic) - indices < n_train
        valid_pairs = [(idx, score) for idx, score in zip(similar_idx_raw, similar_scores_raw)
                       if idx < n_train]

        # Take top 8 valid
        valid_pairs = valid_pairs[:8]
        similar_idx = np.array([p[0] for p in valid_pairs])
        similar_scores = np.array([p[1] for p in valid_pairs])

        # Get P(synthetic) for each similar sample
        similar_p_synth = [train_p_synthetic[i] for i in similar_idx]
        y_pos = np.arange(len(similar_idx))
        colors = ['coral' if p > 0.5 else 'steelblue' for p in similar_p_synth]
        bars = ax.barh(y_pos, similar_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Add proximity score labels on bars
        max_score = max(similar_scores) if len(similar_scores) > 0 and max(similar_scores) > 0 else 1
        for bar, score in zip(bars, similar_scores):
            if bar.get_width() < max_score * 0.3:
                ax.text(bar.get_width() + max_score * 0.02, bar.get_y() + bar.get_height()/2,
                       f'Prox={score:.3f}', va='center', ha='left', fontsize=8, fontweight='bold', color='black')
            else:
                ax.text(bar.get_width() - max_score * 0.02, bar.get_y() + bar.get_height()/2,
                       f'Prox={score:.3f}', va='center', ha='right', fontsize=8, fontweight='bold', color='black')

        ax.set_xlabel('Proximity Score', fontsize=10, fontweight='bold')
        ax.set_title(f'Outlier #{top_outlier_idx}: Top-K Similar\n(P(synth)={top_outlier_score:.3f} - RFX-Fuse unique)',
                     fontsize=11, fontweight='bold', color='darkred')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'#{i} P(synth)={p:.2f}' for i, p in zip(similar_idx, similar_p_synth)], fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='coral', label='P(synth)>0.5'),
                          Patch(facecolor='steelblue', label='P(synth)<0.5')]
        ax.legend(handles=legend_elements, fontsize=8, loc='lower right')
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:40]}', ha='center', va='center', fontsize=10)

    # -------------------------------------------------------------------------
    # Panel (f): Outlier - Local Proximity Importance
    # -------------------------------------------------------------------------
    ax = axes[1, 2]
    if local_prox_imp is not None:
        outlier_prox_imp = local_prox_imp[top_outlier_idx]
        sorted_idx = np.argsort(outlier_prox_imp)[::-1][:8]
        labels = [feature_names[i][:12] if i < len(feature_names) else f'F{i}' for i in sorted_idx]

        # Use raw values if max is too small, otherwise normalize
        max_val = outlier_prox_imp.max()
        if max_val > 0.01:
            vals = [outlier_prox_imp[i] / max_val for i in sorted_idx]
        else:
            vals = [outlier_prox_imp[i] for i in sorted_idx]

        actual_vals = [X_train[top_outlier_idx, i] for i in sorted_idx]
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, vals, color='mediumseagreen', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)

        # Add actual values - position based on bar width
        max_bar_width = max(vals) if max(vals) > 0 else 1
        for bar, actual_val in zip(bars, actual_vals):
            if actual_val == int(actual_val):
                label = f'Actual: {int(actual_val)}'
            else:
                label = f'Actual: {actual_val:.2f}'
            # If bar is too small, position label outside; otherwise inside
            if bar.get_width() < max_bar_width * 0.3:
                ax.text(bar.get_width() + max_bar_width * 0.02, bar.get_y() + bar.get_height()/2,
                       label, va='center', ha='left', fontsize=8, fontweight='bold', color='black')
            else:
                ax.text(bar.get_width() - max_bar_width * 0.02, bar.get_y() + bar.get_height()/2,
                       label, va='center', ha='right', fontsize=8, fontweight='bold', color='black')

        ax.set_xlabel('Proximity Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'Outlier #{top_outlier_idx}: Local Prox Importance\n(Why unique? - RFX-Fuse unique)',
                     fontsize=11, fontweight='bold', color='darkgreen')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Local prox imp not available', ha='center', va='center', fontsize=10)

    plt.tight_layout()

    # Save
    fig_path = SCRIPT_DIR / "anomaly_detection.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()

    # =========================================================================
    # Print Summary (for log / paper updates)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: USE CASE 5 RESULTS (FOR PAPER)")
    print("=" * 80)

    print("\n--- Panel (a): Score Distributions ---")
    print(f"Clean P(synthetic):   μ={clean_scores.mean():.4f}, σ={clean_scores.std():.4f}")
    print(f"Anomaly P(synthetic): μ={anomaly_scores.mean():.4f}, σ={anomaly_scores.std():.4f}")
    print(f"Separation:           {anomaly_scores.mean() - clean_scores.mean():.4f}")

    print("\n--- Panel (b): ROC Curves ---")
    if HAS_SKLEARN:
        print(f"RFX-Fuse AUC:         {rfx_auc:.4f}")
        print(f"Isolation Forest AUC: {iso_auc:.4f}")
        print(f"Difference:           {rfx_auc - iso_auc:+.4f} (RFX-Fuse {'better' if rfx_auc > iso_auc else 'worse'})")

    print("\n--- Panel (c): Overall Feature Importance ---")
    importance = model.feature_importances_()
    sorted_idx = np.argsort(importance)[::-1][:8]
    for rank, i in enumerate(sorted_idx, 1):
        fname = feature_names[i] if i < len(feature_names) else f'F{i}'
        print(f"  {rank}. {fname:<20} {importance[i]:.4f}")

    print(f"\n--- Panel (d): Outlier #{top_outlier_idx} Local Variable Importance ---")
    print(f"(Why predicted as anomaly?)")
    if local_var_imp is not None:
        outlier_var_imp = local_var_imp[top_outlier_idx]
        outlier_var_abs = np.abs(outlier_var_imp)
        sorted_idx_var = np.argsort(outlier_var_abs)[::-1][:8]
        for rank, i in enumerate(sorted_idx_var, 1):
            fname = feature_names[i] if i < len(feature_names) else f'F{i}'
            actual = X_train[top_outlier_idx, i]
            print(f"  {rank}. {fname:<20} imp={outlier_var_abs[i]:.6f}, actual={actual:.2f}")

    print(f"\n--- Panel (e): Outlier #{top_outlier_idx} Top-K Similar ---")
    print(f"(Why unique? - Low scores = truly isolated in manifold space)")
    try:
        similar_idx_raw, similar_scores_raw = model.get_top_k_similar(top_outlier_idx, 30)
        valid_pairs = [(idx, score) for idx, score in zip(similar_idx_raw, similar_scores_raw)
                       if idx < n_train][:8]
        for rank, (idx, score) in enumerate(valid_pairs, 1):
            p_synth = train_p_synthetic[idx]
            status = "outlier" if p_synth > 0.5 else "normal"
            print(f"  {rank}. Sample #{idx:<6} prox={score:.4f}, P(synth)={p_synth:.3f} ({status})")
        avg_prox = np.mean([p[1] for p in valid_pairs])
        print(f"  Average proximity to neighbors: {avg_prox:.4f} (low = isolated)")
    except Exception as e:
        print(f"  Error: {e}")

    print(f"\n--- Panel (f): Outlier #{top_outlier_idx} Local Proximity Importance ---")
    print(f"(What makes it unique? - Features shared with neighbors)")
    if local_prox_imp is not None:
        outlier_prox_imp = local_prox_imp[top_outlier_idx]
        sorted_idx_prox = np.argsort(outlier_prox_imp)[::-1][:8]
        for rank, i in enumerate(sorted_idx_prox, 1):
            fname = feature_names[i] if i < len(feature_names) else f'F{i}'
            actual = X_train[top_outlier_idx, i]
            print(f"  {rank}. {fname:<20} prox_imp={outlier_prox_imp[i]:.6f}, actual={actual:.2f}")

    print("\n--- Key Insights for Paper ---")
    print(f"• Breiman-Cutler: Train on clean → anomalies have high P(synthetic)")
    print(f"• Clean samples: P(syn)={clean_scores.mean():.2f}; Anomalies: P(syn)={anomaly_scores.mean():.2f}")
    if HAS_SKLEARN:
        print(f"• RFX-Fuse AUC={rfx_auc:.2f} vs IF AUC={iso_auc:.2f}")
    print(f"• Top-K for outliers shows LOW proximity (isolated in manifold space)")
    print(f"• Local var imp: why predicted as anomaly")
    print(f"• Local prox imp: what makes it unique (features shared with neighbors)")
    print()


if __name__ == '__main__':
    main()
