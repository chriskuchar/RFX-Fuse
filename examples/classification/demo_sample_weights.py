#!/usr/bin/env python3
"""
RFX-Fuse Sample Weights Demo: Weighted Bootstrap Sampling
=========================================================

Demonstrates the sample_weights parameter for controlling bootstrap draw
probability in classification and regression tasks.

Use case: In credit scoring, recent borrowers are more representative of
current lending conditions. We upweight recent samples so they appear
more frequently in bootstrap draws, while casewise weighting remains
unchanged on the backend.

Dataset: Kaggle Credit Score (100K borrowers, 15 features)
"""

import sys
import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'python'))
import RFXFuse as rfx

N_TREES = 200
USE_GPU = True
GPU_BATCH_SIZE = 50

# =========================================================================
# Data Loading (reuses cached finance data)
# =========================================================================
def load_finance_cached(max_samples=0):
    cache_dir = PROJECT_ROOT / "data" / "cache"
    cache_file = cache_dir / f"claim10_finance_s{max_samples}.npz"
    if not cache_file.exists():
        cache_file = cache_dir / "claim10_finance_s0.npz"
    if not cache_file.exists():
        print(f"ERROR: No finance cache found in {cache_dir}")
        sys.exit(1)

    data = np.load(cache_file, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int32)
    feature_names = [str(f) for f in data['feature_names']]

    if max_samples > 0 and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, feature_names


def split_data(X, y, train_frac=0.8, seed=42):
    np.random.seed(seed)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * train_frac)
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def compute_recency_weights(n_samples, decay=0.5):
    """Simulate recency: later samples are 'newer' and get higher weight."""
    t = np.linspace(0, 1, n_samples)
    weights = np.exp(decay * t)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def compute_class_balance_weights(y):
    """Upweight the minority class to address class imbalance."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = np.ones(total, dtype=np.float32)
    for cls, count in zip(classes, counts):
        weights[y == cls] = total / (len(classes) * count)
    return weights


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 75)
    print("RFX-FUSE SAMPLE WEIGHTS DEMO")
    print("Weighted Bootstrap Sampling for Classification & Regression")
    print("=" * 75)

    X, y, feature_names = load_finance_cached(max_samples=0)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"Default rate: train={y_train.mean()*100:.1f}%, test={y_test.mean()*100:.1f}%")

    # =====================================================================
    # CLASSIFICATION: Baseline vs Recency-Weighted vs Class-Balance-Weighted
    # =====================================================================
    print(f"\n{'='*75}")
    print("CLASSIFICATION: Effect of sample_weights on Bootstrap Sampling")
    print("=" * 75)

    recency_weights = compute_recency_weights(len(X_train))
    balance_weights = compute_class_balance_weights(y_train)

    configs = [
        ("Baseline (uniform)",     None),
        ("Recency-weighted",       recency_weights),
        ("Class-balance-weighted", balance_weights),
    ]

    clf_results = {}

    for name, weights in configs:
        print(f"\n--- {name} ---")
        t0 = time.time()
        clf = rfx.RandomForestClassifier(
            ntree=N_TREES, mtry=5, minndsize=3,
            use_gpu=USE_GPU, batch_size=GPU_BATCH_SIZE, iseed=42,
        )
        if weights is not None:
            clf.fit(X_train, y_train, sample_weights=weights)
        else:
            clf.fit(X_train, y_train)
        elapsed = time.time() - t0

        oob = clf.get_oob_error()
        preds = clf.predict(X_test)
        acc = np.mean(preds == y_test)

        from sklearn.metrics import precision_score, recall_score, f1_score
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        clf_results[name] = {
            'oob': oob, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'time': elapsed
        }

        print(f"  Train time:  {elapsed:.1f}s")
        print(f"  OOB error:   {oob*100:.2f}%")
        print(f"  Test acc:    {acc*100:.2f}%")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  F1:          {f1:.4f}")

        vimp = clf.feature_importances_()
        top3 = np.argsort(vimp)[::-1][:3]
        print(f"  Top-3 features: {', '.join(feature_names[i] for i in top3)}")

    # Summary table
    print(f"\n{'='*75}")
    print("CLASSIFICATION SUMMARY")
    print("=" * 75)
    print(f"  {'Config':<28s} {'OOB%':>7s} {'Acc%':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'Time':>7s}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for name, r in clf_results.items():
        print(f"  {name:<28s} {r['oob']*100:>6.2f}% {r['acc']*100:>6.2f}% {r['prec']:>7.4f} {r['rec']:>7.4f} {r['f1']:>7.4f} {r['time']:>6.1f}s")

    # =====================================================================
    # REGRESSION: Predicting Outstanding Debt with sample_weights
    # =====================================================================
    print(f"\n{'='*75}")
    print("REGRESSION: Predicting Outstanding Debt with sample_weights")
    print("=" * 75)

    debt_col = feature_names.index('Outstanding_Debt') if 'Outstanding_Debt' in feature_names else None
    if debt_col is not None:
        feature_mask = [i for i in range(len(feature_names)) if i != debt_col]
        X_reg_train = X_train[:, feature_mask].astype(np.float32)
        X_reg_test = X_test[:, feature_mask].astype(np.float32)
        y_reg_train = X_train[:, debt_col].astype(np.float32)
        y_reg_test = X_test[:, debt_col].astype(np.float32)
        reg_feature_names = [feature_names[i] for i in feature_mask]

        reg_configs = [
            ("Baseline (uniform)", None),
            ("Recency-weighted",   recency_weights),
        ]

        reg_results = {}

        for name, weights in reg_configs:
            print(f"\n--- {name} ---")
            t0 = time.time()
            reg = rfx.RandomForestRegressor(
                ntree=N_TREES, mtry=4, minndsize=5,
                use_gpu=USE_GPU, batch_size=GPU_BATCH_SIZE, iseed=42,
            )
            if weights is not None:
                reg.fit(X_reg_train, y_reg_train, sample_weights=weights)
            else:
                reg.fit(X_reg_train, y_reg_train)
            elapsed = time.time() - t0

            oob = reg.get_oob_error()
            preds = reg.predict(X_reg_test)
            mse = np.mean((preds - y_reg_test) ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - mse / np.var(y_reg_test)

            reg_results[name] = {
                'oob': oob, 'rmse': rmse, 'r2': r2, 'time': elapsed
            }

            print(f"  Train time:  {elapsed:.1f}s")
            print(f"  OOB MSE:     {oob:.4f}")
            print(f"  Test RMSE:   {rmse:,.2f}")
            print(f"  Test R^2:    {r2:.4f}")

            vimp = reg.feature_importances_()
            top3 = np.argsort(vimp)[::-1][:3]
            print(f"  Top-3 features: {', '.join(reg_feature_names[i] for i in top3)}")

        print(f"\n{'='*75}")
        print("REGRESSION SUMMARY")
        print("=" * 75)
        print(f"  {'Config':<28s} {'OOB MSE':>10s} {'RMSE':>10s} {'R^2':>8s} {'Time':>7s}")
        print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")
        for name, r in reg_results.items():
            print(f"  {name:<28s} {r['oob']:>10.4f} {r['rmse']:>10,.2f} {r['r2']:>8.4f} {r['time']:>6.1f}s")
    else:
        print("  Skipping regression: 'Outstanding_Debt' feature not found")

    # =====================================================================
    # VISUALIZATION
    # =====================================================================
    print(f"\n{'='*75}")
    print("GENERATING VISUALIZATION...")
    print("=" * 75)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('RFX-Fuse: Effect of sample_weights on Bootstrap Sampling',
                     fontsize=14, fontweight='bold')
        gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # Panel 0,0: Classification metrics comparison
        ax = fig.add_subplot(gs[0, 0])
        labels = list(clf_results.keys())
        short_labels = ['Uniform', 'Recency', 'Class-Bal']
        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, [clf_results[l]['prec'] for l in labels], w, label='Precision', color='steelblue', alpha=0.8)
        ax.bar(x,     [clf_results[l]['rec'] for l in labels],  w, label='Recall',    color='coral', alpha=0.8)
        ax.bar(x + w, [clf_results[l]['f1'] for l in labels],   w, label='F1',        color='mediumseagreen', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel('Score')
        ax.set_title('Classification Metrics by Weighting', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        # Panel 0,1: OOB error comparison
        ax2 = fig.add_subplot(gs[0, 1])
        oob_vals = [clf_results[l]['oob'] * 100 for l in labels]
        bars = ax2.bar(short_labels, oob_vals, color=['steelblue', 'coral', 'mediumseagreen'], alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, oob_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')
        ax2.set_ylabel('OOB Error (%)')
        ax2.set_title('OOB Error by Weighting', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Panel 0,2: Weight distributions
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(recency_weights, bins=50, alpha=0.6, color='coral', edgecolor='darkred', label='Recency')
        ax3.hist(balance_weights, bins=50, alpha=0.6, color='mediumseagreen', edgecolor='darkgreen', label='Class-Balance')
        ax3.axvline(1.0, color='steelblue', linestyle='--', linewidth=2, label='Uniform (1.0)')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Count')
        ax3.set_title('Weight Distributions', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # Panel 1,0: Regression R^2 comparison
        if debt_col is not None:
            ax4 = fig.add_subplot(gs[1, 0])
            reg_labels = list(reg_results.keys())
            reg_short = ['Uniform', 'Recency']
            r2_vals = [reg_results[l]['r2'] for l in reg_labels]
            bars = ax4.bar(reg_short, r2_vals, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, r2_vals):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
            ax4.set_ylabel('R^2')
            ax4.set_title('Regression R^2 by Weighting', fontweight='bold')
            ax4.grid(axis='y', alpha=0.3)

        # Panel 1,1: Regression RMSE comparison
        if debt_col is not None:
            ax5 = fig.add_subplot(gs[1, 1])
            rmse_vals = [reg_results[l]['rmse'] for l in reg_labels]
            bars = ax5.bar(reg_short, rmse_vals, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, rmse_vals):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{val:,.0f}', ha='center', fontsize=9, fontweight='bold')
            ax5.set_ylabel('RMSE')
            ax5.set_title('Regression RMSE by Weighting', fontweight='bold')
            ax5.grid(axis='y', alpha=0.3)

        # Panel 1,2: Summary text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        summary = """SAMPLE WEIGHTS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sample_weights controls BOOTSTRAP
DRAW PROBABILITY only.

Casewise weighting (win = nin * cw)
remains untouched internally.

USE CASES:
• Recency: upweight recent data
• Class balance: upweight minority
• Domain knowledge: expert priors
• Curriculum learning: easy→hard

SUPPORTED MODES:
• Classification (CPU & GPU)
• Regression (CPU & GPU)
• Unsupervised (deferred)"""
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig_path = SCRIPT_DIR / "sample_weights_comparison.png"
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"Saved visualization to: {fig_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Could not create visualization: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*75}")
    print("DONE - sample_weights verified for classification and regression")
    print("=" * 75)


if __name__ == "__main__":
    main()
