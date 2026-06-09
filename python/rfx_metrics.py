"""
rfx_metrics.py -- Threshold tuning and classification evaluation for RFX-Fuse.

Zero external dependencies beyond numpy (plotly optional for plots).

Provides:
    tune_threshold()       - Automated decision threshold optimization
    evaluate_classifier()  - Comprehensive classification metrics report
    compare_classifiers()  - Side-by-side comparison of multiple models
    bootstrap_ci()         - Bootstrap confidence intervals for any metric
    ThresholdResult        - Dataclass returned by tune_threshold (with plot_sweep)
    ClassifierEvaluation   - Dataclass returned by evaluate_classifier
                             (with plot_density, plot_roc, plot_pr, plot_calibration)
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np


# ---------------------------------------------------------------------------
# Internal pure-numpy implementations (no sklearn)
# ---------------------------------------------------------------------------

def _roc_curve(y_true, y_score):
    """Compute ROC curve (FPR, TPR, thresholds) from binary labels and scores.

    Returns (fpr, tpr, thresholds) sorted from high to low threshold.
    Includes the (0, 0) endpoint.
    """
    y_true = np.asarray(y_true, dtype=np.intp)
    y_score = np.asarray(y_score, dtype=np.float64)

    desc = np.argsort(y_score, kind='mergesort')[::-1]
    y_score_sorted = y_score[desc]
    y_true_sorted = y_true[desc]

    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_indices = np.concatenate((distinct_indices, [len(y_true) - 1]))

    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    tps = np.concatenate(([0], tps))
    fps = np.concatenate(([0], fps))

    P = float(y_true.sum())
    N = float(len(y_true)) - P

    fpr = fps / max(N, 1)
    tpr = tps / max(P, 1)
    thresholds = np.concatenate(([y_score_sorted[0] + 1], y_score_sorted[threshold_indices]))

    return fpr, tpr, thresholds


def _pr_curve(y_true, y_score):
    """Compute Precision-Recall curve from binary labels and scores.

    Returns (precision, recall, thresholds).
    precision and recall have one extra element (the start point recall=0, precision=1).
    """
    y_true = np.asarray(y_true, dtype=np.intp)
    y_score = np.asarray(y_score, dtype=np.float64)

    desc = np.argsort(y_score)[::-1]
    y_score_sorted = y_score[desc]
    y_true_sorted = y_true[desc]

    tps = np.cumsum(y_true_sorted)
    total_predicted = np.arange(1, len(y_true) + 1, dtype=np.float64)

    precision_arr = tps / total_predicted
    P = float(y_true.sum())
    recall_arr = tps / max(P, 1)

    distinct = np.concatenate((np.diff(y_score_sorted) != 0, [True]))
    distinct_idx = np.where(distinct)[0]

    precision_arr = precision_arr[distinct_idx]
    recall_arr = recall_arr[distinct_idx]
    thresholds = y_score_sorted[distinct_idx]

    precision_arr = np.concatenate(([1.0], precision_arr))
    recall_arr = np.concatenate(([0.0], recall_arr))

    return precision_arr, recall_arr, thresholds


def _auc_trapz(x, y):
    """Trapezoidal AUC for a curve that is already ordered (or monotonic in x).

    If x is not monotonically increasing, sorts by x first using a stable sort
    so that tied x-values preserve their original y-order.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.any(np.diff(x) < 0):
        order = np.argsort(x, kind='stable')
        x, y = x[order], y[order]
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(_trapz(y, x))


def _average_precision(y_true, y_score):
    """Average precision (area under PR curve via step-function sum)."""
    precision, recall, _ = _pr_curve(y_true, y_score)
    return float(np.sum(np.diff(recall) * precision[1:]))


def _roc_auc(y_true, y_score):
    """ROC AUC via trapezoidal rule on the ROC curve."""
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    return _auc_trapz(fpr, tpr)


def _ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov statistic: max |TPR - FPR| across thresholds."""
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def _brier_score(y_true, y_prob):
    """Brier score: mean squared error of probability estimates."""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true, dtype=float)) ** 2))


def _log_loss(y_true, y_prob):
    """Binary cross-entropy (log loss), clipped to avoid log(0)."""
    eps = 1e-15
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _confusion_counts(y_true, y_pred):
    """Return (tp, tn, fp, fn) as plain ints."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    tp = int(((yp == 1) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    return tp, tn, fp, fn


def _mcc(tp, tn, fp, fn):
    """Matthews Correlation Coefficient."""
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / max(denom, 1e-10)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ThresholdResult:
    """Result of threshold tuning via tune_threshold()."""
    threshold: float
    precision: float
    recall: float
    f1: float
    objective: str
    all_thresholds: np.ndarray
    all_metrics: dict

    def __repr__(self):
        return (
            f"ThresholdResult(threshold={self.threshold:.4f}, "
            f"precision={self.precision:.4f}, recall={self.recall:.4f}, "
            f"f1={self.f1:.4f}, objective='{self.objective}')"
        )

    def plot_sweep(self, show_optimal=True):
        """Interactive Plotly chart of precision/recall/F1 across all thresholds.

        Args:
            show_optimal: Draw a vertical line at the selected threshold.

        Returns:
            plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for plot_sweep(). "
                "Install it with: pip install plotly"
            )

        fig = go.Figure()
        colors = {'precision': '#636EFA', 'recall': '#EF553B', 'f1': '#00CC96',
                  'fbeta': '#AB63FA', 'mcc': '#FFA15A'}
        for name, vals in self.all_metrics.items():
            fig.add_trace(go.Scatter(
                x=self.all_thresholds, y=vals,
                mode='lines', name=name.capitalize(),
                line=dict(color=colors.get(name, '#19D3F3'), width=2),
            ))

        if show_optimal:
            fig.add_vline(
                x=self.threshold, line_dash="dash", line_color="black",
                line_width=2,
                annotation_text=f"Optimal = {self.threshold:.3f}",
                annotation_position="top right",
            )

        fig.update_layout(
            title="Threshold Sweep: Precision / Recall / F1",
            xaxis_title="Decision Threshold",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05]),
            legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
            template="plotly_white",
        )
        return fig


@dataclass
class ClassifierEvaluation:
    """Comprehensive classification evaluation from evaluate_classifier()."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    mcc: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    gini_coefficient: float
    ks_statistic: float
    confusion_matrix: np.ndarray
    roc_curve: dict
    pr_curve: dict
    threshold: float
    support: dict
    y_proba: np.ndarray
    y_true: np.ndarray

    def __repr__(self):
        return (
            f"ClassifierEvaluation(accuracy={self.accuracy:.4f}, "
            f"precision={self.precision:.4f}, recall={self.recall:.4f}, "
            f"f1={self.f1:.4f}, roc_auc={self.roc_auc:.4f}, "
            f"pr_auc={self.pr_auc:.4f}, threshold={self.threshold:.4f})"
        )

    def summary(self):
        """Return formatted summary string."""
        cm = self.confusion_matrix
        lines = [
            f"Classifier Evaluation (threshold={self.threshold:.2f})",
            "=" * 50,
            f"Accuracy:    {self.accuracy:.4f}    ROC AUC:      {self.roc_auc:.4f}",
            f"Precision:   {self.precision:.4f}    PR AUC:       {self.pr_auc:.4f}",
            f"Recall:      {self.recall:.4f}    MCC:          {self.mcc:.4f}",
            f"F1 Score:    {self.f1:.4f}    Specificity:  {self.specificity:.4f}",
            f"Brier Score: {self.brier_score:.4f}    Log Loss:     {self.log_loss:.4f}",
            f"Gini Coeff:  {self.gini_coefficient:.4f}    KS Statistic: {self.ks_statistic:.4f}",
            "",
            "Confusion Matrix:",
            "              Predicted",
            "              Neg    Pos",
            f"Actual Neg  {cm[0,0]:>5d}  {cm[0,1]:>5d}",
            f"       Pos  {cm[1,0]:>5d}  {cm[1,1]:>5d}",
            "",
            f"Support: {self.support['positive']} positive, "
            f"{self.support['negative']} negative "
            f"({self.support['total']} total)",
        ]
        return "\n".join(lines)

    def plot_density(self, pos_name="Positive", neg_name="Negative",
                     show_threshold=True, opacity=0.6, nbins=100):
        """Interactive Plotly density plot of predicted probabilities.

        Args:
            pos_name: Label for the positive class.
            neg_name: Label for the negative class.
            show_threshold: Whether to draw a vertical line at the threshold.
            opacity: Histogram opacity (0-1).
            nbins: Number of histogram bins.

        Returns:
            plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for plot_density(). "
                "Install it with: pip install plotly"
            )

        pos_mask = self.y_true == 1
        neg_mask = ~pos_mask

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=self.y_proba[neg_mask],
            name=neg_name,
            opacity=opacity,
            nbinsx=nbins,
            histnorm='probability density',
            marker_color='#636EFA',
        ))
        fig.add_trace(go.Histogram(
            x=self.y_proba[pos_mask],
            name=pos_name,
            opacity=opacity,
            nbinsx=nbins,
            histnorm='probability density',
            marker_color='#EF553B',
        ))

        if show_threshold:
            fig.add_vline(
                x=self.threshold,
                line_dash="dash",
                line_color="black",
                line_width=2,
                annotation_text=f"Threshold = {self.threshold:.2f}",
                annotation_position="top",
            )

        fig.update_layout(
            barmode='overlay',
            title="Predicted Probability Distribution by Class",
            xaxis_title="Predicted Probability",
            yaxis_title="Density",
            legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.95),
            template="plotly_white",
        )
        return fig

    def plot_roc(self, show_auc=True):
        """Interactive Plotly ROC curve.

        Args:
            show_auc: Annotate the AUC value on the plot.

        Returns:
            plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for plot_roc(). pip install plotly")

        fpr = self.roc_curve.get('fpr', np.array([]))
        tpr = self.roc_curve.get('tpr', np.array([]))
        if len(fpr) == 0:
            raise ValueError("ROC curve data is empty")

        label = f"ROC (AUC = {self.roc_auc:.4f})" if show_auc else "ROC"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=label,
            line=dict(color='#636EFA', width=2),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random',
            line=dict(color='grey', width=1, dash='dash'),
            showlegend=False,
        ))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1], constrain='domain'),
            yaxis=dict(range=[0, 1.05], scaleanchor='x'),
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
            template="plotly_white",
        )
        return fig

    def plot_pr(self, show_auc=True, show_baseline=True):
        """Interactive Plotly Precision-Recall curve.

        Args:
            show_auc: Annotate the PR AUC value on the plot.
            show_baseline: Draw the no-skill baseline (prevalence).

        Returns:
            plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for plot_pr(). pip install plotly")

        pr_prec = self.pr_curve.get('precision', np.array([]))
        pr_rec = self.pr_curve.get('recall', np.array([]))
        if len(pr_prec) == 0:
            raise ValueError("PR curve data is empty")

        label = f"PR (AUC = {self.pr_auc:.4f})" if show_auc else "PR"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pr_rec, y=pr_prec, mode='lines', name=label,
            line=dict(color='#EF553B', width=2),
        ))
        if show_baseline:
            prevalence = self.support['positive'] / max(self.support['total'], 1)
            fig.add_hline(
                y=prevalence, line_dash="dash", line_color="grey", line_width=1,
                annotation_text=f"No-skill ({prevalence:.2f})",
                annotation_position="bottom right",
            )
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            legend=dict(yanchor="bottom", y=0.05, xanchor="left", x=0.05),
            template="plotly_white",
        )
        return fig

    def plot_calibration(self, n_bins=10, strategy='uniform'):
        """Interactive Plotly calibration / reliability diagram.

        Args:
            n_bins: Number of probability bins.
            strategy: 'uniform' for equal-width bins, 'quantile' for equal-count.

        Returns:
            plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("plotly is required for plot_calibration(). pip install plotly")

        y_bin = (self.y_true == 1).astype(float) if not np.issubdtype(self.y_true.dtype, np.floating) else self.y_true

        if strategy == 'quantile':
            quantiles = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(self.y_proba, quantiles)
            bins = np.unique(bins)
        else:
            bins = np.linspace(0, 1, n_bins + 1)

        mean_predicted = []
        fraction_positive = []
        counts = []
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                mask = (self.y_proba >= bins[i]) & (self.y_proba <= bins[i + 1])
            else:
                mask = (self.y_proba >= bins[i]) & (self.y_proba < bins[i + 1])
            if mask.sum() == 0:
                continue
            mean_predicted.append(float(self.y_proba[mask].mean()))
            fraction_positive.append(float(y_bin[mask].mean()))
            counts.append(int(mask.sum()))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Perfectly Calibrated',
            line=dict(color='grey', width=1, dash='dash'),
            showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=mean_predicted, y=fraction_positive, mode='lines+markers',
            name='Model',
            text=[f"n={c}" for c in counts],
            hovertemplate="Mean predicted: %{x:.3f}<br>Fraction positive: %{y:.3f}<br>%{text}",
            line=dict(color='#00CC96', width=2),
            marker=dict(size=8),
        ))
        fig.update_layout(
            title="Calibration Curve (Reliability Diagram)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
            template="plotly_white",
        )
        return fig


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def tune_threshold(clf, X, y, objective='f1', beta=1.0, min_precision=None,
                   min_recall=None, target_precision=None, target_recall=None,
                   thresholds=None, pos_label=1, verbose=True):
    """Find the optimal classification threshold for a fitted classifier.

    Args:
        clf: Fitted classifier with predict_proba(X) method.
        X: Feature matrix (n_samples, n_features).
        y: True labels (n_samples,).
        objective: Metric to optimize -- 'f1', 'fbeta', 'recall', 'precision', or 'mcc'.
        beta: Beta value for F-beta score (only used when objective='fbeta').
              beta > 1 weights recall higher, beta < 1 weights precision higher.
        min_precision: Constraint: keep precision >= this value.
        min_recall: Constraint: keep recall >= this value.
        target_precision: Find threshold closest to this precision (overrides objective).
        target_recall: Find threshold closest to this recall (overrides objective).
        thresholds: Custom search grid. Default: np.linspace(0.01, 0.99, 199).
        pos_label: Which class is positive (default: 1).
        verbose: Print result summary.

    Returns:
        ThresholdResult with optimal threshold and metrics at that threshold.
    """
    probas = clf.predict_proba(X)
    if probas.ndim == 2:
        probas = probas[:, pos_label]

    y = np.asarray(y).ravel()

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 199)

    y_pos = (y == pos_label)

    precisions = np.empty(len(thresholds))
    recalls = np.empty(len(thresholds))
    f1s = np.empty(len(thresholds))
    fbetas = np.empty(len(thresholds))
    mccs = np.empty(len(thresholds))

    beta_sq = beta * beta

    for i, t in enumerate(thresholds):
        preds = probas >= t
        tp = int((preds & y_pos).sum())
        fp = int((preds & ~y_pos).sum())
        fn = int((~preds & y_pos).sum())
        tn = int((~preds & ~y_pos).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1_val = 2 * prec * rec / max(prec + rec, 1e-10)
        fb_val = (1 + beta_sq) * prec * rec / max(beta_sq * prec + rec, 1e-10)
        precisions[i] = prec
        recalls[i] = rec
        f1s[i] = f1_val
        fbetas[i] = fb_val
        mccs[i] = _mcc(tp, tn, fp, fn)

    all_metrics = {
        'precision': precisions,
        'recall': recalls,
        'f1': f1s,
        'fbeta': fbetas,
        'mcc': mccs,
    }

    if target_precision is not None:
        idx = np.argmin(np.abs(precisions - target_precision))
    elif target_recall is not None:
        idx = np.argmin(np.abs(recalls - target_recall))
    else:
        metric_map = {
            'f1': f1s, 'recall': recalls, 'precision': precisions,
            'fbeta': fbetas, 'mcc': mccs,
        }
        if objective not in metric_map:
            raise ValueError(
                f"objective must be one of {list(metric_map.keys())}, got '{objective}'"
            )
        scores = metric_map[objective].copy()
        mask = np.ones(len(thresholds), dtype=bool)
        if min_precision is not None:
            mask &= precisions >= min_precision
        if min_recall is not None:
            mask &= recalls >= min_recall
        if not mask.any():
            import warnings
            warnings.warn(
                f"No threshold satisfies constraints (min_precision={min_precision}, "
                f"min_recall={min_recall}). Returning best unconstrained {objective}."
            )
            mask[:] = True
        scores[~mask] = -np.inf
        idx = int(np.argmax(scores))

    result = ThresholdResult(
        threshold=float(thresholds[idx]),
        precision=float(precisions[idx]),
        recall=float(recalls[idx]),
        f1=float(f1s[idx]),
        objective=objective,
        all_thresholds=thresholds,
        all_metrics=all_metrics,
    )

    if verbose:
        mode = "target" if (target_precision or target_recall) else "objective"
        print(f"Threshold Tuning ({mode}={objective})")
        print(f"  Best threshold: {result.threshold:.4f}")
        print(f"  Precision:      {result.precision:.4f}")
        print(f"  Recall:         {result.recall:.4f}")
        print(f"  F1:             {result.f1:.4f}")
        if objective == 'fbeta':
            print(f"  F-beta (B={beta}): {float(fbetas[idx]):.4f}")
        if objective == 'mcc':
            print(f"  MCC:            {float(mccs[idx]):.4f}")

    return result


def evaluate_classifier(clf=None, X=None, y=None, y_proba=None, y_true=None,
                        threshold=0.5, pos_label=1, verbose=True):
    """Compute comprehensive classification metrics at a given threshold.

    Can be called two ways:
        evaluate_classifier(clf, X, y, threshold=0.5)
        evaluate_classifier(y_proba=probas, y_true=labels, threshold=0.5)

    Args:
        clf: Fitted classifier with predict_proba(X). Optional if y_proba provided.
        X: Feature matrix. Optional if y_proba provided.
        y: True labels. Optional if y_true provided.
        y_proba: Pre-computed probability array (1D, positive class). Alternative to clf+X.
        y_true: True labels. Alternative to y.
        threshold: Decision threshold or a ThresholdResult object.
        pos_label: Which class is positive (default: 1).
        verbose: Print formatted summary.

    Returns:
        ClassifierEvaluation with all metrics, curves, and plotting methods.
    """
    if isinstance(threshold, ThresholdResult):
        threshold = threshold.threshold

    if y_proba is None:
        if clf is None or X is None:
            raise ValueError("Provide either (clf, X, y) or (y_proba=, y_true=)")
        probas = clf.predict_proba(X)
        if probas.ndim == 2:
            probas = probas[:, pos_label]
        labels = np.asarray(y).ravel()
    else:
        probas = np.asarray(y_proba).ravel()
        labels = np.asarray(y_true).ravel()

    y_pos = (labels == pos_label)
    preds = (probas >= threshold).astype(int)

    tp, tn, fp, fn = _confusion_counts(y_pos.astype(int), preds)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1_val = 2 * precision * recall / max(precision + recall, 1e-10)
    mcc_val = _mcc(tp, tn, fp, fn)

    y_binary = y_pos.astype(int)
    brier = _brier_score(y_binary, probas)
    logloss = _log_loss(y_binary, probas)

    fpr, tpr, roc_thresholds = _roc_curve(y_binary, probas)
    roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds}

    pr_prec, pr_rec, pr_thresholds = _pr_curve(y_binary, probas)
    pr_curve_data = {'precision': pr_prec, 'recall': pr_rec, 'thresholds': pr_thresholds}

    roc_auc_val = _auc_trapz(fpr, tpr)
    pr_auc_val = _average_precision(y_binary, probas)
    gini = 2.0 * roc_auc_val - 1.0
    ks_stat = float(np.max(tpr - fpr)) if len(fpr) > 0 else 0.0

    cm = np.array([[tn, fp], [fn, tp]], dtype=int)

    support = {
        'positive': int(y_pos.sum()),
        'negative': int((~y_pos).sum()),
        'total': total,
    }

    result = ClassifierEvaluation(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1_val,
        specificity=specificity,
        mcc=mcc_val,
        roc_auc=roc_auc_val,
        pr_auc=pr_auc_val,
        brier_score=brier,
        log_loss=logloss,
        gini_coefficient=gini,
        ks_statistic=ks_stat,
        confusion_matrix=cm,
        roc_curve=roc_curve_data,
        pr_curve=pr_curve_data,
        threshold=threshold,
        support=support,
        y_proba=probas,
        y_true=labels,
    )

    if verbose:
        print(result.summary())

    return result


def compare_classifiers(evaluations, names=None, sort_by='roc_auc', ascending=False):
    """Side-by-side comparison table of multiple ClassifierEvaluation objects.

    Args:
        evaluations: List of ClassifierEvaluation objects or list of
                     (name, ClassifierEvaluation) tuples.
        names: Optional list of model names (ignored if tuples provided).
        sort_by: Column to sort by (any metric name). Default: 'roc_auc'.
        ascending: Sort ascending instead of descending.

    Returns:
        dict with keys:
            'table'  -- list of dicts (one per model), each containing all scalar metrics
            'best'   -- name of the best model (by sort_by)
            'text'   -- formatted comparison string ready to print

    Example:
        eval_rf = rfx.evaluate_classifier(rf, X, y, verbose=False)
        eval_xgb = rfx.evaluate_classifier(xgb, X, y, verbose=False)
        result = rfx.compare_classifiers([eval_rf, eval_xgb], names=['RF', 'XGBoost'])
        print(result['text'])
    """
    if isinstance(evaluations[0], tuple):
        pairs = evaluations
    else:
        if names is None:
            names = [f"Model_{i}" for i in range(len(evaluations))]
        pairs = list(zip(names, evaluations))

    metric_keys = [
        'accuracy', 'precision', 'recall', 'f1', 'specificity',
        'mcc', 'roc_auc', 'pr_auc', 'brier_score', 'log_loss',
        'gini_coefficient', 'ks_statistic',
    ]

    table = []
    for name, ev in pairs:
        row = {'name': name}
        for k in metric_keys:
            row[k] = getattr(ev, k, float('nan'))
        row['threshold'] = ev.threshold
        table.append(row)

    reverse = not ascending
    table.sort(key=lambda r: r.get(sort_by, 0), reverse=reverse)

    best_name = table[0]['name']

    header_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc',
                      'mcc', 'brier_score', 'log_loss', 'gini_coefficient', 'ks_statistic']
    col_w = 12
    name_w = max(len(r['name']) for r in table) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in header_metrics)
    sep = "-" * len(header)

    lines = ["Classifier Comparison", sep, header, sep]
    for row in table:
        line = f"{row['name']:<{name_w}}"
        for m in header_metrics:
            val = row.get(m, float('nan'))
            line += f"{val:>{col_w}.4f}"
        lines.append(line)
    lines.append(sep)
    lines.append(f"Best by {sort_by}: {best_name}")

    return {
        'table': table,
        'best': best_name,
        'text': "\n".join(lines),
    }


def bootstrap_ci(y_true, y_proba, metric='roc_auc', threshold=0.5,
                 n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence intervals for a classification metric.

    Args:
        y_true: True binary labels (array-like).
        y_proba: Predicted probabilities for positive class (array-like).
        metric: One of 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall',
                'accuracy', 'mcc', 'brier_score', 'log_loss', 'ks_statistic'.
        threshold: Decision threshold (used for threshold-dependent metrics).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default: 0.95 = 95%).
        seed: Random seed for reproducibility.

    Returns:
        dict with keys:
            'mean'   -- mean of bootstrap distribution
            'median' -- median of bootstrap distribution
            'ci_lower' / 'ci_upper' -- confidence interval bounds
            'std'    -- standard deviation
            'scores' -- all bootstrap scores (ndarray)

    Example:
        result = rfx.bootstrap_ci(y_test, probas, metric='roc_auc', ci=0.95)
        print(f"ROC AUC: {result['mean']:.4f} ({result['ci_lower']:.4f} - {result['ci_upper']:.4f})")
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    n = len(y_true)
    rng = np.random.RandomState(seed)

    def _compute(yt, yp):
        preds = (yp >= threshold).astype(int)
        tp, tn, fp, fn = _confusion_counts(yt, preds)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)

        if metric == 'accuracy':
            return (tp + tn) / max(tp + tn + fp + fn, 1)
        elif metric == 'precision':
            return prec
        elif metric == 'recall':
            return rec
        elif metric == 'f1':
            return 2 * prec * rec / max(prec + rec, 1e-10)
        elif metric == 'mcc':
            return _mcc(tp, tn, fp, fn)
        elif metric == 'brier_score':
            return _brier_score(yt, yp)
        elif metric == 'log_loss':
            return _log_loss(yt, yp)
        elif metric == 'roc_auc':
            return _roc_auc(yt, yp)
        elif metric == 'pr_auc':
            return _average_precision(yt, yp)
        elif metric == 'ks_statistic':
            return _ks_statistic(yt, yp)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    scores = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            scores[b] = float('nan')
            continue
        scores[b] = _compute(y_true[idx], y_proba[idx])

    valid = scores[~np.isnan(scores)]
    alpha = (1 - ci) / 2

    return {
        'mean': float(np.mean(valid)) if len(valid) > 0 else float('nan'),
        'median': float(np.median(valid)) if len(valid) > 0 else float('nan'),
        'ci_lower': float(np.percentile(valid, 100 * alpha)) if len(valid) > 0 else float('nan'),
        'ci_upper': float(np.percentile(valid, 100 * (1 - alpha))) if len(valid) > 0 else float('nan'),
        'std': float(np.std(valid)) if len(valid) > 0 else float('nan'),
        'scores': scores,
    }
