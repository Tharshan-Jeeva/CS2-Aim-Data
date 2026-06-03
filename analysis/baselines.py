"""Baseline classifiers for human-vs-aimbot trajectory classification.

Two models:
    - LogisticRegression on standardised features (linear baseline)
    - RandomForestClassifier (non-linear baseline)

Evaluation: Leave-One-Participant-Out CV. With only one participant available
(pre-recruitment), falls back to GroupKFold over sessions so the pipeline can
still be exercised against synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.dataset import FeatureMatrix


@dataclass
class FoldResult:
    fold: int
    held_out_group: str
    n_train: int
    n_test: int
    accuracy: float
    f1: float
    roc_auc: float
    confusion: np.ndarray


@dataclass
class CVReport:
    model_name: str
    splitter: str
    folds: list[FoldResult] = field(default_factory=list)

    def mean(self, attr: str) -> float:
        if not self.folds:
            return float("nan")
        return float(np.mean([getattr(f, attr) for f in self.folds]))

    def std(self, attr: str) -> float:
        if not self.folds:
            return float("nan")
        return float(np.std([getattr(f, attr) for f in self.folds]))

    def format(self) -> str:
        lines = [f"=== {self.model_name} ({self.splitter}) ==="]
        if not self.folds:
            lines.append("  (no folds — see splitter description above)")
            return "\n".join(lines)
        for fr in self.folds:
            lines.append(
                f"  fold {fr.fold} held_out={fr.held_out_group:>10s}  "
                f"n_test={fr.n_test:5d}  acc={fr.accuracy:.3f}  "
                f"f1={fr.f1:.3f}  auc={fr.roc_auc:.3f}"
            )
        lines.append(
            f"  MEAN  acc={self.mean('accuracy'):.3f}±{self.std('accuracy'):.3f}  "
            f"f1={self.mean('f1'):.3f}±{self.std('f1'):.3f}  "
            f"auc={self.mean('roc_auc'):.3f}±{self.std('roc_auc'):.3f}"
        )
        return "\n".join(lines)


def make_model(name: str) -> Pipeline:
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    if name == "rf":
        return Pipeline([
            # Tree models are scale-invariant; scaler kept for API symmetry.
            ("scale", StandardScaler(with_mean=False)),
            ("clf", RandomForestClassifier(
                n_estimators=300, n_jobs=-1, random_state=0,
                class_weight="balanced")),
        ])
    raise ValueError(f"Unknown model: {name}")


def _choose_splitter(groups: np.ndarray):
    """LOPO if we have >=2 participants, else GroupKFold over sessions."""
    n_groups = len(set(groups.tolist()))
    if n_groups >= 2:
        return LeaveOneGroupOut(), groups, "LeaveOneParticipantOut"
    # Fallback: split by session id so windows from the same session don't
    # leak across train/test.
    return GroupKFold(n_splits=min(5, max(2, n_groups))), groups, "GroupKFold"


def cross_validate(fm: FeatureMatrix, model_name: str) -> CVReport:
    """Run CV and return per-fold metrics.

    Returns an empty report (with a descriptive `splitter` field) when CV is
    not meaningful — e.g. a single session or a single-class dataset.
    """
    if len(np.unique(fm.y)) < 2:
        return CVReport(model_name=model_name,
                        splitter="SKIPPED: dataset is single-class")

    if len(set(fm.participant.tolist())) >= 2:
        groups = fm.participant
        group_kind = "participant"
    else:
        groups = fm.session
        group_kind = "session"

    if len(set(groups.tolist())) < 2:
        return CVReport(model_name=model_name,
                        splitter=f"SKIPPED: only one {group_kind}")

    splitter, groups, splitter_name = _choose_splitter(groups)
    report = CVReport(model_name=model_name, splitter=f"{splitter_name} ({group_kind})")

    for fold_idx, (tr, te) in enumerate(splitter.split(fm.X, fm.y, groups)):
        if len(np.unique(fm.y[tr])) < 2 or len(np.unique(fm.y[te])) < 2:
            # Skip degenerate folds (all-human or all-bot) — common with
            # very small pilot datasets. They produce undefined AUC.
            continue
        model = make_model(model_name)
        model.fit(fm.X[tr], fm.y[tr])
        pred = model.predict(fm.X[te])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(fm.X[te])[:, 1]
        else:
            proba = pred.astype(float)
        held_out = str(groups[te[0]])
        report.folds.append(FoldResult(
            fold=fold_idx,
            held_out_group=held_out,
            n_train=int(len(tr)),
            n_test=int(len(te)),
            accuracy=float(accuracy_score(fm.y[te], pred)),
            f1=float(f1_score(fm.y[te], pred, zero_division=0)),
            roc_auc=float(roc_auc_score(fm.y[te], proba)),
            confusion=confusion_matrix(fm.y[te], pred, labels=[0, 1]),
        ))
    return report


def fit_final_model(fm: FeatureMatrix, model_name: str) -> Pipeline:
    """Fit on all available data — for downstream use (e.g. analysis of
    classifier confidence vs subjective rating). NOT for headline metrics."""
    model = make_model(model_name)
    model.fit(fm.X, fm.y)
    return model
