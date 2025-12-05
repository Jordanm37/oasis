"""
Hybrid F1 Score Metric for Kaggle Competition: Needle in the Hashtag.

Computes a weighted combination of:
- Risk Tier F1 (70%): Macro F1 on 3 risk tiers (benign, recovery, risky)
- Persona F1 (30%): Macro F1 on all 13 persona classes

This hybrid approach prioritises safety (catching risky content) while rewarding
precise identification of specific harm types.

All columns of the solution and submission dataframes are passed to your metric, except for the Usage column.
"""

import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import f1_score


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


# Risk tier mappings (13 personas -> 3 tiers)
BENIGN_TIER = ['benign']
RECOVERY_TIER = ['recovery_ed']
RISKY_TIER = [
    'ed_risk', 'pro_ana', 'bullying', 'hate_speech', 'incel_misogyny',
    'extremist', 'misinfo', 'conspiracy', 'gamergate', 'alpha', 'trad'
]

# Weights for hybrid scoring
RISK_TIER_WEIGHT = 0.70
PERSONA_WEIGHT = 0.30


def _compute_tier_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse 13 persona columns into 3 risk tier columns.
    
    A tier is 1 if ANY of its constituent personas is 1.
    """
    tier_df = pd.DataFrame(index=df.index)
    
    # Benign tier: only 'benign' column
    benign_cols = [c for c in BENIGN_TIER if c in df.columns]
    tier_df['tier_benign'] = df[benign_cols].max(axis=1) if benign_cols else 0
    
    # Recovery tier: only 'recovery_ed' column
    recovery_cols = [c for c in RECOVERY_TIER if c in df.columns]
    tier_df['tier_recovery'] = df[recovery_cols].max(axis=1) if recovery_cols else 0
    
    # Risky tier: any of the 11 risky personas
    risky_cols = [c for c in RISKY_TIER if c in df.columns]
    tier_df['tier_risky'] = df[risky_cols].max(axis=1) if risky_cols else 0
    
    return tier_df


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Hybrid F1 Score: 70% Risk Tier + 30% Persona.
    
    Computes a weighted combination of:
    - Risk Tier F1 (70%): How well you distinguish benign vs recovery vs risky
    - Persona F1 (30%): How well you identify specific persona types
    
    Parameters
    ----------
    solution : pd.DataFrame
        Ground truth with columns: ID, alpha, benign, bullying, ..., trad
    submission : pd.DataFrame  
        Predictions with same columns as solution
    row_id_column_name : str
        Name of the ID column (e.g., "ID")
        
    Returns
    -------
    float
        Hybrid score = 0.70 * Risk_Tier_F1 + 0.30 * Persona_F1
        
    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "ID"
    >>> # Perfect prediction on benign
    >>> y_true = pd.DataFrame({'ID': [1], 'benign': [1], 'recovery_ed': [0], 'ed_risk': [0]})
    >>> y_pred = pd.DataFrame({'ID': [1], 'benign': [1], 'recovery_ed': [0], 'ed_risk': [0]})
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    1.0
    '''
    # Make copies to avoid modifying originals
    solution = solution.copy()
    submission = submission.copy()
    
    # Remove the row ID column
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    # Validate all columns are numeric
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    
    # Validate columns match
    if set(solution.columns) != set(submission.columns):
        missing = set(solution.columns) - set(submission.columns)
        extra = set(submission.columns) - set(solution.columns)
        msg = "Column mismatch."
        if missing:
            msg += f" Missing columns: {missing}."
        if extra:
            msg += f" Extra columns: {extra}."
        raise ParticipantVisibleError(msg)
    
    # Ensure same column order
    submission = submission[solution.columns]
    
    # Validate binary values (0 or 1)
    for col in submission.columns:
        unique_vals = submission[col].unique()
        if not all(v in [0, 1] for v in unique_vals):
            raise ParticipantVisibleError(
                f'Submission column {col} must contain only 0 or 1 values. Found: {unique_vals}'
            )
    
    # =========================================
    # COMPONENT 1: Persona F1 (30%)
    # =========================================
    y_true_persona = solution.values
    y_pred_persona = submission.values
    persona_f1 = float(f1_score(y_true_persona, y_pred_persona, average='macro', zero_division=0))
    
    # =========================================
    # COMPONENT 2: Risk Tier F1 (70%)
    # =========================================
    # Collapse 13 personas into 3 risk tiers
    solution_tiers = _compute_tier_labels(solution)
    submission_tiers = _compute_tier_labels(submission)
    
    y_true_tier = solution_tiers.values
    y_pred_tier = submission_tiers.values
    tier_f1 = float(f1_score(y_true_tier, y_pred_tier, average='macro', zero_division=0))
    
    # =========================================
    # HYBRID SCORE
    # =========================================
    hybrid_score = (RISK_TIER_WEIGHT * tier_f1) + (PERSONA_WEIGHT * persona_f1)
    
    return hybrid_score
