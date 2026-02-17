

"""
intake_processor.py
-------------------
Receives the raw JSON payload from the PWA.
Converts it into the 10 clinical scores.
Feeds those scores into adhd_predictor.predict().

This is the single bridge between your frontend and your ML model.

Flow:
  PWA (sliders + game) → JSON → process_intake() → predict() → result
"""

import numpy as np
from adhd_predictor import predict


# ── CONVERSION LOGIC ──────────────────────────────────────────────────────────

def calculate_aqvis(hits: int, misses: int, target_trials: int) -> float:
    """
    Visual Attention Quotient.
    Measures how well the user detected targets (yellow circles).

    Formula: 100 - (misses / target_trials * 100) * penalty_weight
    Range:   0 → 100. Higher = better attention.

    Example:
        9 target trials, 2 misses → missed 22% of targets
        Aqvis = 100 - (2/9 * 100) = 77.8
    """
    if target_trials == 0:
        return 50.0  # neutral fallback if no targets shown
    miss_rate = misses / target_trials
    return round(100 - (miss_rate * 100), 1)


def calculate_rcqvis(false_alarms: int, distractor_trials: int) -> float:
    """
    Visual Response Control Quotient.
    Measures how well the user suppressed responses to distractors (white circles).

    Formula: 100 - (false_alarms / distractor_trials * 100) * penalty_weight
    Range:   0 → 100. Higher = better impulse control.

    Example:
        21 distractor trials, 3 false alarms → clicked 14% of distractors
        RCQvis = 100 - (3/21 * 100) = 85.7
    """
    if distractor_trials == 0:
        return 50.0
    false_alarm_rate = false_alarms / distractor_trials
    return round(100 - (false_alarm_rate * 100), 1)


def calculate_rtv(reaction_times: list) -> float:
    """
    Reaction Time Variability.
    NOT sent to the KNN model directly — but logged for future ML features.

    High RTV (high standard deviation) = inconsistent attention = ADHD signal.
    This is the most important raw biomarker you are collecting.

    Example:
        RTs = [312, 445, 289, 501, 398] → SD = 82ms → high variability
    """
    if len(reaction_times) < 2:
        return 0.0
    return round(float(np.std(reaction_times)), 2)


def derive_totals(aqvis: float, aqaudi: float,
                  rcqvis: float, rcqaudi: float) -> tuple:
    """
    Aqtot and RCQtot are simple averages of visual + auditory scores.
    When auditory game is not yet built, we estimate from visual only.
    """
    aqtot  = round((aqvis + aqaudi) / 2, 1)
    rcqtot = round((rcqvis + rcqaudi) / 2, 1)
    return aqtot, rcqtot


# ── MAIN BRIDGE FUNCTION ──────────────────────────────────────────────────────

def process_intake(payload: dict) -> dict:
    """
    Takes the raw JSON from the PWA.
    Returns a prediction result + the full calculated feature set.

    Parameters
    ----------
    payload : dict — exact structure the PWA sends (see example below)

    Returns
    -------
    {
        "user_id"    : "user_abc123",
        "features"   : { all 10 clinical scores },
        "rtv"        : 82.4,          ← logged but not in model yet
        "prediction" : {
            "subtype"    : 1,
            "label"      : "ADHD — Inattentive",
            "confidence" : 0.68,
            "all_probs"  : {0: 0.12, 1: 0.68, 2: 0.20}
        }
    }
    """

    # ── Step 1: Pull raw values from payload ──────────────────────────────────
    user_id = payload.get("id", "unknown")
    gender  = payload.get("Gender", "1")       # logged, not used by model

    # Conners scores — straight from sliders (0–4)
    conners = payload["conners"]
    cIM = float(conners["cIM"])
    cHR = float(conners["cHR"])
    cIE = float(conners["cIE"])
    cSC = float(conners["cSC"])

    # Game raw events
    game            = payload["game"]
    total_trials    = game["total_trials"]
    target_trials   = game["target_trials"]
    distractor_trials = total_trials - target_trials
    hits            = game["hits"]
    misses          = game["misses"]
    false_alarms    = game["false_alarms"]
    reaction_times  = game.get("reaction_times", [])

    # ── Step 2: Convert game events → clinical scores ─────────────────────────
    Aqvis  = calculate_aqvis(hits, misses, target_trials)
    RCQvis = calculate_rcqvis(false_alarms, distractor_trials)
    rtv    = calculate_rtv(reaction_times)

    # Auditory scores — placeholder until auditory game is built
    # Once auditory game ships, replace these with real calculations
    Aqaudi  = Aqvis   # temporary: assume same as visual
    RCQaudi = RCQvis  # temporary: assume same as visual

    # Derive totals
    Aqtot, RCQtot = derive_totals(Aqvis, Aqaudi, RCQvis, RCQaudi)

    # ── Step 3: Build feature dict for logging/debugging ──────────────────────
    features = {
        "cIM"    : cIM,
        "cHR"    : cHR,
        "cIE"    : cIE,
        "cSC"    : cSC,
        "Aqtot"  : Aqtot,
        "Aqaudi" : Aqaudi,
        "Aqvis"  : Aqvis,
        "RCQtot" : RCQtot,
        "RCQaudi": RCQaudi,
        "RCQvis" : RCQvis,
    }

    # ── Step 4: Feed into predictor ───────────────────────────────────────────
    prediction = predict(**features)

    return {
        "user_id"    : user_id,
        "gender"     : gender,
        "features"   : features,
        "rtv"        : rtv,         # log this — future ML feature
        "prediction" : prediction
    }


# ── EXAMPLE / QUICK TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # This is exactly what your PWA will send
    sample_payload = {
        "id"     : "user_abc123",
        "Gender" : "1",

        "conners": {
            "cIM": 3,   # Often loses focus
            "cHR": 2,   # Sometimes restless
            "cIE": 3,   # Often impulsive
            "cSC": 1    # Low self-concept
        },

        "game": {
            "total_trials"    : 30,
            "target_trials"   : 9,
            "hits"            : 7,
            "misses"          : 2,
            "correct_rejections": 18,
            "false_alarms"    : 3,
            "reaction_times"  : [312, 445, 289, 501, 398, 412, 356]
        }
    }

    print("=" * 55)
    print("  Intake Processor — Quick Test")
    print("=" * 55)

    result = process_intake(sample_payload)

    print(f"\nUser       : {result['user_id']}")
    print(f"\nCalculated features:")
    for k, v in result["features"].items():
        print(f"  {k:<10} {v}")

    print(f"\nRTV (jitter): {result['rtv']} ms SD")

    p = result["prediction"]
    print(f"\nPrediction : {p['label']}")
    print(f"Confidence : {p['confidence']:.0%}")
    print(f"All probs  : {p['all_probs']}")
    print("=" * 55)
