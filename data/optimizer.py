import os
import pandas as pd
import joblib


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "5G_energy_consumption_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "energy_model_final.pkl"
)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)


# ============================================================
# PREPARE HISTORICAL CONFIGURATION DATA
# ============================================================

# We use historically observed configurations.
# This prevents the optimizer from inventing completely
# new TXpower/ESMODE combinations.

configuration_stats = (
    df.groupby(
        ["ESMODE", "TXpower"]
    )
    .agg(
        observations=("Energy", "count"),
        average_energy=("Energy", "mean")
    )
    .reset_index()
)


# ============================================================
# ENERGY PREDICTION
# ============================================================

def predict_energy(
    load: float,
    esmode: float,
    txpower: float,
    hour: int,
    dayofweek: int
) -> float:

    """
    Predict energy consumption using the trained
    Random Forest model.
    """

    input_data = pd.DataFrame([{
        "load": load,
        "ESMODE": esmode,
        "TXpower": txpower,
        "hour": hour,
        "dayofweek": dayofweek
    }])

    prediction = model.predict(input_data)[0]

    return float(prediction)


# ============================================================
# LOAD CATEGORY
# ============================================================

def get_load_category(load: float) -> str:

    if load < 0.30:
        return "LOW"

    elif load < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_tower(
    load: float,
    current_esmode: float,
    current_txpower: float,
    hour: int,
    dayofweek: int
) -> dict:

    """
    Find an energy-efficient configuration for the
    current tower state.

    This is a decision-support recommendation.
    It does not directly control a telecom tower.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not 0 <= load <= 1:
        raise ValueError(
            "load must be between 0 and 1"
        )

    if not 0 <= hour <= 23:
        raise ValueError(
            "hour must be between 0 and 23"
        )

    if not 0 <= dayofweek <= 6:
        raise ValueError(
            "dayofweek must be between 0 and 6"
        )

    if current_txpower < 0:
        raise ValueError(
            "current_txpower cannot be negative"
        )


    # --------------------------------------------------------
    # Current configuration prediction
    # --------------------------------------------------------

    current_energy = predict_energy(
        load=load,
        esmode=current_esmode,
        txpower=current_txpower,
        hour=hour,
        dayofweek=dayofweek
    )


    # --------------------------------------------------------
    # Determine load category
    # --------------------------------------------------------

    load_category = get_load_category(load)


    # --------------------------------------------------------
    # Determine comparable historical load range
    # --------------------------------------------------------

    if load_category == "LOW":

        tolerance = 0.10

    elif load_category == "MEDIUM":

        tolerance = 0.05

    else:

        tolerance = 0.02


    lower_load = max(
        0,
        load - tolerance
    )

    upper_load = min(
        1,
        load + tolerance
    )


    # --------------------------------------------------------
    # Find historically similar operating conditions
    # --------------------------------------------------------

    similar_data = df[
        (df["load"] >= lower_load)
        &
        (df["load"] <= upper_load)
    ].copy()


    # --------------------------------------------------------
    # Find configurations actually observed under
    # similar load conditions
    # --------------------------------------------------------

    candidates = (
        similar_data
        .groupby(
            ["ESMODE", "TXpower"]
        )
        .agg(
            observations=("Energy", "count"),
            historical_average_energy=(
                "Energy",
                "mean"
            )
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Require minimum historical evidence
    # --------------------------------------------------------

    candidates = candidates[
        candidates["observations"] >= 10
    ].copy()


    # --------------------------------------------------------
    # Never recommend increasing TX power
    # --------------------------------------------------------

    candidates = candidates[
        candidates["TXpower"]
        <= current_txpower
    ].copy()


    # --------------------------------------------------------
    # HIGH LOAD:
    #
    # We only allow conservative recommendations.
    # --------------------------------------------------------

    if load_category == "HIGH":

        # Allow only a small TX power reduction.
        minimum_txpower = (
            current_txpower * 0.95
        )

        candidates = candidates[
            candidates["TXpower"]
            >= minimum_txpower
        ].copy()


    # --------------------------------------------------------
    # If no reliable candidate exists,
    # keep the current configuration.
    # --------------------------------------------------------

    if candidates.empty:

        return {
            "status": "NO_SAFE_RECOMMENDATION",

            "current_energy": round(
                current_energy,
                2
            ),

            "optimized_energy": round(
                current_energy,
                2
            ),

            "saving": 0.0,

            "saving_percent": 0.0,

            "recommended_ESMODE":
                current_esmode,

            "recommended_TXpower":
                current_txpower,

            "load_category":
                load_category,

            "candidate_count": 0,

            "historical_observations": 0,

            "evidence_strength":
                "NONE",

            "recommendation":
                "KEEP CURRENT CONFIGURATION"
        }


    # --------------------------------------------------------
    # Predict energy for each candidate
    # --------------------------------------------------------

    predicted_energies = []

    for _, candidate in candidates.iterrows():

        predicted = predict_energy(
            load=load,
            esmode=float(
                candidate["ESMODE"]
            ),
            txpower=float(
                candidate["TXpower"]
            ),
            hour=hour,
            dayofweek=dayofweek
        )

        predicted_energies.append(
            predicted
        )


    candidates["predicted_energy"] = (
        predicted_energies
    )


    # --------------------------------------------------------
    # Combine AI prediction with historical evidence
    #
    # 70% ML prediction
    # 30% historical average
    # --------------------------------------------------------

    candidates["score"] = (
        0.70 *
        candidates["predicted_energy"]
        +
        0.30 *
        candidates["historical_average_energy"]
    )


    # --------------------------------------------------------
    # Select best candidate
    # --------------------------------------------------------

    best_index = candidates[
        "score"
    ].idxmin()

    best = candidates.loc[
        best_index
    ]


    optimized_energy = float(
        best["predicted_energy"]
    )


    # --------------------------------------------------------
    # Calculate raw saving
    # --------------------------------------------------------

    raw_saving = (
        current_energy
        -
        optimized_energy
    )

    raw_saving_percent = (
        raw_saving
        /
        current_energy
    ) * 100


    # --------------------------------------------------------
    # Conservative saving cap
    #
    # This prevents extreme model predictions from being
    # presented as guaranteed savings.
    # --------------------------------------------------------

    if load_category == "LOW":

        maximum_reported_saving = 30.0

    elif load_category == "MEDIUM":

        maximum_reported_saving = 20.0

    else:

        maximum_reported_saving = 10.0


    reported_saving_percent = min(
        max(raw_saving_percent, 0),
        maximum_reported_saving
    )


    # --------------------------------------------------------
    # Calculate consistent displayed saving
    # --------------------------------------------------------

    reported_saving = (
        current_energy
        *
        reported_saving_percent
        /
        100
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Make optimized_energy mathematically consistent
    # with the reported saving.
    #
    # Example:
    #
    # Current = 50
    # Saving = 30%
    #
    # Optimized = 35
    # --------------------------------------------------------

    reported_optimized_energy = (
        current_energy
        -
        reported_saving
    )


    # --------------------------------------------------------
    # Recommendation logic
    # --------------------------------------------------------

    if load_category == "HIGH":

        recommendation = (
            "CONSERVATIVE OPTIMIZATION "
            "RECOMMENDED"
        )

    elif reported_saving_percent < 2:

        recommendation = (
            "KEEP CURRENT CONFIGURATION"
        )

    else:

        recommendation = (
            "ENERGY OPTIMIZATION RECOMMENDED"
        )


    # --------------------------------------------------------
    # Evidence strength
    # --------------------------------------------------------

    observations = int(
        best["observations"]
    )

    if observations >= 100:

        evidence = "STRONG"

    elif observations >= 30:

        evidence = "MODERATE"

    else:

        evidence = "LIMITED"


    # --------------------------------------------------------
    # Return backend-friendly result
    # --------------------------------------------------------

    return {

        "status": "SUCCESS",

        "current_energy": round(
            current_energy,
            2
        ),

        "optimized_energy": round(
            reported_optimized_energy,
            2
        ),

        "saving": round(
            reported_saving,
            2
        ),

        "saving_percent": round(
            reported_saving_percent,
            2
        ),

        "recommended_ESMODE": round(
            float(best["ESMODE"]),
            6
        ),

        "recommended_TXpower": round(
            float(best["TXpower"]),
            6
        ),

        "load_category":
            load_category,

        "candidate_count":
            int(len(candidates)),

        "historical_observations":
            observations,

        "evidence_strength":
            evidence,

        "recommendation":
            recommendation
    }


# ============================================================
# LOCAL TEST
#
# This runs ONLY when you execute:
#
#     python data/optimizer.py
#
# It will NOT run when FastAPI imports this file.
# ============================================================

if __name__ == "__main__":

    result = optimize_tower(

        load=0.18,

        current_esmode=0,

        current_txpower=7.101719,

        hour=2,

        dayofweek=6
    )

    print()
    print("===================================")
    print("AI OPTIMIZATION RESULT")
    print("===================================")

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )