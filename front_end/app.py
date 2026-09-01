import streamlit as st
import pandas as pd
import os
import sys


# ============================================================
# PROJECT PATH
# ============================================================

# app.py is inside:
# Telecom.prj/front_end/
#
# So we go one level up to:
# Telecom.prj/

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT AI OPTIMIZER
# ============================================================

try:
    from data.optimizer import optimize_tower
except Exception as e:
    st.error(
        "Unable to load the AI optimizer."
    )
    st.error(str(e))
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Energy Dashboard",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "5G_energy_consumption_dataset.csv"
)


if not os.path.exists(DATA_PATH):

    st.error(
        f"Dataset not found:\n{DATA_PATH}"
    )

    st.stop()


try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📡 Telecom Energy Intelligence")

st.subheader(
    "AI-Powered Base Station Energy Optimization"
)

st.write(
    "Monitor telecom base stations and use AI to "
    "identify energy-efficient operating configurations."
)

st.divider()


# ============================================================
# SIDEBAR — TOWER SELECTION
# ============================================================

st.sidebar.header("⚙️ Tower Configuration")


# Get available base stations
towers = sorted(
    df["BS"].dropna().unique()
)


if len(towers) == 0:

    st.error(
        "No base stations found in the dataset."
    )

    st.stop()


selected_tower = st.sidebar.selectbox(
    "Select Base Station",
    towers
)


# ============================================================
# GET SELECTED TOWER DATA
# ============================================================

tower_data = df[
    df["BS"] == selected_tower
].copy()


if tower_data.empty:

    st.error(
        "No data available for the selected base station."
    )

    st.stop()


# ============================================================
# GET LATEST RECORD
# ============================================================

latest = tower_data.iloc[-1]


# ============================================================
# CURRENT TOWER VALUES
# ============================================================

current_energy = float(
    latest["Energy"]
)

current_load = float(
    latest["load"]
)

current_esmode = float(
    latest["ESMODE"]
)

current_txpower = float(
    latest["TXpower"]
)


# ============================================================
# TIME INFORMATION
# ============================================================

time_value = pd.to_datetime(
    latest["Time"]
)

current_hour = int(
    time_value.hour
)

current_dayofweek = int(
    time_value.dayofweek
)


# ============================================================
# CURRENT STATUS
# ============================================================

st.header(
    f"📡 Base Station: {selected_tower}"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current Load",
        f"{current_load * 100:.1f}%"
    )


with col2:

    st.metric(
        "Current Energy",
        f"{current_energy:.2f}"
    )


with col3:

    st.metric(
        "TX Power",
        f"{current_txpower:.3f}"
    )


with col4:

    st.metric(
        "ESMODE",
        f"{current_esmode:.3f}"
    )


st.divider()


# ============================================================
# AI OPTIMIZATION
# ============================================================

st.header(
    "🤖 AI Energy Optimization"
)

st.write(
    "The AI evaluates historically observed configurations "
    "under similar load conditions and recommends an "
    "energy-efficient configuration."
)


if st.button(
    "⚡ Optimize Tower",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "AI is evaluating tower configurations..."
    ):

        try:

            result = optimize_tower(

                load=current_load,

                current_esmode=current_esmode,

                current_txpower=current_txpower,

                hour=current_hour,

                dayofweek=current_dayofweek

            )

            # Store AI result
            st.session_state[
                "optimization_result"
            ] = result

        except Exception as e:

            st.error(
                f"Optimization failed: {e}"
            )


# ============================================================
# DISPLAY AI RESULT
# ============================================================

if "optimization_result" in st.session_state:

    result = st.session_state[
        "optimization_result"
    ]


    st.divider()

    st.header(
        "📊 AI Optimization Result"
    )


    # ========================================================
    # MAIN METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Energy",
            f"{result['current_energy']:.2f}"
        )


    with col2:

        st.metric(
            "Optimized Energy",
            f"{result['optimized_energy']:.2f}",
            delta=f"-{result['saving']:.2f}"
        )


    with col3:

        st.metric(
            "Potential Saving",
            f"{result['saving_percent']:.1f}%"
        )


    st.divider()


    # ========================================================
    # RECOMMENDED CONFIGURATION
    # ========================================================

    st.subheader(
        "⚙️ Recommended Configuration"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Recommended ESMODE",
            f"{result['recommended_ESMODE']:.3f}"
        )


    with col2:

        st.metric(
            "Recommended TX Power",
            f"{result['recommended_TXpower']:.3f}"
        )


    with col3:

        st.metric(
            "Load Category",
            result["load_category"]
        )


    # ========================================================
    # EVIDENCE
    # ========================================================

    st.subheader(
        "🔎 Recommendation Evidence"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Configurations Evaluated",
            result["candidate_count"]
        )


    with col2:

        st.metric(
            "Historical Observations",
            result["historical_observations"]
        )


    with col3:

        st.metric(
            "Evidence Strength",
            result["evidence_strength"]
        )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    recommendation = result[
        "recommendation"
    ]


    if recommendation == (
        "ENERGY OPTIMIZATION RECOMMENDED"
    ):

        st.success(
            "💡 " + recommendation
        )

    elif recommendation == (
        "CONSERVATIVE OPTIMIZATION RECOMMENDED"
    ):

        st.warning(
            "⚠️ " + recommendation
        )

    else:

        st.info(
            "ℹ️ " + recommendation
        )


    # ========================================================
    # ENERGY COMPARISON
    # ========================================================

    st.subheader(
        "📉 Energy Comparison"
    )


    chart_data = pd.DataFrame(

        {

            "Energy": [

                result["current_energy"],

                result["optimized_energy"]

            ]

        },

        index=[

            "Current",

            "Optimized"

        ]

    )


    st.bar_chart(
        chart_data
    )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.subheader(
        "🧠 AI Recommendation Explanation"
    )


    saving_percent = result[
        "saving_percent"
    ]


    if saving_percent > 20:

        st.write(
            f"The AI identified a potential energy "
            f"reduction of approximately "
            f"**{saving_percent:.1f}%** under the "
            f"current operating conditions."
        )

    elif saving_percent > 5:

        st.write(
            f"The AI identified a moderate potential "
            f"energy reduction of approximately "
            f"**{saving_percent:.1f}%**."
        )

    else:

        st.write(
            "The AI found limited energy-saving "
            "potential under the current conditions."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-powered telecom base station energy "
    "optimization • Hackathon Prototype"
)