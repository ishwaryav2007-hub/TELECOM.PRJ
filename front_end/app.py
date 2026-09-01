import streamlit as st
import pandas as pd
import os
import sys


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Energy Intelligence",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT AI OPTIMIZER
# ============================================================

try:
    from data.optimizer import optimize_tower
except Exception as e:
    st.error("Unable to load AI optimizer.")
    st.error(str(e))
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "5G_energy_consumption_dataset.csv"
)

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

required_columns = [
    "BS",
    "Time",
    "Energy",
    "load",
    "ESMODE",
    "TXpower"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()


df["Time"] = pd.to_datetime(
    df["Time"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "BS",
        "Energy",
        "load",
        "ESMODE",
        "TXpower"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title("📡 Telecom Energy Intelligence")

st.subheader(
    "AI-Powered Base Station Energy Optimization"
)

st.write(
    "Monitor telecom base stations, identify energy "
    "consumption patterns, and use AI to discover "
    "energy-efficient operating configurations."
)


# ============================================================
# NETWORK OVERVIEW
# ============================================================

st.divider()

st.header("🌐 Network Overview")


total_towers = df["BS"].nunique()

average_energy = df["Energy"].mean()

total_energy = df["Energy"].sum()

average_load = df["load"].mean()


# Number of unique towers in each load category

low_load_towers = df[
    df["load"] < 0.30
]["BS"].nunique()

medium_load_towers = df[
    (df["load"] >= 0.30) &
    (df["load"] < 0.70)
]["BS"].nunique()

high_load_towers = df[
    df["load"] >= 0.70
]["BS"].nunique()


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "📡 Base Stations",
        total_towers
    )


with col2:
    st.metric(
        "⚡ Average Energy",
        f"{average_energy:.2f}"
    )


with col3:
    st.metric(
        "🔋 Total Energy",
        f"{total_energy:.2f}"
    )


with col4:
    st.metric(
        "📊 Average Load",
        f"{average_load * 100:.1f}%"
    )


# ============================================================
# NETWORK LOAD STATUS
# ============================================================

st.subheader("🚦 Network Load Status")


col1, col2, col3 = st.columns(3)


with col1:
    st.success(
        f"🟢 LOW LOAD\n\n"
        f"**{low_load_towers} towers**"
    )


with col2:
    st.warning(
        f"🟡 MEDIUM LOAD\n\n"
        f"**{medium_load_towers} towers**"
    )


with col3:
    st.error(
        f"🔴 HIGH LOAD\n\n"
        f"**{high_load_towers} towers**"
    )


# ============================================================
# TOP ENERGY CONSUMERS
# ============================================================

st.divider()

st.header("🔥 Highest Energy-Consuming Towers")


tower_summary = (
    df.groupby("BS")
    .agg(
        Average_Energy=("Energy", "mean"),
        Maximum_Energy=("Energy", "max"),
        Average_Load=("load", "mean"),
        Observations=("Energy", "count")
    )
    .reset_index()
)


top_energy = (
    tower_summary
    .sort_values(
        "Average_Energy",
        ascending=False
    )
    .head(10)
)


display_top = top_energy.copy()

display_top["Average_Energy"] = (
    display_top["Average_Energy"].round(2)
)

display_top["Maximum_Energy"] = (
    display_top["Maximum_Energy"].round(2)
)

display_top["Average_Load"] = (
    display_top["Average_Load"] * 100
).round(1)


display_top = display_top.rename(
    columns={
        "BS": "Base Station",
        "Average_Energy": "Avg Energy",
        "Maximum_Energy": "Max Energy",
        "Average_Load": "Avg Load (%)",
        "Observations": "Observations"
    }
)


st.dataframe(
    display_top,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ENERGY CONSUMPTION CHART
# ============================================================

st.subheader(
    "📊 Energy Consumption by Base Station"
)


energy_chart = top_energy[
    ["BS", "Average_Energy"]
].copy()


energy_chart = energy_chart.set_index(
    "BS"
)


energy_chart = energy_chart.rename(
    columns={
        "Average_Energy": "Average Energy"
    }
)


st.bar_chart(
    energy_chart
)


# ============================================================
# AI PRIORITY
# ============================================================

st.divider()

st.header("🎯 AI Optimization Priority")


st.write(
    "Towers are prioritized using their relative energy "
    "consumption and load. This helps identify where "
    "optimization may have the greatest impact."
)


priority = tower_summary.copy()


# Normalize energy

energy_min = priority["Average_Energy"].min()

energy_max = priority["Average_Energy"].max()


if energy_max > energy_min:

    energy_score = (
        priority["Average_Energy"] - energy_min
    ) / (
        energy_max - energy_min
    )

else:

    energy_score = 0


# Normalize load

load_min = priority["Average_Load"].min()

load_max = priority["Average_Load"].max()


if load_max > load_min:

    load_score = (
        priority["Average_Load"] - load_min
    ) / (
        load_max - load_min
    )

else:

    load_score = 0


# Priority score

priority["Priority_Score"] = (
    0.7 * energy_score +
    0.3 * load_score
)


priority = priority.sort_values(
    "Priority_Score",
    ascending=False
)


priority_top = priority.head(10).copy()


def get_priority(score):

    if score >= 0.70:
        return "🔴 HIGH"

    elif score >= 0.40:
        return "🟡 MEDIUM"

    else:
        return "🟢 LOW"


priority_top["Priority"] = (
    priority_top["Priority_Score"]
    .apply(get_priority)
)


priority_display = priority_top[
    [
        "BS",
        "Average_Energy",
        "Average_Load",
        "Priority",
        "Priority_Score"
    ]
].copy()


priority_display["Average_Energy"] = (
    priority_display["Average_Energy"]
    .round(2)
)

priority_display["Average_Load"] = (
    priority_display["Average_Load"] * 100
).round(1)

priority_display["Priority_Score"] = (
    priority_display["Priority_Score"]
    .round(2)
)


priority_display = priority_display.rename(
    columns={
        "BS": "Base Station",
        "Average_Energy": "Avg Energy",
        "Average_Load": "Avg Load (%)",
        "Priority": "AI Priority",
        "Priority_Score": "Priority Score"
    }
)


st.dataframe(
    priority_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INDIVIDUAL TOWER ANALYSIS
# ============================================================

st.divider()

st.header("📡 Individual Base Station Analysis")


towers = sorted(
    df["BS"].dropna().unique()
)


selected_tower = st.selectbox(
    "Select a base station",
    towers
)


tower_data = df[
    df["BS"] == selected_tower
].copy()


tower_data = tower_data.sort_values(
    "Time"
)


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
# LOAD STATUS
# ============================================================

if current_load < 0.30:

    load_status = "🟢 LOW"

elif current_load < 0.70:

    load_status = "🟡 MEDIUM"

else:

    load_status = "🔴 HIGH"


st.caption(
    f"Latest observation: {time_value}"
)


st.info(
    f"Current Load Status: **{load_status}**"
)


# ============================================================
# CURRENT METRICS
# ============================================================

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


# ============================================================
# ENERGY TREND
# ============================================================

st.subheader(
    "📈 Energy Consumption Trend"
)


trend_data = tower_data[
    ["Time", "Energy"]
].dropna()


if not trend_data.empty:

    trend_data = trend_data.set_index(
        "Time"
    )

    st.line_chart(
        trend_data["Energy"]
    )


# ============================================================
# INDIVIDUAL AI OPTIMIZATION
# ============================================================

st.subheader(
    "🤖 AI Energy Optimization"
)


st.write(
    "The AI evaluates historically observed "
    "configurations under similar operating conditions "
    "and recommends an energy-efficient configuration."
)


if st.button(
    "⚡ Optimize Selected Tower",
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

            st.session_state[
                "tower_result"
            ] = result

        except Exception as e:

            st.error(
                f"Optimization failed: {e}"
            )


# ============================================================
# INDIVIDUAL AI RESULT
# ============================================================

if "tower_result" in st.session_state:

    result = st.session_state[
        "tower_result"
    ]


    st.divider()

    st.header(
        "📊 AI Optimization Result"
    )


    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RECOMMENDED CONFIGURATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # CURRENT VS OPTIMIZED
    # --------------------------------------------------------

    st.subheader(
        "📉 Current vs AI Optimized Energy"
    )


    comparison = pd.DataFrame(

        {
            "Energy": [
                result["current_energy"],
                result["optimized_energy"]
            ]
        },

        index=[
            "Current",
            "AI Optimized"
        ]
    )


    st.bar_chart(
        comparison
    )


    # --------------------------------------------------------
    # AI DECISION SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "🧠 AI Decision Summary"
    )


    st.write(
        f"The system classified the current load as "
        f"**{result['load_category']}**."
    )


    st.write(
        f"The optimizer evaluated "
        f"**{result['candidate_count']} configurations** "
        f"using **{result['historical_observations']} "
        f"historical observations**."
    )


    st.write(
        f"The recommended configuration uses an ESMODE of "
        f"**{result['recommended_ESMODE']:.3f}** and TX Power "
        f"of **{result['recommended_TXpower']:.3f}**."
    )


    st.write(
        f"The estimated energy saving is "
        f"**{result['saving']:.2f} units "
        f"({result['saving_percent']:.1f}%)**."
    )


# ============================================================
# NETWORK-WIDE ANALYSIS — LAST
# ============================================================

st.divider()

st.header("⚡ Network-Wide AI Analysis")


st.write(
    "Run the AI optimizer across the latest available "
    "observation from every base station."
)


st.warning(
    "⏳ This analysis may take longer because the AI "
    "optimizer is executed for every tower."
)


if st.button(
    "🚀 Analyze Entire Network",
    type="secondary",
    use_container_width=True
):

    network_results = []

    tower_list = sorted(
        df["BS"].dropna().unique()
    )

    total = len(tower_list)

    progress = st.progress(0)

    status_text = st.empty()


    for index, tower in enumerate(
        tower_list
    ):

        status_text.write(
            f"Analyzing tower "
            f"{index + 1} of {total}: {tower}"
        )


        tower_data = df[
            df["BS"] == tower
        ].sort_values("Time")


        latest = tower_data.iloc[-1]


        try:

            time_value = pd.to_datetime(
                latest["Time"]
            )


            result = optimize_tower(

                load=float(
                    latest["load"]
                ),

                current_esmode=float(
                    latest["ESMODE"]
                ),

                current_txpower=float(
                    latest["TXpower"]
                ),

                hour=int(
                    time_value.hour
                ),

                dayofweek=int(
                    time_value.dayofweek
                )
            )


            network_results.append(
                {
                    "Base Station":
                        tower,

                    "Current Energy":
                        result["current_energy"],

                    "Optimized Energy":
                        result["optimized_energy"],

                    "Saving":
                        result["saving"],

                    "Saving %":
                        result["saving_percent"],

                    "Load Category":
                        result["load_category"],

                    "Recommendation":
                        result["recommendation"]
                }
            )


        except Exception as e:

            network_results.append(
                {
                    "Base Station":
                        tower,

                    "Current Energy":
                        float(
                            latest["Energy"]
                        ),

                    "Optimized Energy":
                        None,

                    "Saving":
                        None,

                    "Saving %":
                        None,

                    "Load Category":
                        "ERROR",

                    "Recommendation":
                        str(e)
                }
            )


        progress.progress(
            (index + 1) / total
        )


    status_text.success(
        "✅ Network analysis completed!"
    )


    network_results_df = pd.DataFrame(
        network_results
    )


    st.session_state[
        "network_results"
    ] = network_results_df


# ============================================================
# DISPLAY NETWORK RESULTS
# ============================================================

if "network_results" in st.session_state:

    network_results = st.session_state[
        "network_results"
    ]


    st.subheader(
        "📊 Network AI Results"
    )


    valid_results = network_results[
        network_results["Saving %"].notna()
    ].copy()


    if not valid_results.empty:

        total_current = (
            valid_results[
                "Current Energy"
            ].sum()
        )


        total_optimized = (
            valid_results[
                "Optimized Energy"
            ].sum()
        )


        total_saving = (
            valid_results[
                "Saving"
            ].sum()
        )


        if total_current > 0:

            network_saving_percent = (
                total_saving /
                total_current
            ) * 100

        else:

            network_saving_percent = 0


        # ----------------------------------------------------
        # NETWORK RESULT METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Towers Analyzed",
                len(valid_results)
            )


        with col2:

            st.metric(
                "Current Energy",
                f"{total_current:.2f}"
            )


        with col3:

            st.metric(
                "Optimized Energy",
                f"{total_optimized:.2f}"
            )


        with col4:

            st.metric(
                "Potential Saving",
                f"{network_saving_percent:.1f}%"
            )


        # ----------------------------------------------------
        # RESULTS TABLE
        # ----------------------------------------------------

        st.subheader(
            "🏆 Tower Optimization Results"
        )


        result_display = valid_results.copy()


        result_display[
            "Current Energy"
        ] = (
            result_display[
                "Current Energy"
            ].round(2)
        )


        result_display[
            "Optimized Energy"
        ] = (
            result_display[
                "Optimized Energy"
            ].round(2)
        )


        result_display[
            "Saving"
        ] = (
            result_display[
                "Saving"
            ].round(2)
        )


        result_display[
            "Saving %"
        ] = (
            result_display[
                "Saving %"
            ].round(1)
        )


        st.dataframe(
            result_display,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # BIGGEST SAVING OPPORTUNITIES
        # ----------------------------------------------------

        st.subheader(
            "💡 Biggest Optimization Opportunities"
        )


        biggest_savings = (
            valid_results
            .sort_values(
                "Saving",
                ascending=False
            )
            .head(5)
        )


        savings_chart = (
            biggest_savings[
                [
                    "Base Station",
                    "Saving"
                ]
            ]
            .set_index(
                "Base Station"
            )
        )


        st.bar_chart(
            savings_chart
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-powered telecom base station energy optimization "
    "• Hackathon Prototype"
)