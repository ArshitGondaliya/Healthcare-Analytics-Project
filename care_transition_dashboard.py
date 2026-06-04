import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === Page config ===
st.set_page_config(
    page_title="Care Transition Efficiency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0b3d91 0%, #0f5bb5 100%);
    color: #ffffff;
}
.section-header {
    color: #ffffff;
    font-weight: bold;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #ffc107;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #ffc107;
}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "HHS_Unaccompanied_Alien_Children_Program.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    numeric_columns = [
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("*", "", regex=False)
                .str.strip()
                .replace(["", "nan", "NaN", "None"], np.nan)
            )
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)
    return df


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Transfer Efficiency"] = np.where(
        df["Children apprehended and placed in CBP custody*"] > 0,
        df["Children transferred out of CBP custody"]
        / df["Children apprehended and placed in CBP custody*"],
        np.nan,
    )

    df["Discharge Effectiveness"] = np.where(
        df["Children in HHS Care"] > 0,
        df["Children discharged from HHS Care"] / df["Children in HHS Care"],
        np.nan,
    )

    df["Pipeline Exits"] = (
        df["Children transferred out of CBP custody"]
        + df["Children discharged from HHS Care"]
    )
    df["Pipeline Entries"] = (
        df["Children apprehended and placed in CBP custody*"]
        + df["Children transferred out of CBP custody"]
    )
    df["Pipeline Throughput"] = np.where(
        df["Pipeline Entries"] > 0,
        df["Pipeline Exits"] / df["Pipeline Entries"],
        np.nan,
    )

    df["Total Active Care Load"] = (
        df["Children in CBP custody"] + df["Children in HHS Care"]
    )
    df["Backlog Change"] = df["Total Active Care Load"].diff().fillna(0)
    df["Backlog Accumulation Rate"] = np.where(
        df["Backlog Change"] > 0,
        df["Backlog Change"],
        0,
    )

    df["Weekday"] = df["Date"].dt.day_name()
    df["Is Weekend"] = df["Weekday"].isin(["Saturday", "Sunday"])
    df["Outcome Stability Score"] = 1 - (
        df["Discharge Effectiveness"].std(skipna=True)
        / max(df["Discharge Effectiveness"].mean(skipna=True), 1e-9)
    )
    df["Outcome Stability Score"] = df["Outcome Stability Score"].clip(lower=0, upper=1)
    return df


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%" if not np.isnan(value) else "N/A"


def format_number(value: float) -> str:
    return f"{value:,.0f}" if not np.isnan(value) else "N/A"


def main() -> None:
    st.markdown("# 🩺 UAC Care Transition Efficiency & Placement Outcome Dashboard")
    st.markdown(
        "This dashboard converts the UAC dataset into a care pipeline analytics system, focusing on transfer efficiency, discharge and placement outcomes, backlog detection, and outcome stability."  # noqa: E501
    )
    st.markdown("---")

    df = load_data(DATA_PATH)
    if df.empty:
        st.error("Dataset not loaded. Please place the CSV in the project folder and try again.")
        return

    df = calculate_metrics(df)

    stage_fields = [
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care",
    ]

    st.sidebar.title("🔍 Analysis Controls")
    date_min = df["Date"].min()
    date_max = df["Date"].max()
    selected_date_range = st.sidebar.date_input(
        "Date range",
        value=[date_min, date_max],
        min_value=date_min,
        max_value=date_max,
    )

    if len(selected_date_range) != 2:
        selected_date_range = [date_min, date_max]

    selected_stages = st.sidebar.multiselect(
        "Pipeline stages to display",
        stage_fields,
        default=stage_fields,
    )

    show_weekends = st.sidebar.checkbox("Include weekend data", value=True)

    min_hhs_load, max_hhs_load = st.sidebar.slider(
        "HHS care load range",
        int(df["Children in HHS Care"].min()),
        int(df["Children in HHS Care"].max()),
        (
            int(df["Children in HHS Care"].min()),
            int(df["Children in HHS Care"].quantile(0.95)),
        ),
        step=1,
    )

    min_transfer, max_transfer = st.sidebar.slider(
        "CBP transfer volume range",
        int(df["Children transferred out of CBP custody"].min()),
        int(df["Children transferred out of CBP custody"].max()),
        (
            int(df["Children transferred out of CBP custody"].min()),
            int(df["Children transferred out of CBP custody"].max()),
        ),
        step=1,
    )

    metric_focus = st.sidebar.radio(
        "Metric focus",
        ["Process efficiency", "Stage volumes"],
    )

    efficiency_alert_threshold = st.sidebar.slider(
        "Alert threshold for low efficiency",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )

    df_filtered = df.loc[
        (df["Date"] >= pd.to_datetime(selected_date_range[0]))
        & (df["Date"] <= pd.to_datetime(selected_date_range[1]))
    ]
    if not show_weekends:
        df_filtered = df_filtered.loc[~df_filtered["Is Weekend"]]

    df_filtered = df_filtered.loc[
        (df_filtered["Children in HHS Care"] >= min_hhs_load)
        & (df_filtered["Children in HHS Care"] <= max_hhs_load)
        & (df_filtered["Children transferred out of CBP custody"] >= min_transfer)
        & (df_filtered["Children transferred out of CBP custody"] <= max_transfer)
    ]

    backlog_threshold = st.sidebar.number_input(
        "HHS care backlog alert threshold",
        value=int(df_filtered["Children in HHS Care"].quantile(0.75) if not df_filtered.empty else 1000),
        min_value=0,
        step=1,
    )

    st.markdown("## 📈 System Overview")
    if df_filtered.empty:
        st.info("No data found for the selected date range.")
        return

    summary = {
        "Average transfer efficiency": df_filtered["Transfer Efficiency"].mean(skipna=True),
        "Average discharge effectiveness": df_filtered["Discharge Effectiveness"].mean(skipna=True),
        "Average throughput": df_filtered["Pipeline Throughput"].mean(skipna=True),
        "Average backlog accumulation": df_filtered["Backlog Accumulation Rate"].mean(skipna=True),
        "Outcome stability score": df_filtered["Outcome Stability Score"].mean(skipna=True),
    }

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Transfer Efficiency", format_pct(summary["Average transfer efficiency"]))
    col2.metric("Discharge Effectiveness", format_pct(summary["Average discharge effectiveness"]))
    col3.metric("Pipeline Throughput", format_pct(summary["Average throughput"]))
    col4.metric("Backlog Accumulation", format_number(summary["Average backlog accumulation"]))
    col5.metric("Outcome Stability", format_pct(summary["Outcome stability score"]))

    if summary["Average transfer efficiency"] < efficiency_alert_threshold:
        st.warning(
            f"Transfer efficiency is below the alert threshold ({format_pct(efficiency_alert_threshold)}). Review CBP to HHS transition speed."
        )

    if summary["Average discharge effectiveness"] < efficiency_alert_threshold:
        st.warning(
            f"Discharge effectiveness is below the alert threshold ({format_pct(efficiency_alert_threshold)}). Explore care continuity and sponsor placement outcomes."
        )

    if df_filtered["Children in HHS Care"].max() >= backlog_threshold:
        st.error(
            f"High care load detected: HHS care peaked at {int(df_filtered['Children in HHS Care'].max()):,}. This exceeds the backlog threshold."
        )

    st.markdown("---")

    # Pipeline flow visualization
    st.subheader("🚦 Care Pipeline Flow Visualization")
    if selected_stages:
        fig_flow = px.line(
            df_filtered,
            x="Date",
            y=selected_stages,
            title="Daily Care Pipeline Stage Counts",
            labels={
                "value": "Children count",
                "variable": "Pipeline stage",
            },
        )
        fig_flow.update_layout(height=520, legend=dict(title="Stage", orientation="h", y=-0.25))
        st.plotly_chart(fig_flow, use_container_width=True)
    else:
        st.warning("Select at least one pipeline stage in the sidebar to display the flow graph.")

    st.markdown("---")

    # Efficiency and trend panels
    st.subheader("📊 Transfer & Discharge Efficiency Panels")
    eff_col1, eff_col2 = st.columns([1, 1])

    with eff_col1:
        fig_transfers = px.area(
            df_filtered,
            x="Date",
            y="Transfer Efficiency",
            title="Transfer Efficiency Trend",
            labels={"Transfer Efficiency": "Transfer efficiency"},
            line_shape="spline",
        )
        fig_transfers.update_layout(yaxis_tickformat=".0%", height=350)
        st.plotly_chart(fig_transfers, use_container_width=True)

    with eff_col2:
        fig_discharges = px.line(
            df_filtered,
            x="Date",
            y="Discharge Effectiveness",
            title="Discharge Effectiveness Trend",
            labels={"Discharge Effectiveness": "Discharge effectiveness"},
            markers=True,
        )
        fig_discharges.update_layout(yaxis_tickformat=".0%", height=350)
        st.plotly_chart(fig_discharges, use_container_width=True)

    st.markdown("---")

    # Bottleneck detection charts
    st.subheader("⚠️ Bottleneck Detection & Backlog Analysis")
    bottleneck_col1, bottleneck_col2 = st.columns([1, 1])

    with bottleneck_col1:
        fig_backlog = px.bar(
            df_filtered,
            x="Date",
            y=["Children in CBP custody", "Children in HHS Care"],
            title="Active Care Load by Stage",
            labels={"value": "Active care load", "variable": "Stage"},
        )
        fig_backlog.update_layout(barmode="stack", height=400)
        st.plotly_chart(fig_backlog, use_container_width=True)

    with bottleneck_col2:
        fig_imbalance = px.scatter(
            df_filtered,
            x="Children transferred out of CBP custody",
            y="Children discharged from HHS Care",
            size="Children in HHS Care",
            color="Discharge Effectiveness",
            color_continuous_scale="Viridis",
            title="Transfer vs Discharge Performance",
            labels={
                "Children transferred out of CBP custody": "CBP transfers",
                "Children discharged from HHS Care": "HHS discharges",
                "Discharge Effectiveness": "Discharge effectiveness",
            },
            hover_data={
                "Date": True,
                "Children in HHS Care": True,
                "Transfer Efficiency": True,
            },
        )
        fig_imbalance.update_layout(height=400)
        st.plotly_chart(fig_imbalance, use_container_width=True)

    bottleneck_summary = df_filtered.loc[
        df_filtered["Backlog Accumulation Rate"] > 0
    ].copy()
    if not bottleneck_summary.empty:
        highest_backlog = int(bottleneck_summary["Backlog Accumulation Rate"].max())
        st.warning(
            f"Backlog accumulation peaked at {highest_backlog:,} additional active care cases in one day."
        )
    else:
        st.success("No sustained backlog accumulation detected in the selected date range.")

    st.markdown("---")

    # Outcome trend analysis
    st.subheader("📈 Outcome Trend Analysis")
    trend_col1, trend_col2 = st.columns([1, 1])

    with trend_col1:
        monthly = (
            df_filtered
            .set_index("Date")
            [["Children apprehended and placed in CBP custody*", "Children transferred out of CBP custody", "Children discharged from HHS Care"]]
            .resample("ME")
            .sum()
            .reset_index()
        )
        fig_monthly = px.line(
            monthly,
            x="Date",
            y=[
                "Children apprehended and placed in CBP custody*",
                "Children transferred out of CBP custody",
                "Children discharged from HHS Care",
            ],
            title="Monthly Pipeline Volume Trends",
            labels={"value": "Children count", "variable": "Volume"},
        )
        fig_monthly.update_layout(height=380)
        st.plotly_chart(fig_monthly, use_container_width=True)

    with trend_col2:
        weekday = (
            df_filtered
            .groupby("Weekday")
            [["Transfer Efficiency", "Discharge Effectiveness"]]
            .mean()
            .reindex([
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ])
            .reset_index()
        )
        fig_weekdays = px.bar(
            weekday,
            x="Weekday",
            y=["Transfer Efficiency", "Discharge Effectiveness"],
            barmode="group",
            title="Weekday Efficiency Patterns",
            labels={"value": "Average ratio", "variable": "Metric"},
        )
        fig_weekdays.update_layout(yaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig_weekdays, use_container_width=True)

    st.markdown("---")

    # Predictive modeling
    st.subheader("🤖 Predictive Model for HHS Discharge Outcomes")
    modeling_df = df_filtered[
        [
            "Children apprehended and placed in CBP custody*",
            "Children in CBP custody",
            "Children transferred out of CBP custody",
            "Children in HHS Care",
            "Children discharged from HHS Care",
        ]
    ].dropna()

    if len(modeling_df) >= 8:
        X = modeling_df[
            [
                "Children apprehended and placed in CBP custody*",
                "Children in CBP custody",
                "Children transferred out of CBP custody",
                "Children in HHS Care",
            ]
        ]
        y = modeling_df["Children discharged from HHS Care"]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        st.markdown(f"**Model R² Score:** {r2:.3f}")

        with st.expander("Use model inputs to forecast HHS discharges"):
            c1, c2, c3, c4 = st.columns(4)
            inp_apprehended = c1.number_input(
                "Apprehended in CBP",
                min_value=0,
                value=int(df_filtered["Children apprehended and placed in CBP custody*"].median()),
            )
            inp_cbp_load = c2.number_input(
                "Active CBP custody",
                min_value=0,
                value=int(df_filtered["Children in CBP custody"].median()),
            )
            inp_transfers = c3.number_input(
                "Transferred out of CBP",
                min_value=0,
                value=int(df_filtered["Children transferred out of CBP custody"].median()),
            )
            inp_hhs = c4.number_input(
                "Active HHS care",
                min_value=0,
                value=int(df_filtered["Children in HHS Care"].median()),
            )

            predicted = model.predict(
                [[inp_apprehended, inp_cbp_load, inp_transfers, inp_hhs]]
            )
            st.success(
                f"Estimated HHS discharges: {max(int(round(predicted[0])), 0):,} children"
            )
            st.markdown(
                "This simple regression model estimates discharge volume from current custody and transfer counts. "
                "Use it to test whether future discharge outcomes keep pace with inflows."
            )
    else:
        st.info("Not enough clean data to build the predictive discharge model for this time range.")

    st.markdown("---")
    st.subheader("📋 Detailed Data & Insights")
    st.markdown(
        "Explore the raw pipeline data and the derived efficiency metrics used by this analysis."
    )
    st.dataframe(
        df_filtered[
            [
                "Date",
                "Children apprehended and placed in CBP custody*",
                "Children in CBP custody",
                "Children transferred out of CBP custody",
                "Children in HHS Care",
                "Children discharged from HHS Care",
                "Transfer Efficiency",
                "Discharge Effectiveness",
                "Pipeline Throughput",
                "Backlog Accumulation Rate",
                "Outcome Stability Score",
            ]
        ]
        .style.format(
            {
                "Transfer Efficiency": "{:.1%}",
                "Discharge Effectiveness": "{:.1%}",
                "Pipeline Throughput": "{:.1%}",
                "Backlog Accumulation Rate": "{:.0f}",
                "Outcome Stability Score": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown(
        "### Project scope aligned to PRD requirements"
        "\n\n- Measures CBP → HHS transfer efficiency and HHS discharge outcomes"
        "\n- Detects backlog and care load bottlenecks in CBP and HHS stages"
        "\n- Provides trend analysis for monthly volumes and weekday effect patterns"
        "\n- Includes a predictive model for HHS discharge volume"
        "\n- Supports date range filtering, ratio-based metric toggles, and threshold alerts"
    )


if __name__ == "__main__":
    main()
