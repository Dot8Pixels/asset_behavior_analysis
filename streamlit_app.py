import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import kurtosis, norm, skew

# Streamlit Config
st.set_page_config(page_title="Asset Daily Return Analysis", layout="wide")

# Sidebar for settings
st.sidebar.header("Settings")

# Ticker input
default_ticker = "BTC-USD"  # Default ticker
ticker = st.sidebar.text_input(
    "Enter Asset Ticker (e.g., BTC-USD, NVDA)", default_ticker
)

# Start & end date input
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2014-09-17"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))


# Fetch data function
@st.cache_data
def get_asset_data(ticker, start, end) -> pd.DataFrame | None:
    try:
        crypto = yf.download(ticker, start=start, end=end)

        if crypto is None:
            raise ValueError("Found response from yf.download is None")
        if crypto.empty:
            raise ValueError("Found response from yf.download is empty dataframe")

        crypto.columns = crypto.columns.get_level_values(0)
        crypto["Return"] = crypto["Close"].pct_change()
        return crypto.dropna()

    except ValueError as ve:
        stack = traceback.format_stack()
        stack_str = "".join(stack)
        st.error(
            f"""❌ Invalid Ticker: '{ticker}'. Please enter a valid cryptocurrency symbol (e.g., BTC-USD, ETH-USD).
            Please check: https://finance.yahoo.com/lookup/\n\n
            ValueError Exception: {ve}
            Traceback: {stack_str}"""
        )
        return


# Load data
asset_df = get_asset_data(ticker, start_date, end_date)


if asset_df is not None:
    # Compute Metrics
    asset_df["Mean_30"] = asset_df["Return"].rolling(30).mean()
    asset_df["SD_30"] = asset_df["Return"].rolling(30).std()
    asset_df["+1SD"] = asset_df["Mean_30"] + asset_df["SD_30"]
    asset_df["-1SD"] = asset_df["Mean_30"] - asset_df["SD_30"]
    asset_df["+2SD"] = asset_df["Mean_30"] + 2 * asset_df["SD_30"]
    asset_df["-2SD"] = asset_df["Mean_30"] - 2 * asset_df["SD_30"]

    mean_return = asset_df["Return"].mean() * 100
    median_return = asset_df["Return"].median() * 100
    std_dev = asset_df["Return"].std() * 100
    max_return = asset_df["Return"].max() * 100
    min_return = asset_df["Return"].min() * 100
    one_sd_pos, one_sd_neg = mean_return + std_dev, mean_return - std_dev
    two_sd_pos, two_sd_neg = mean_return + 2 * std_dev, mean_return - 2 * std_dev
    skewness = skew(asset_df["Return"])
    kurt = kurtosis(asset_df["Return"])

    # Latest Rolling Stats
    latest_stats = (
        asset_df[["Mean_30", "SD_30", "+1SD", "-1SD", "+2SD", "-2SD"]].dropna().iloc[-1]
        * 100
    )

    # Layout for plots and tables
    st.title(f"📊 {ticker} Daily Return Analysis")

    col1, col2 = st.columns((2, 1))

    with col1:
        st.subheader("📈 Daily Return with Rolling Mean and ±1/2 SD Bands")

        # Create the time series plot with Plotly
        fig = go.Figure()

        # Add daily returns
        fig.add_trace(
            go.Scatter(
                x=asset_df.index,
                y=asset_df["Return"],
                mode="lines",
                name="Daily Return",
                line=dict(color="rgba(0, 0, 255, 0.3)"),
            )
        )

        # Add 30-day mean
        fig.add_trace(
            go.Scatter(
                x=asset_df.index,
                y=asset_df["Mean_30"],
                mode="lines",
                name="30-Day Mean",
                line=dict(color="black"),
            )
        )

        # Add +1 SD line
        fig.add_trace(
            go.Scatter(
                x=asset_df.index,
                y=asset_df["+1SD"],
                mode="lines",
                name="+1 SD",
                line=dict(color="gold", dash="dash"),
            )
        )

        # Add -1 SD line
        fig.add_trace(
            go.Scatter(
                x=asset_df.index,
                y=asset_df["-1SD"],
                mode="lines",
                name="-1 SD",
                line=dict(color="gold", dash="dash"),
            )
        )

        # Add +2 SD line
        fig.add_trace(
            go.Scatter(
                x=asset_df.index,
                y=asset_df["+2SD"],
                mode="lines",
                name="+2 SD",
                line=dict(color="red", dash="dash"),
            )
        )

        # Add -2 SD line
        fig.add_trace(
            go.Scatter(
                x=asset_df.index,
                y=asset_df["-2SD"],
                mode="lines",
                name="-2 SD",
                line=dict(color="red", dash="dash"),
            )
        )

        # Update layout
        fig.update_layout(
            template="none",
            xaxis_title="Date",
            yaxis_title="Daily Return",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=400,
            margin=dict(l=60, r=60, t=60, b=60),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Daily Return Summary (%)")
        summary_df = pd.DataFrame(
            {
                "Value": [
                    mean_return,
                    median_return,
                    std_dev,
                    max_return,
                    min_return,
                    one_sd_pos,
                    one_sd_neg,
                    two_sd_pos,
                    two_sd_neg,
                    skewness,
                    kurt,
                ]
            },
            index=[
                "Mean Return (%)",
                "Median Return (%)",
                "Standard Deviation (%)",
                "Max Return (%)",
                "Min Return (%)",
                "+1 SD (%)",
                "-1 SD (%)",
                "+2 SD (%)",
                "-2 SD (%)",
                "Skewness",
                "Kurtosis",
            ],
        )
        summary_df.index.name = "Metric"
        st.dataframe(summary_df)

    # Distribution Plot with Plotly
    st.subheader("📊 Distribution: Daily Returns vs Normal Distribution")

    # Create histogram with Plotly
    histogram_fig = go.Figure()

    # Add histogram
    histogram_fig.add_trace(
        go.Histogram(
            x=asset_df["Return"],
            nbinsx=100,
            opacity=0.6,
            name="Observed Return",
            marker_color="lightblue",
            histnorm="probability density",
        )
    )

    # Generate normal distribution curve
    returns = asset_df["Return"]
    x_range = np.linspace(returns.min(), returns.max(), 100)
    y_range = norm.pdf(x_range, returns.mean(), returns.std())

    # Add normal distribution line
    histogram_fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="Normal Distribution",
            line=dict(color="red"),
        )
    )

    # Add vertical lines for mean and standard deviations
    histogram_fig.add_vline(
        x=returns.mean(),
        line_dash="dash",
        line_color="black",
        annotation_text="Mean",
        annotation_position="top right",
    )

    histogram_fig.add_vline(
        x=returns.mean() + returns.std(),
        line_dash="dash",
        line_color="gold",
        annotation_text="+1 SD",
        annotation_position="top right",
    )

    histogram_fig.add_vline(
        x=returns.mean() - returns.std(),
        line_dash="dash",
        line_color="gold",
        annotation_text="-1 SD",
        annotation_position="top right",
    )

    histogram_fig.add_vline(
        x=returns.mean() + 2 * returns.std(),
        line_dash="dash",
        line_color="red",
        annotation_text="+2 SD",
        annotation_position="top right",
    )

    histogram_fig.add_vline(
        x=returns.mean() - 2 * returns.std(),
        line_dash="dash",
        line_color="red",
        annotation_text="-2 SD",
        annotation_position="top right",
    )

    # Update layout
    histogram_fig.update_layout(
        template="none",
        xaxis_title="Daily Return",
        yaxis_title="Density",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        # margin=dict(l=60, r=60, t=60, b=60),
        hovermode="x unified",
    )

    st.plotly_chart(histogram_fig, use_container_width=True)

    # Latest Rolling Statistics
    st.subheader("📉 Latest 30-Day Rolling Stats (%)")
    st.dataframe(latest_stats.to_frame(name="Latest Value (%)"))

    # Compute additional metrics for report
    total_days = len(asset_df)
    up_days = (asset_df["Return"] > 0).sum()
    down_days = (asset_df["Return"] < 0).sum()
    up_days_pct = (up_days / total_days) * 100
    down_days_pct = (down_days / total_days) * 100
    mean_up = asset_df.loc[asset_df["Return"] > 0, "Return"].mean() * 100
    mean_down = asset_df.loc[asset_df["Return"] < 0, "Return"].mean() * 100

    # Extreme movement calculations
    threshold_up = two_sd_pos  # More than +2 standard deviations
    threshold_down = two_sd_neg  # Less than -2 standard deviations
    extreme_up_days = (asset_df["Return"] * 100 > threshold_up).sum()
    extreme_down_days = (asset_df["Return"] * 100 < threshold_down).sum()
    extreme_up_pct = (extreme_up_days / total_days) * 100
    extreme_down_pct = (extreme_down_days / total_days) * 100

    # Report Section
    st.subheader("📊 Trading Statistics Report")

    # # Create a visual summary of up/down days
    # updown_fig = go.Figure()

    # # Add pie chart for up/down days
    # updown_fig.add_trace(
    #     go.Pie(
    #         labels=["Positive Days", "Negative Days"],
    #         values=[up_days, down_days],
    #         marker_colors=["#66BB6A", "#EF5350"],
    #         hole=0.4,
    #         textinfo="percent+label",
    #         hoverinfo="label+percent+value",
    #     )
    # )

    # updown_fig.update_layout(
    #     title="📈 Distribution of Positive vs Negative Days",
    #     height=300,
    #     margin=dict(l=20, r=20, t=50, b=20),
    # )

    # st.plotly_chart(updown_fig, use_container_width=True)

    st.markdown(f"""
    ### 📈 Daily Performance Summary
    - 📅 **Total Trading Days:** {total_days} days
    - ✅ **Days Closed Positive:** {up_days} days ({up_days_pct:.2f}%)
    - 📉 **Days Closed Negative:** {down_days} days ({down_days_pct:.2f}%)
    - 📊 **Average Gain on Up Days:** {mean_up:.2f}%
    - 📉 **Average Loss on Down Days:** {mean_down:.2f}%

    ### 🔍 Extreme Price Movements
    - 🔺 **Total Strong Up Days (> +{threshold_up:.2f}%):** {extreme_up_days} days ({extreme_up_pct:.2f}%)
    - 🔻 **Total Strong Down Days (< -{threshold_down:.2f}%):** {extreme_down_days} days ({extreme_down_pct:.2f}%)

    ---

    ### 📊 Daily Return Summary (%)
    - 🟢 **Mean Return:** {mean_return:.2f}%
    → Average daily return over all days.
    - 🟢 **Median Return:** {median_return:.2f}%
    → The midpoint of daily returns, meaning 50% of the time {ticker} closes above this value.
    - 🟠 **Standard Deviation:** {std_dev:.2f}%
    → Measures price volatility, most returns fall within ±{std_dev:.2f}%.
    - 🟥 **Max Daily Return:** {max_return:.2f}%
    → Largest single-day gain recorded.
    - 🟥 **Min Daily Return:** {min_return:.2f}%
    → Largest single-day loss recorded.

    ### 📊 Standard Deviation Ranges
    - 🔵 **±1 SD Range:** ({one_sd_pos:.2f}%, {one_sd_neg:.2f}%)
    → 68% of daily moves stay within this range.
    - 🔵 **±2 SD Range:** ({two_sd_pos:.2f}%, {two_sd_neg:.2f}%)
    → 95% of daily moves stay within this range.

    ### 📉 Statistical Distribution Insights
    - 🌀 **Skewness:** {skewness:.2f}
    → Negative skew means the asset has more extreme negative moves, while positive skew means more extreme positive moves.
    - 🌀 **Kurtosis:** {kurt:.2f}
    → A high kurtosis indicates that returns experience more frequent extreme movements than a normal distribution.
    """)
