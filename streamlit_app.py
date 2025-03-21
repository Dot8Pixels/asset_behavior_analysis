import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            asset_df.index,
            asset_df["Return"],
            label="Daily Return",
            alpha=0.3,
            color="blue",
        )
        ax.plot(asset_df.index, asset_df["Mean_30"], label="30-Day Mean", color="black")
        ax.plot(
            asset_df.index,
            asset_df["+1SD"],
            linestyle="dashed",
            color="gold",
            label="+1 SD",
        )
        ax.plot(
            asset_df.index,
            asset_df["-1SD"],
            linestyle="dashed",
            color="gold",
            label="-1 SD",
        )
        ax.plot(
            asset_df.index,
            asset_df["+2SD"],
            linestyle="dashed",
            color="red",
            label="+2 SD",
        )
        ax.plot(
            asset_df.index,
            asset_df["-2SD"],
            linestyle="dashed",
            color="red",
            label="-2 SD",
        )
        ax.legend()
        ax.set_ylabel("Daily Return")
        ax.set_xlabel("Date")
        st.pyplot(fig)

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

    # Histogram Plot
    st.subheader("📊 Distribution: Daily Returns vs Normal Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(
        asset_df[["Return"]],
        bins=50,  # You can modify bins dynamically
        kde=False,
        color="lightblue",
        stat="density",
        label="Observed Return",
    )
    xmin, xmax = plt.xlim()
    x_vals = np.linspace(xmin, xmax, 100)
    ax.plot(
        x_vals,
        norm.pdf(x_vals, asset_df["Return"].mean(), asset_df["Return"].std()),
        color="red",
        label="Normal Distribution",
    )
    ax.axvline(
        asset_df["Return"].mean(), color="black", linestyle="dashed", label="Mean"
    )
    ax.axvline(
        asset_df["Return"].mean() + asset_df["Return"].std(),
        color="gold",
        linestyle="dashed",
        label="+1 SD",
    )
    ax.axvline(
        asset_df["Return"].mean() - asset_df["Return"].std(),
        color="gold",
        linestyle="dashed",
        label="-1 SD",
    )
    ax.axvline(
        asset_df["Return"].mean() + 2 * asset_df["Return"].std(),
        color="red",
        linestyle="dashed",
        label="+2 SD",
    )
    ax.axvline(
        asset_df["Return"].mean() - 2 * asset_df["Return"].std(),
        color="red",
        linestyle="dashed",
        label="-2 SD",
    )
    ax.legend()
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Density")
    st.pyplot(fig)

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
