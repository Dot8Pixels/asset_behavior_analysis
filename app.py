import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import yfinance as yf
from scipy.stats import kurtosis, norm, skew

# Streamlit Config
st.set_page_config(page_title="BTC Daily Return Analysis", layout="wide")

# Sidebar Input
st.sidebar.header("Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2014-09-17"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))


# Load BTC Data
@st.cache_data
def get_btc_data(start, end):
    btc = yf.download("BTC-USD", start=start, end=end)
    btc["Return"] = btc["Close"].pct_change()
    return btc.dropna()


btc = get_btc_data(start_date, end_date)

# Compute Metrics
btc["Mean_30"] = btc["Return"].rolling(30).mean()
btc["SD_30"] = btc["Return"].rolling(30).std()
btc["+1SD"] = btc["Mean_30"] + btc["SD_30"]
btc["-1SD"] = btc["Mean_30"] - btc["SD_30"]
btc["+2SD"] = btc["Mean_30"] + 2 * btc["SD_30"]
btc["-2SD"] = btc["Mean_30"] - 2 * btc["SD_30"]

mean_return = btc["Return"].mean() * 100
median_return = btc["Return"].median() * 100
std_dev = btc["Return"].std() * 100
max_return = btc["Return"].max() * 100
min_return = btc["Return"].min() * 100
one_sd_pos, one_sd_neg = mean_return + std_dev, mean_return - std_dev
two_sd_pos, two_sd_neg = mean_return + 2 * std_dev, mean_return - 2 * std_dev
skewness = skew(btc["Return"])
kurt = kurtosis(btc["Return"])

# Latest Rolling Stats
latest_stats = (
    btc[["Mean_30", "SD_30", "+1SD", "-1SD", "+2SD", "-2SD"]].dropna().iloc[-1] * 100
)

# Layout for plots and tables
st.title("📊 BTC Daily Return Analysis")

col1, col2 = st.columns((2, 1))

with col1:
    st.subheader("📈 BTC Daily Return with Rolling Mean and ±1/2 SD Bands")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(btc.index, btc["Return"], label="Daily Return", alpha=0.3, color="blue")
    ax.plot(btc.index, btc["Mean_30"], label="30-Day Mean", color="black")
    ax.plot(btc.index, btc["+1SD"], linestyle="dashed", color="gold", label="+1 SD")
    ax.plot(btc.index, btc["-1SD"], linestyle="dashed", color="gold", label="-1 SD")
    ax.plot(btc.index, btc["+2SD"], linestyle="dashed", color="red", label="+2 SD")
    ax.plot(btc.index, btc["-2SD"], linestyle="dashed", color="red", label="-2 SD")
    ax.legend()
    ax.set_ylabel("Daily Return")
    ax.set_xlabel("Date")
    st.pyplot(fig)

with col2:
    st.subheader("📊 BTC Daily Return Summary (%)")
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
    st.dataframe(summary_df)

# Histogram Plot
st.subheader("📊 Distribution of BTC Daily Returns vs Normal Distribution")

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(
    btc["Return"],
    bins=100,
    kde=False,
    color="lightblue",
    stat="density",
    label="Observed Return",
)
xmin, xmax = plt.xlim()
x_vals = np.linspace(xmin, xmax, 100)
ax.plot(
    x_vals,
    norm.pdf(x_vals, btc["Return"].mean(), btc["Return"].std()),
    color="red",
    label="Normal Distribution",
)
ax.axvline(btc["Return"].mean(), color="black", linestyle="dashed", label="Mean")
ax.axvline(
    btc["Return"].mean() + btc["Return"].std(),
    color="gold",
    linestyle="dashed",
    label="+1 SD",
)
ax.axvline(
    btc["Return"].mean() - btc["Return"].std(),
    color="gold",
    linestyle="dashed",
    label="-1 SD",
)
ax.axvline(
    btc["Return"].mean() + 2 * btc["Return"].std(),
    color="red",
    linestyle="dashed",
    label="+2 SD",
)
ax.axvline(
    btc["Return"].mean() - 2 * btc["Return"].std(),
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
# st.dataframe(pd.DataFrame(latest_stats, columns=["Latest Value (%)"]))
st.dataframe(latest_stats.to_frame(name="Latest Value (%)"))
