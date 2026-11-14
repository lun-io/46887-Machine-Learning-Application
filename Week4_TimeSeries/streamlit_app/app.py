import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

st.title("Sales Forecast")
st.write("Pick some data, choose how far ahead to look, and get a forecast with an uncertainty band.")

# 1. Get the data
source = st.radio("Data", ["Sample data", "Upload my own CSV"])

from pathlib import Path
if source == "Sample data":
    csv_path = Path(__file__).resolve().parent.parent / "daily_sales.csv"
    data = pd.read_csv(csv_path)
else:
    file = st.file_uploader("Upload a CSV file", type="csv")
    if file is None:
        st.info("Upload a file to begin.")
        st.stop()
    data = pd.read_csv(file)

st.write("First rows of your data:")
st.dataframe(data.head())

# 2. Choices
date_col = st.selectbox("Which column has the dates?", data.columns)
value_col = st.selectbox("Which column has the numbers?", data.columns, index=len(data.columns) - 1)
days = st.slider("How many days to forecast?", 7, 90, 30)
use_holidays = st.checkbox("Include US holidays", value=True)

# 3. Clean the data
df = data[[date_col, value_col]].copy()
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna()

if len(df) < 30:
    st.error("Need at least 30 rows with a valid date and number. Check the column choices above.")
    st.stop()

# 4. Forecast
if st.button("Make forecast"):
    with st.spinner("Fitting the model..."):
        model = Prophet()
        if use_holidays:
            model.add_country_holidays(country_name="US")
        model.fit(df)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

    future_part = forecast.tail(days)

    # 5. Show it
    fig = go.Figure()
    fig.add_scatter(x=df["ds"], y=df["y"], name="history", line=dict(color="black"))
    fig.add_scatter(x=future_part["ds"], y=future_part["yhat_upper"], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=future_part["ds"], y=future_part["yhat_lower"], fill="tonexty",
                    fillcolor="rgba(196,18,48,0.2)", line=dict(width=0), name="uncertainty band")
    fig.add_scatter(x=future_part["ds"], y=future_part["yhat"], name="forecast", line=dict(color="#C41230"))
    fig.update_layout(title=f"Forecast for the next {days} days")
    st.plotly_chart(fig)

    st.write("The shaded band is a range of likely values, not a promise.")

    table = future_part[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    table.columns = ["date", "forecast", "low", "high"]
    table[["forecast", "low", "high"]] = table[["forecast", "low", "high"]].round(0)
    st.dataframe(table)

    st.download_button("Download forecast as CSV", table.to_csv(index=False), "forecast.csv")
