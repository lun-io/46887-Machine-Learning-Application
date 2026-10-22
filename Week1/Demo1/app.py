import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Data Explorer", layout="wide")
st.title("Data Explorer")
st.caption("Upload a CSV file, pick a column, and see it over time.")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("This file has no numeric columns to plot.")
        st.stop()

    date_col = st.selectbox("Date column", df.columns)
    variable = st.selectbox("Column to plot", numeric_cols)

    fig = px.line(df, x=date_col, y=variable, title=f"{variable} over time")
    st.plotly_chart(fig, width="stretch")

    with st.expander("See raw data"):
        st.dataframe(df, hide_index=True, width="stretch")
else:
    st.info("Upload a CSV file to get started. It needs a date column and at least one numeric column.")
