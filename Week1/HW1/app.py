import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")
st.title("Sleep Quality Predictor")
st.caption("Upload a CSV file to explore data and predict a numeric outcome.")
# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 3:
        st.error("Please upload a CSV with at least three numeric columns.")
        st.stop()

  # Data Visualization
    st.subheader("Data Visualization")

    chart_type = st.selectbox(
        "Choose a chart type",
        ["Line", "Scatter", "Histogram"]
    )

    x_col = st.selectbox("Choose X column", numeric_cols)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=x_col)

    else:
        y_col = st.selectbox("Choose Y column", numeric_cols)

        if chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)

        else:
            fig = px.scatter(df, x=x_col, y=y_col)

    st.plotly_chart(fig, use_container_width=True)
  # Linear Regression
    st.subheader("Linear Regression Predictor")

    x1 = st.selectbox("Choose first input (X1)", numeric_cols)
    x2 = st.selectbox("Choose second input (X2)", numeric_cols, index=1)
    target = st.selectbox("Choose target (Y)", numeric_cols, index=2)

    if len({x1, x2, target}) < 3:
        st.warning("Please choose three different columns.")
        st.stop()

    X = df[[x1, x2]]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    st.write("Training R2:", round(model.score(X, y), 3))

    # Make a prediction
    value1 = st.number_input(f"Enter {x1}", value=float(X[x1].mean()))
    value2 = st.number_input(f"Enter {x2}", value=float(X[x2].mean()))

    new_data = pd.DataFrame({
        x1: [value1],
        x2: [value2]
    })

    prediction = model.predict(new_data)[0]

    st.write("Prediction:", round(prediction, 2))
    
    st.caption(
        "This model uses two selected variables to predict a numeric outcome. "
        "Incorrect predictions may lead to poor decisions."
    )
