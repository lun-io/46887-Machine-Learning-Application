import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")

st.title("Sleep Quality Predictor")
st.caption("Upload a CSV file to explore data and predict a numeric outcome.")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Select numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Remove ID column
    numeric_cols = [col for col in numeric_cols if col != "Person ID"]

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

    # Default column selections
    default_x1 = (
        numeric_cols.index("Physical Activity Level")
        if "Physical Activity Level" in numeric_cols else 0
    )

    default_x2 = (
        numeric_cols.index("Stress Level")
        if "Stress Level" in numeric_cols else 1
    )

    default_y = (
        numeric_cols.index("Quality of Sleep")
        if "Quality of Sleep" in numeric_cols else 2
    )

    # Select predictors and target
    x1 = st.selectbox(
        "Choose first input (X1)",
        numeric_cols,
        index=default_x1
    )

    x2 = st.selectbox(
        "Choose second input (X2)",
        numeric_cols,
        index=default_x2
    )

    target = st.selectbox(
        "Choose target (Y)",
        numeric_cols,
        index=default_y
    )

    # Prevent selecting the same column
    if len({x1, x2, target}) < 3:
        st.warning("Please choose three different columns.")
        st.stop()

    # Prepare training data
    model_df = df[[x1, x2, target]].dropna()

    if len(model_df) < 3:
        st.error("Not enough valid rows to train the model.")
        st.stop()

    X = model_df[[x1, x2]]
    y = model_df[target]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Display training R2
    st.write("Training R2:", round(model.score(X, y), 3))

    # User inputs for prediction
    st.write("Enter new values to make a prediction:")

    value1 = st.number_input(
        f"Enter {x1}",
        value=float(X[x1].mean())
    )

    value2 = st.number_input(
        f"Enter {x2}",
        value=float(X[x2].mean())
    )

    # Prepare prediction data
    new_data = pd.DataFrame({
        x1: [value1],
        x2: [value2]
    })

    # Make prediction
    prediction = model.predict(new_data)[0]

    st.metric(
        label=f"Predicted {target}",
        value=round(prediction, 2)
    )

    # Model explanation
    st.caption(
        "This model uses two selected variables to predict a numeric outcome. "
        "Incorrect predictions may lead to poor decisions."
    )

