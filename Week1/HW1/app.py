
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Page settings
st.set_page_config(
    page_title="CSV Data Explorer & Predictor",
    layout="wide"
)

st.title("CSV Data Explorer & Predictor")

st.caption(
    "Upload a CSV dataset, explore your data, "
    "create visualizations, and make predictions using linear regression."
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type="csv"
)

if uploaded_file is not None:

    # Read data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    # Find numeric columns
    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    # Exclude common identifier columns
    numeric_cols = [
        col for col in numeric_cols
        if col.lower().strip() not in
        ["id", "person id", "customer id", "record id"]
        and not col.lower().strip().endswith("_id")
    ]

    if len(numeric_cols) < 3:
        st.error(
            "Please upload a CSV file with at least "
            "three numeric columns."
        )
        st.stop()

    # ---------------------------------
    # Data Visualization
    # ---------------------------------

    st.subheader("Data Visualization")

    chart_type = st.selectbox(
        "Choose a chart type",
        ["Line", "Scatter", "Histogram"]
    )

    x_col = st.selectbox(
        "Choose X column",
        numeric_cols
    )

    if chart_type == "Histogram":

        fig = px.histogram(
            df,
            x=x_col,
            title=f"Distribution of {x_col}"
        )

    else:

        y_col = st.selectbox(
            "Choose Y column",
            numeric_cols,
            index=1
        )

        if chart_type == "Line":

            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )

        else:

            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=f"{x_col} vs {y_col}"
            )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------
    # Linear Regression
    # ---------------------------------

    st.subheader("Linear Regression Predictor")

    st.write(
        "Choose two numeric input variables (X1 and X2) "
        "and one numeric target variable (Y)."
    )

    # Default selections
    default_x1 = (
        numeric_cols.index("Physical Activity Level")
        if "Physical Activity Level" in numeric_cols
        else 0
    )

    default_x2 = (
        numeric_cols.index("Stress Level")
        if "Stress Level" in numeric_cols
        else 1
    )

    default_y = (
        numeric_cols.index("Quality of Sleep")
        if "Quality of Sleep" in numeric_cols
        else 2
    )

    # Select input variables
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

    # Select target variable
    target = st.selectbox(
        "Choose target (Y)",
        numeric_cols,
        index=default_y
    )

    # Check for duplicate columns
    if len({x1, x2, target}) < 3:

        st.warning(
            "Please choose three different columns "
            "for X1, X2, and Y."
        )
        st.stop()

    # Prepare model data
    model_df = df[[x1, x2, target]].dropna()

    if len(model_df) < 3:
        st.error(
            "Not enough valid rows to train the model."
        )
        st.stop()

    X = model_df[[x1, x2]]
    y = model_df[target]

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Show model performance
    r2 = model.score(X, y)

    st.write(
        "Training R²:",
        round(r2, 3)
    )

    st.caption(
        "Training R² measures how well the model fits "
        "the training data. It does not measure "
        "performance on unseen data."
    )

    # ---------------------------------
    # Prediction
    # ---------------------------------

    st.write("Enter new values to make a prediction:")

    value1 = st.number_input(
        f"Enter {x1}",
        value=float(X[x1].mean())
    )

    value2 = st.number_input(
        f"Enter {x2}",
        value=float(X[x2].mean())
    )

    # Use the same column names and order as training
    new_data = pd.DataFrame({
        x1: [value1],
        x2: [value2]
    })

    # Make prediction
    prediction = model.predict(new_data)[0]

    st.metric(
        label=f"Predicted {target}",
        value=f"{prediction:.2f}"
    )

    # ---------------------------------
    # Model Explanation
    # ---------------------------------

    st.subheader("About This Model")

    st.write(
        "This linear regression model predicts a selected "
        "numeric target using two input variables. "
        "Incorrect predictions may lead to misleading "
        "conclusions or poor decisions."
    )

    st.info(
        "This app is for educational purposes. "
        "Predictions depend on the uploaded dataset "
        "and the variables selected."
    )
