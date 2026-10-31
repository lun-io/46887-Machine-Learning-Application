import os

import pandas as pd
import plotly.express as px
import streamlit as st

from labeler import label_comment

# ---- Settings ----------------------------------------------------------
TITLE = "RidePulse"
LOW_RATING, HIGH_RATING = 2, 4
CSV_FILE = "feedback.csv"

COLORS = {
    "driver_behavior": "#6366F1",
    "wait_time": "#F59E0B",
    "pricing": "#8B5CF6",
    "vehicle_quality": "#06B6D4",
    "safety": "#DC2626",
    "positive": "#10B981",
    "unclear": "#9CA3AF"
}
# ------------------------------------------------------------------------

st.set_page_config(page_title=TITLE, page_icon="🚗", layout="wide")
st.title(TITLE)
st.caption("Share your ride experience. AI analyzes customer feedback to help improve our service.")



# Load the saved feedback, or start with an empty table
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=["comment", "rating", "label"])

tab_customer, tab_owner = st.tabs([
    "Customer: Leave Feedback",
    "Owner: Analyze Feedback"
])


# ---- Customer tab: submit ride feedback -------------------------------
with tab_customer:
    st.subheader("How was your ride?")

    with st.form("feedback_form", clear_on_submit=True):
        comment = st.text_area(
            "Tell us about your ride experience"
        )
        stars = st.feedback("stars")
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        if not comment.strip() and stars is None:
            st.error("Please provide a comment or a star rating.")
        else:
            # Only classify when a comment is provided
            if comment.strip():
                label = label_comment(comment.strip())
            else:
                label = None

            # Streamlit stars range from 0 to 4
            rating = stars + 1 if stars is not None else None

            new_row = pd.DataFrame([{
                "comment": comment.strip(),
                "rating": rating,
                "label": label
            }])

            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)

            st.success("Thank you for sharing your ride experience!")

            if label is not None:
                st.caption(f"AI feedback category: {label}")



# ---- Owner tab: analyze ride feedback ---------------------------------
with tab_owner:
    if df.empty:
        st.info("No feedback yet. Add the first ride review.")
    else:
        # Prepare ratings for analysis
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

        has_comment = df["comment"].fillna("").str.strip() != ""
        has_rating = df["rating"].notna()

        negative_labels = [
            "driver_behavior",
            "wait_time",
            "pricing",
            "vehicle_quality",
            "safety"
        ]

        # Flag comments that need manual review
        flag = has_comment & (
            (df["label"] == "unclear")
            | (df["label"] == "safety")
            | ((df["label"] == "positive")
               & has_rating
               & (df["rating"] <= LOW_RATING))
            | ((df["label"].isin(negative_labels))
               & has_rating
               & (df["rating"] >= HIGH_RATING))
        )

        # ---- Key Metrics ----------------------------------------------
        st.subheader("Ride Experience Overview")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Total Feedback", len(df))

        avg_rating = df["rating"].mean()
        m2.metric(
            "Average Rating",
            f"{avg_rating:.1f} / 5" if pd.notna(avg_rating) else "N/A"
        )

        m3.metric(
            "Safety Reports",
            int((df["label"] == "safety").sum())
        )

        m4.metric("Manual Review", int(flag.sum()))

        # ---- Charts ---------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feedback by Category")

            counts = (
                df["label"]
                .dropna()
                .value_counts()
                .rename_axis("label")
                .reset_index(name="count")
            )

            if not counts.empty:
                fig = px.bar(
                    counts,
                    x="label",
                    y="count",
                    color="label",
                    text="count",
                    color_discrete_map=COLORS
                )

                fig.update_layout(
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title="Number of Comments"
                )

                st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Average Rating by Category")

            avg = (
                df.dropna(subset=["label", "rating"])
                .groupby("label")["rating"]
                .mean()
                .round(1)
                .reset_index()
            )

            if not avg.empty:
                fig = px.bar(
                    avg,
                    x="label",
                    y="rating",
                    color="label",
                    text="rating",
                    color_discrete_map=COLORS,
                    range_y=[0, 5]
                )

                fig.update_layout(
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title="Average Stars"
                )

                st.plotly_chart(fig, width="stretch")

        # ---- Manual Review -------------------------------------------
        st.subheader("Comments Requiring Manual Review")

        st.caption(
            "Safety concerns, unclear comments, or feedback "
            "that conflicts with the star rating."
        )

        if flag.sum() > 0:
            st.dataframe(
                df[flag],
                hide_index=True,
                width="stretch"
            )
        else:
            st.success("No comments require manual review.")

        # ---- All Feedback --------------------------------------------
        with st.expander("All Feedback"):
            st.dataframe(df, hide_index=True, width="stretch")

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            file_name="feedback.csv"
        )

        st.caption(
            "On the free Streamlit Cloud, saved CSV data "
            "may disappear when the app restarts."
        )

