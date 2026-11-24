import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Transaction Monitor', layout='wide')

# ---- load the trained model (made in the Colab notebook) ----
bundle = joblib.load(APP_DIR / 'model.joblib')
model = bundle['model']
history_scores = bundle['history_scores']
features = bundle['features']


def score(amount, typical, last_hour, gap, age):
    """Turn one transaction into the five features and return its anomaly score."""
    row = pd.DataFrame([{
        'log_amount': np.log1p(amount),
        'amount_ratio': amount / typical,
        'transactions_last_hour': last_hour,
        'log_gap': np.log1p(gap),
        'log_age': np.log1p(age),
    }])[features]
    return -model.score_samples(row)[0]


# ---- sidebar: the review budget ----
st.title('Transaction Monitor')
st.write('Each new transaction is scored by Isolation Forest. '
         'If the score is above the cutoff, it goes to review.')

pct = st.sidebar.slider('Send this share of transactions to review (%)', 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f'Score cutoff: **{cutoff:.3f}**')
st.sidebar.caption('The cutoff is set from past transactions: '
                   'a transaction scoring above it is more unusual than the top '
                   f'{pct}% of history.')

tab1, tab2 = st.tabs(['Check one transaction', 'Live stream'])

# ---- tab 1: type in one transaction ----
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input('Amount ($)', min_value=1.0, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=1.0, value=50.0)
    last_hour = c1.number_input('Transactions in the last hour', min_value=0, value=1)
    gap = c2.number_input('Minutes since previous transaction', min_value=0.0, value=120.0)
    age = c2.number_input('Account age (days)', min_value=0, value=700)

    if st.button('Score this transaction'):
        s = score(amount, typical, last_hour, gap, age)
        if s > cutoff:
            st.error(f'REVIEW  (score {s:.3f}, cutoff {cutoff:.3f})')
        else:
            st.success(f'OK  (score {s:.3f}, cutoff {cutoff:.3f})')
        share = (history_scores < s).mean() * 100
        st.write(f'This is more unusual than {share:.0f}% of past transactions.')

# ---- tab 2: transactions arriving one by one ----
with tab2:
    st.write(
        "Start the stream to check transactions one at a time. "
        "Changing speed preserves progress and review results."
    )

    speed = st.select_slider(
        "Speed",
        options=["Slow", "Medium", "Fast"],
        value="Medium",
    )
    interval = {"Slow": 0.5, "Medium": 0.35, "Fast": 0.17}[speed]

    # Keep the current run across widget changes.
    if "monitor_run" not in st.session_state:
        st.session_state.monitor_run = None

    run = st.session_state.monitor_run
    status = run["status"] if run is not None else "idle"

    c1, c2, c3, c4 = st.columns(4)

    if c1.button("Start the stream", disabled=run is not None):
        try:
            stream = pd.read_csv(APP_DIR / "new_transactions.csv")
            if stream.empty:
                st.error("No transactions found. Check new_transactions.csv.")
            else:
                st.session_state.monitor_run = {
                    "data": stream,
                    "next_index": 0,
                    "flagged": [],
                    "last": None,
                    "status": "running",
                    "budget": pct,
                    "cutoff": float(cutoff),
                    "error": None,
                }
                st.rerun()
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            st.error(
                "The transaction file could not be loaded. "
                "Check new_transactions.csv, then click Start again."
            )

    if c2.button("Pause", disabled=status != "running"):
        st.session_state.monitor_run = {**run, "status": "paused"}
        st.rerun()

    if c3.button("Resume", disabled=status != "paused"):
        st.session_state.monitor_run = {
            **run,
            "status": "running",
            "error": None,
        }
        st.rerun()

    if c4.button(
        "Reset for a new run",
        disabled=run is None or status == "running",
        help="Clears this run's progress and results. Download them first.",
    ):
        st.session_state.monitor_run = None
        st.rerun()

    st.caption(
        "Progress is saved within this browser session. "
        "Download results before resetting or refreshing the page."
    )

    # Process one transaction per update, rather than one long loop.
    @st.fragment(
        run_every=interval if status == "running" else None
    )
    def update_stream():
        current = st.session_state.monitor_run

        if current is None:
            st.info("Choose a budget and speed, then click Start the stream.")
            return

        data = current["data"]
        total = len(data)

        if current["status"] == "running":
            i = current["next_index"]
            row = data.iloc[i]

            try:
                s = score(
                    row["amount"],
                    row["customer_typical_amount"],
                    row["transactions_last_hour"],
                    row["minutes_since_previous"],
                    row["account_age_days"],
                )
                if not np.isfinite(s):
                    raise ValueError("Non-finite score")
            except Exception:
                st.session_state.monitor_run = {
                    **current,
                    "status": "paused",
                    "error": (
                        f"Transaction at row {i + 1} could not be scored. "
                        "Earlier results are preserved. No result was "
                        "recorded for this transaction. Retry with Resume; "
                        "if the problem persists, check the source data "
                        "and model setup, then reset for a new run."
                    ),
                }
                st.rerun()

            needs_check = s > current["cutoff"]
            flagged = list(current["flagged"])

            if needs_check:
                flagged.append({
                    "id": row["transaction_id"],
                    "amount": row["amount"],
                    "typical": row["customer_typical_amount"],
                    "last_hour": row["transactions_last_hour"],
                    "score": round(s, 3),
                })

            # Save the result and next position together before displaying.
            current = {
                **current,
                "next_index": i + 1,
                "flagged": flagged,
                "last": {
                    "id": row["transaction_id"],
                    "amount": row["amount"],
                    "score": s,
                    "review": needs_check,
                },
                "status": "done" if i + 1 == total else "running",
            }
            st.session_state.monitor_run = current

            if current["status"] == "done":
                st.rerun()

        st.caption(
            f"This run uses a {current['budget']}% review budget "
            f"and cutoff {current['cutoff']:.3f}."
        )

        if pct != current["budget"]:
            st.warning(
                "The sidebar budget has changed. This run keeps its "
                "original cutoff so results remain consistent. "
                "To use the new budget, pause if needed, download "
                "your results, reset, and start a new run."
            )

        if current["error"]:
            st.error(current["error"])
        elif current["status"] == "paused":
            st.warning(
                "Paused — screening is incomplete. Progress and review "
                "results are preserved. Click Resume to continue."
            )
        elif current["status"] == "running":
            st.info(
                "Screening in progress. You may change speed safely "
                "or click Pause. The results below are partial."
            )

        last = current["last"]
        if last is not None:
            message = (
                f"{last['id']}: ${last['amount']:.2f} → "
                f"{'REVIEW' if last['review'] else 'ok'} "
                f"(score {last['score']:.3f})"
            )
            if last["review"]:
                st.error(message)
            else:
                st.info(message)

        st.write(
            f"Received **{current['next_index']}** of {total}. "
            f"Sent to review: **{len(current['flagged'])}**"
        )

        st.subheader("Sent to review")
        if current["flagged"]:
            review_table = pd.DataFrame(current["flagged"])
            st.dataframe(review_table, hide_index=True)
            if current["status"] != "running":
                st.download_button(
                    "Download current review results",
                    data=review_table.to_csv(index=False).encode("utf-8"),
                    file_name="review_results.csv",
                    mime="text/csv",
                )
        else:
            st.write("No transactions flagged so far.")

        if current["status"] == "done":
            st.success(
                f"Screening complete: all {total} transactions processed. "
                f"{len(current['flagged'])} transactions were flagged "
                "for review. Results remain available until you reset."
            )

    update_stream()
