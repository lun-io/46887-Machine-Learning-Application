import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Transaction Monitor', layout='wide')

# ---- load the trained model (made in the Colab notebook) ----
bundle = joblib.load('model.joblib')
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
    st.write('Press the button. The app receives the new transactions one at a time '
             'and scores each one as it arrives.')
    speed = st.select_slider('Speed', options=['Slow', 'Medium', 'Fast'], value='Medium')
    pause = {'Slow': 0.5, 'Medium': 0.15, 'Fast': 0.02}[speed]

    if st.button('Start the stream'):
        stream = pd.read_csv('new_transactions.csv')
        box = st.empty()
        counts = st.empty()
        st.subheader('Sent to review')
        table = st.empty()
        flagged = []

        for i, r in stream.iterrows():
            s = score(r['amount'], r['customer_typical_amount'],
                      r['transactions_last_hour'], r['minutes_since_previous'],
                      r['account_age_days'])
            msg = f"{r['transaction_id']}: ${r['amount']:.2f}  score {s:.3f}"
            if s > cutoff:
                flagged.append({'id': r['transaction_id'], 'amount': r['amount'],
                                'typical': r['customer_typical_amount'],
                                'last_hour': r['transactions_last_hour'],
                                'score': round(s, 3)})
                box.error(msg + '  ->  REVIEW')
            else:
                box.info(msg + '  ->  ok')
            counts.write(f'Received **{i + 1}** of {len(stream)}.  '
                         f'Sent to review: **{len(flagged)}**')
            if flagged:
                table.dataframe(pd.DataFrame(flagged), hide_index=True)
            time.sleep(pause)

        st.success(f'Done. {len(flagged)} transactions were sent to review.')

        # if the file has an answer key, check how we did
        if 'needs_review' in stream.columns:
            answer = stream.set_index('transaction_id')['needs_review']
            hits = sum(answer[f['id']] for f in flagged)
            st.write(f'{hits} of the {len(flagged)} flagged transactions really needed '
                     f'review. In total, {answer.sum()} transactions needed review.')
