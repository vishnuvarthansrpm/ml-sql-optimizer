import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="SQL Optimizer App", page_icon="🧮", layout="centered")

st.markdown("""
# 🧮 SQL Optimizer App
Welcome to the SQL Optimizer!  
This app predicts the runtime of SQL queries based on their structure.
""")

st.markdown("""
*Enter the required features of your SQL query below and click **Predict Runtime**.  
You can download results after prediction. Logical validation, status and info messages, and a clean UI will guide you.*
""")

# --- Check for model file and load ---
MODEL_PATH = 'sql_optimizer_model.pkl'
model = None
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file `{MODEL_PATH}` not found. Please upload the trained model or add it to this folder.")
else:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")

# --- Define feature columns ---
feature_cols = [
    'rows_returned', 'has_group_by', 'num_joins', 'has_subquery',
    'query_type_AGGREGATE', 'query_type_GROUP BY', 'query_type_JOIN', 'query_type_SELECT'
]

# --- Input form ---
with st.form("predict_form"):
    st.subheader("🔽 Query Feature Inputs")
    rows_returned = st.number_input("Rows Returned", min_value=0, max_value=1_000_000, value=1000, step=100, help="Result rows expected from your query")
    has_group_by = st.checkbox("Has GROUP BY clause", value=False)
    num_joins = st.number_input("Number of JOINs", min_value=0, max_value=20, value=1, step=1)
    has_subquery = st.checkbox("Has Subquery", value=False)
    st.caption("Select your query type (pick one)")
    qtype = st.radio("Query Type", [
        "AGGREGATE", "GROUP BY", "JOIN", "SELECT"
    ], horizontal=True)

    # One-hot encoding for query type
    query_type = {
        'query_type_AGGREGATE': int(qtype == 'AGGREGATE'),
        'query_type_GROUP BY': int(qtype == 'GROUP BY'),
        'query_type_JOIN': int(qtype == 'JOIN'),
        'query_type_SELECT': int(qtype == 'SELECT')
    }

    # --- Validation & Info Boxes ---
    if rows_returned > 500_000:
        st.warning("⚠️ Caution: Returning more than 500,000 rows can be resource intensive.")
    if num_joins > 5:
        st.info("Note: For queries with more than 5 joins, execution can be slow on large databases.")

    submit = st.form_submit_button("Predict Runtime")

    result = None
    if submit:
        # Additional logical validation
        if rows_returned < 0 or num_joins < 0:
            st.error("Both 'Rows Returned' and 'Number of JOINs' must be non-negative.")
        elif model is None:
            st.stop()
        else:
            features = [
                rows_returned,
                int(has_group_by),
                num_joins,
                int(has_subquery),
                query_type['query_type_AGGREGATE'],
                query_type['query_type_GROUP BY'],
                query_type['query_type_JOIN'],
                query_type['query_type_SELECT']
            ]
            try:
                X = np.array(features).reshape(1, -1)
                pred = model.predict(X)
                pred_val = float(pred[0])
                st.success(f"**Predicted Query Runtime:** {pred_val:.4f} units")
                # Download result button
                st.download_button(
                    label="📥 Download Result",
                    data=f"Predicted Query Runtime: {pred_val:.4f} units\nInputs: {features}",
                    file_name="sql_runtime_prediction.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")

st.markdown("""
---
*Powered by Streamlit | ML-Driven SQL Prediction Demo*  
""")
