"""
app.py
======
FastAPI scoring + training service for fraud detection.
Deployed on Render. Called by Vercel (POST /score) and Render cron (POST /train).
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from dotenv import load_dotenv

from fraud_pipeline import load_from_supabase, MLPipeline

load_dotenv()

app = FastAPI()

SCORING_SECRET = os.getenv("SCORING_SECRET")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fraud_model.joblib")

# ── Feature query for scoring (no label, unfulfilled + unscored) ──────────────
QUERY_SCORING_BATCH = """
SELECT
    o.order_id,
    o.order_subtotal,
    o.shipping_fee,
    o.tax_amount,
    o.order_total,
    o.risk_score,
    o.promo_used,
    o.payment_method,
    o.device_type,
    o.ip_country,
    o.shipping_state,
    CASE WHEN o.billing_zip != o.shipping_zip THEN 1 ELSE 0 END  AS zip_mismatch,
    EXTRACT(HOUR FROM o.order_datetime::timestamp)               AS order_hour,
    EXTRACT(DOW  FROM o.order_datetime::timestamp)               AS order_dow,
    c.gender,
    c.customer_segment,
    c.loyalty_tier,
    c.is_active                                                  AS customer_is_active,
    DATE_PART('year', AGE(NOW(), c.birthdate::timestamp))::INT   AS customer_age,
    DATE_PART('day', o.order_datetime::timestamp
                   - c.created_at::timestamp) / 365.25          AS customer_tenure_years,
    s.carrier,
    s.shipping_method,
    s.distance_band,
    s.late_delivery,
    COUNT(oi.order_item_id)                                      AS num_items,
    SUM(oi.quantity)                                             AS total_qty
FROM orders o
JOIN customers   c  ON o.customer_id  = c.customer_id
JOIN shipments   s  ON o.order_id     = s.order_id
JOIN order_items oi ON o.order_id     = oi.order_id
WHERE o.fulfilled = 0
  AND NOT EXISTS (
      SELECT 1 FROM order_predictions p WHERE p.order_id = o.order_id
  )
GROUP BY
    o.order_id, o.order_subtotal, o.shipping_fee, o.tax_amount, o.order_total,
    o.risk_score, o.promo_used, o.payment_method, o.device_type, o.ip_country,
    o.shipping_state, o.billing_zip, o.shipping_zip, o.order_datetime,
    c.gender, c.customer_segment, c.loyalty_tier, c.is_active,
    c.birthdate, c.created_at,
    s.carrier, s.shipping_method, s.distance_band, s.late_delivery
"""

# ── Feature query for training (only human-labeled rows) ─────────────────────
QUERY_TRAINING_LABELED = """
SELECT
    o.order_id,
    o.order_subtotal,
    o.shipping_fee,
    o.tax_amount,
    o.order_total,
    o.risk_score,
    o.promo_used,
    o.payment_method,
    o.device_type,
    o.ip_country,
    o.shipping_state,
    CASE WHEN o.billing_zip != o.shipping_zip THEN 1 ELSE 0 END  AS zip_mismatch,
    EXTRACT(HOUR FROM o.order_datetime::timestamp)               AS order_hour,
    EXTRACT(DOW  FROM o.order_datetime::timestamp)               AS order_dow,
    c.gender,
    c.customer_segment,
    c.loyalty_tier,
    c.is_active                                                  AS customer_is_active,
    DATE_PART('year', AGE(NOW(), c.birthdate::timestamp))::INT   AS customer_age,
    DATE_PART('day', o.order_datetime::timestamp
                   - c.created_at::timestamp) / 365.25          AS customer_tenure_years,
    s.carrier,
    s.shipping_method,
    s.distance_band,
    s.late_delivery,
    COUNT(oi.order_item_id)                                      AS num_items,
    SUM(oi.quantity)                                             AS total_qty,
    o.admin_fraud_label
FROM orders o
JOIN customers   c  ON o.customer_id  = c.customer_id
JOIN shipments   s  ON o.order_id     = s.order_id
JOIN order_items oi ON o.order_id     = oi.order_id
WHERE o.admin_fraud_label IS NOT NULL
GROUP BY
    o.order_id, o.order_subtotal, o.shipping_fee, o.tax_amount, o.order_total,
    o.risk_score, o.promo_used, o.payment_method, o.device_type, o.ip_country,
    o.shipping_state, o.billing_zip, o.shipping_zip, o.order_datetime,
    o.admin_fraud_label,
    c.gender, c.customer_segment, c.loyalty_tier, c.is_active,
    c.birthdate, c.created_at,
    s.carrier, s.shipping_method, s.distance_band, s.late_delivery
"""


def check_secret(secret: str):
    if not SCORING_SECRET or secret != SCORING_SECRET:
        raise HTTPException(status_code=401, detail="Invalid scoring secret")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(x_scoring_secret: str = Header(...)):
    check_secret(x_scoring_secret)

    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Call POST /train first."
        )

    model = joblib.load(MODEL_PATH)

    df = load_from_supabase(QUERY_SCORING_BATCH)

    if df.empty:
        return {"scored": 0, "message": "No unfulfilled unscored orders found."}

    order_ids = df["order_id"].tolist()
    features = df.drop(columns=["order_id"])

    proba = model.predict_proba(features)[:, 1]
    predicted = (proba > 0.5).astype(int)

    # Write results back to Supabase
    from fraud_pipeline import load_credentials
    import psycopg2
    from datetime import datetime, timezone

    creds = load_credentials()
    conn = psycopg2.connect(**creds, sslmode="require")
    cursor = conn.cursor()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (int(order_id), float(prob), int(pred), timestamp)
        for order_id, prob, pred in zip(order_ids, proba, predicted)
    ]

    cursor.executemany("""
        INSERT INTO order_predictions
            (order_id, fraud_probability, predicted_late_delivery, prediction_timestamp)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (order_id) DO UPDATE
            SET fraud_probability        = EXCLUDED.fraud_probability,
                predicted_late_delivery  = EXCLUDED.predicted_late_delivery,
                prediction_timestamp     = EXCLUDED.prediction_timestamp
    """, rows)

    conn.commit()
    conn.close()

    print(f"Scored {len(rows)} orders")
    return {"scored": len(rows), "message": f"Scored {len(rows)} orders successfully."}


@app.post("/train")
def train(x_scoring_secret: str = Header(...)):
    check_secret(x_scoring_secret)

    df = load_from_supabase(QUERY_TRAINING_LABELED)

    if df.empty or df["admin_fraud_label"].nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Not enough labeled data to train. Need rows with both fraud and legit labels."
        )

    pipe = MLPipeline(
        df           = df,
        target       = "admin_fraud_label",
        models       = ["lr", "rf", "gb"],
        tune         = True,
        output_path  = MODEL_PATH,
        drop_cols    = ["order_id"],
        cat_strategy = "onehot",
        scale        = True,
        test_size    = 0.2,
        random_state = 42,
        cv_folds     = 5,
        verbose      = False,
    )

    pipe.run()

    return {"message": "Model trained and saved successfully.", "model_path": MODEL_PATH}