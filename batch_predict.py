import requests
import csv
import random

# Generate 1000 random test queries
samples = []
for _ in range(1000):
    sample = {
        "rows_returned": random.randint(1, 5000),
        "has_group_by": random.choice([0, 1]),
        "num_joins": random.randint(0, 5),
        "has_subquery": random.choice([0, 1]),
        "query_type_AGGREGATE": random.choice([0, 1]),
        "query_type_GROUP BY": random.choice([0, 1]),
        "query_type_JOIN": random.choice([0, 1]),
        "query_type_SELECT": random.choice([0, 1])
    }
    samples.append(sample)

# Send requests and collect predictions
with open("batch_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = list(samples[0].keys()) + ["predicted_runtime"]
    writer.writerow(header)
    for payload in samples:
        resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
        prediction = resp.json()["predicted_runtime"]
        writer.writerow(list(payload.values()) + [prediction])

print("Batch predictions saved to batch_predictions.csv")
