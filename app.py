import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Global variables for the trained model and encoders
model = None
le_platform = None
le_query_type = None
model_accuracy = 0
training_data = None

def generate_training_data():
    # Generate comprehensive training dataset
    np.random.seed(42)
    data = []
    platforms = ['AzureSynapse', 'Snowflake']
    query_types = ['SELECT', 'JOIN', 'AGGREGATION', 'WINDOW', 'SUBQUERY']
    print("Generating enterprise query dataset...")
    for i in range(1500):
        platform = np.random.choice(platforms)
        query_type = np.random.choice(query_types)
        num_tables = np.random.randint(1, 20)
        data_size_gb = np.random.exponential(scale=40)
        concurrent_queries = np.random.randint(1, 60)
        cpu_utilization = np.random.uniform(20, 95)
        memory_utilization = np.random.uniform(25, 90)
        io_utilization = np.random.uniform(10, 85)
        hour_of_day = np.random.randint(0, 24)
        is_business_hours = 1 if 9 <= hour_of_day <= 17 else 0
        complexity_score = num_tables*0.4 + (num_tables-1)*0.3 + np.log(data_size_gb+1)*0.3
        platform_factor = 0.9 if platform == 'Snowflake' else 1.1
        business_factor = 1.2 if is_business_hours else 1.0
        execution_time = complexity_score*3.5 + concurrent_queries*0.45 + cpu_utilization + memory_utilization + io_utilization*3 + np.random.exponential(scale=8) * platform_factor * business_factor
        execution_time = max(1.0, min(300, execution_time + np.random.normal(0, execution_time*0.15)))
        cost_usd = execution_time/3600*5.5*data_size_gb/100
        data.append({
            'platform': platform,
            'query_type': query_type,
            'num_tables': num_tables,
            'data_size_gb': round(data_size_gb, 2),
            'concurrent_queries': concurrent_queries,
            'cpu_utilization': round(cpu_utilization, 1),
            'memory_utilization': round(memory_utilization, 1),
            'io_utilization': round(io_utilization, 1),
            'hour_of_day': hour_of_day,
            'is_business_hours': is_business_hours,
            'complexity_score': round(complexity_score, 2),
            'execution_time_seconds': round(execution_time, 2),
            'cost_usd': round(cost_usd, 4)
        })
    return pd.DataFrame(data)

def train_model():
    # Train the ML model and return status
    global model, le_platform, le_query_type, model_accuracy, training_data
    try:
        df = generate_training_data()
        training_data = df
        le_platform = LabelEncoder()
        le_query_type = LabelEncoder()
        df['platform_encoded'] = le_platform.fit_transform(df['platform'])
        df['query_type_encoded'] = le_query_type.fit_transform(df['query_type'])
        df['resource_avg'] = (df['cpu_utilization'] + df['memory_utilization'] + df['io_utilization'])/3
        df['data_per_table'] = df['data_size_gb']/df['num_tables']
        df['load_factor'] = df['concurrent_queries']*df['resource_avg']/100
        feature_columns = [
            'platform_encoded', 'query_type_encoded', 'num_tables', 'data_size_gb',
            'concurrent_queries', 'cpu_utilization', 'memory_utilization', 'io_utilization',
            'is_business_hours', 'complexity_score', 'resource_avg', 'data_per_table', 'load_factor'
        ]
        X = df[feature_columns]
        y = df['execution_time_seconds']

        model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                     random_state=42, n_jobs=-1)
        model.fit(X, y)
        predictions = model.predict(X)
        model_accuracy = r2_score(y, predictions)*100

        platform_dist = df['platform'].value_counts().to_dict()
        avg_time = df['execution_time_seconds'].mean()
        total_cost = df['cost_usd'].sum()
        return (f"MODEL TRAINING SUCCESSFUL!\n\n"
                f"PERFORMANCE METRICS\n"
                f"Accuracy: {model_accuracy:.1f}%\n"
                f"Algorithm: Random Forest (200 trees)\n"
                f"Features: {len(feature_columns)} engineered features\n"
                f"DATASET STATISTICS\n"
                f"Total Queries: {len(df)}\n"
                f"Platform Distribution: {platform_dist}\n"
                f"Average Execution Time: {avg_time:.2f} seconds\n"
                f"Total Dataset Cost: {total_cost:.2f}\n\n"
                f"Model is ready for predictions!")
    except Exception as e:
        return f"Training Failed: {str(e)}"

def predict_query_time(platform, query_type, num_tables, data_size_gb, concurrent_queries, cpu_util, memory_util, io_util, business_hours):
    global model, le_platform, le_query_type
    if model is None:
        return "Please train the model first!"

    try:
        platform_encoded = le_platform.transform([platform])[0]
        query_type_encoded = le_query_type.transform([query_type])[0]
        complexity_score = num_tables*0.4 + (num_tables-1)*0.3 + np.log(data_size_gb+1)*0.3
        resource_avg = (cpu_util + memory_util + io_util)/3
        data_per_table = data_size_gb/num_tables
        load_factor = concurrent_queries*resource_avg/100
        is_business_hours = 1 if business_hours else 0
        features = [platform_encoded, query_type_encoded, num_tables, data_size_gb,
                    concurrent_queries, cpu_util, memory_util, io_util,
                    is_business_hours, complexity_score, resource_avg, data_per_table, load_factor]
        prediction = model.predict([features])[0]
        cost_estimate = prediction/3600*5.5*data_size_gb/100

        # Example optimizations
        optimizations = [('Smart Indexing', 0.82, 18), ('Resource Scaling', 0.70, 30),
                         ('Query Rewriting', 0.85, 15), ('Partition Pruning', 0.75, 25)]

        result = (f"PREDICTION RESULTS\n"
                  f"Predicted Execution Time: {prediction:.2f} seconds\n"
                  f"Estimated Cost: {cost_estimate:.4f}\n"
                  f"QUERY PARAMETERS\n"
                  f"Platform: {platform}\nQuery Type: {query_type}\n"
                  f"Tables: {num_tables}\nData Size: {data_size_gb:.1f} GB\n"
                  f"Concurrency: {concurrent_queries}\nBusiness Hours: {business_hours}\n"
                  f"Complexity Score: {complexity_score:.2f}\n\n"
                  f"OPTIMIZATION RECOMMENDATIONS\n")
        for name, factor, improvement in optimizations:
            opt_time = prediction * factor
            result += f"{name}: {opt_time:.1f}s ({improvement}% improvement)\n"
        return result
    except Exception as e:
        return f"Prediction Error: {str(e)}"

with gr.Blocks(title="ML-Driven SQL Query Optimizer", theme=gr.themes.Soft) as demo:
    gr.HTML("""<div style='text-align: center; margin-bottom: 30px'>
    <h1 style='color: #1f77b4; font-size: 2.5em;'>ML-Driven SQL Query Optimizer</h1>
    <div style="background: linear-gradient(135deg, #667eea 0, #764ba2 100); color: white; padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h3>Vishnu Vardhan K (2021WC86559)</h3>
    <p><strong>Wipro Technologies, Coimbatore – BITS Pilani M.Tech</strong></p>
    <p>Mid-Semester Project Review - September 2025</p>
    </div></div>
    """)

    with gr.Tabs():
        with gr.TabItem("Model Training"):
            gr.HTML("<h2>ML Model Training & Validation</h2>")
            with gr.Row():
                with gr.Column(scale=1):
                    train_btn = gr.Button("Train Enterprise ML Model", variant="primary", size="lg")
                with gr.Column(scale=2):
                    training_output = gr.Textbox(label="Training Results", placeholder="Click Train Enterprise ML Model to start...", lines=15)
            train_btn.click(train_model, outputs=training_output)

        with gr.TabItem("Query Prediction"):
            gr.HTML("<h2>Real-Time Query Performance Prediction</h2>")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3>Input Parameters</h3>")
                    platform = gr.Dropdown(['AzureSynapse', 'Snowflake'], label="Platform", value="AzureSynapse")
                    query_type = gr.Dropdown(['SELECT', 'JOIN', 'AGGREGATION', 'WINDOW', 'SUBQUERY'], label="Query Type", value="JOIN")
                    num_tables = gr.Slider(1, 30, value=8, step=1, label="Number of Tables")
                    data_size_gb = gr.Slider(1.0, 500.0, value=75.0, step=5.0, label="Data Size (GB)")
                    concurrent_queries = gr.Slider(1, 100, value=25, step=1, label="Concurrent Queries")
                    cpu_util = gr.Slider(10.0, 95.0, value=70.0, step=5.0, label="CPU Utilization")
                    memory_util = gr.Slider(20.0, 90.0, value=65.0, step=5.0, label="Memory Utilization")
                    io_util = gr.Slider(5.0, 85.0, value=50.0, step=5.0, label="IO Utilization")
                    business_hours = gr.Checkbox(label="Business Hours (9-17)", value=True)
                    predict_btn = gr.Button("Predict Execution Time", variant="primary", size="lg")
                with gr.Column(scale=2):
                    prediction_output = gr.Textbox(label="Prediction Results & Optimization Recommendations", placeholder="Enter parameters and click Predict Execution Time", lines=20)
            predict_btn.click(
                predict_query_time,
                inputs=[platform, query_type, num_tables, data_size_gb, concurrent_queries, cpu_util, memory_util, io_util, business_hours],
                outputs=prediction_output
            )
    demo.launch(share=True, debug=True)
