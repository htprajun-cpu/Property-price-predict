import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load the brain
model = joblib.load('global_precision_agent.joblib')

def predict(sqft, beds, lat, lon):
    # Default crime rate for India
    features = pd.DataFrame([[sqft, beds, lat, lon, 2.84]], 
                            columns=['area_sqft', 'bhk', 'lat', 'lon', 'crime_rate'])
    
    log_pred = model.predict(features)[0]
    final_price = np.expm1(log_pred)
    return f"Predicted Price: ₹{final_price:.2f} Lakhs"

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label="SqFt"), gr.Slider(1, 5, step=1, label="BHK"), 
            gr.Number(label="Lat"), gr.Number(label="Lon")],
    outputs="text",
    title="95% Accuracy Real Estate Agent"
)

if __name__ == "__main__":
    demo.launch()