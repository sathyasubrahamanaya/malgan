import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.title("Mal-GAN Production Prototype")

# Sidebar: Choose between generating synthetic malware or detecting malware.
option = st.sidebar.radio("Select Option", ["Generate Synthetic Malware", "Detect Malware"])

@st.cache_resource
def load_models():
    try:
        generator = load_model("generator.h5")
    except Exception as e:
        st.error("Error loading generator model: " + str(e))
        raise e
    try:
        with open("blackBox_file.pkl", "rb") as f:
            blackBox = pickle.load(f)
    except Exception as e:
        st.error("Error loading blackBox model: " + str(e))
        raise e
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error("Error loading scaler: " + str(e))
        raise e
    try:
        with open("column_order.pkl", "rb") as f:
            column_order = pickle.load(f)
    except Exception as e:
        st.error("Error loading column order: " + str(e))
        raise e
    return generator, blackBox, scaler, column_order

# Load models and associated files.
generator, blackBox, scaler, column_order = load_models()

# Expected number of features based on the column order.
expected_features = len(column_order)

if option == "Generate Synthetic Malware":
    st.header("Generate Synthetic Malware Sample")
    st.write("Click the button below to generate a synthetic malware sample using the trained generator model.")
    
    if st.button("Generate"):
        # Create random inputs for malware sample and noise; dimensions match the expected number of features.
        malware_sample = np.random.rand(1, expected_features)
        noise_sample = np.random.normal(0, 1, (1, expected_features))
        synthetic_sample = generator.predict([malware_sample, noise_sample])
        st.write("Synthetic Malware Sample Generated:")
        st.write(synthetic_sample)
        
elif option == "Detect Malware":
    st.header("Detect Malware Sample")

    st.write(
        "Provide a sample for malware detection. You can either upload a CSV file containing raw feature data "
        "or manually enter a comma-separated feature vector. The sample will be scaled using the production scaler."
       
    )
   
    
    # Option 1: CSV file upload.
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("CSV file loaded. Preview:")
            st.dataframe(df.head())
            
            # If the CSV file has column names, reorder if needed.
            if set(column_order).issubset(df.columns):
                df = df[column_order]
            elif df.shape[1] == expected_features:
                # If no column names are provided, assume the order is correct.
                pass
            else:
                st.error(f"Uploaded CSV must have {expected_features} features in the correct order.")
                df = None
            
            if df is not None:
                # Convert to numpy array and scale using the loaded scaler.
                raw_features = df.values.astype(np.float32)
                scaled_features = scaler.transform(raw_features)
                predictions = blackBox.predict(scaled_features)
                # Append predictions to the DataFrame.
                df_result = df.copy()
                df_result["Prediction"] = predictions
                st.write("Predictions (1 = Malware, 0 = Benign):")
                st.dataframe(df_result)
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
    else:
        # Option 2: Manual input via text area.
        user_input = st.text_area("Or Enter Feature Vector (comma-separated)", 
            "0.5,0.3,0.2,0.7,0.6,0.4,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5")
        if st.button("Detect Single Sample"):
            try:
                # Convert the input string into a list of floats.
                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) != expected_features:
                    st.error(f"Expected {expected_features} features, but got {len(values)}. Please enter the correct number.")
                else:
                    sample = np.array(values, dtype=np.float32).reshape(1, -1)
                    # Scale the sample using the production scaler.
                    scaled_sample = scaler.transform(sample)
                    prediction = blackBox.predict(scaled_sample)
                    result = "Malware" if prediction[0] == 1 else "Benign"
                    st.write("Prediction:", result, f"({prediction[0]})")
            except Exception as e:
                st.error(f"Error processing input: {e}")
