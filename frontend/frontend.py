import streamlit as st
import pandas as pd
import requests
import io
from PIL import Image
import base64

FASTAPI_URL = "http://127.0.0.1:8000/run_automl"

st.set_page_config(page_title="AutoML + NLP System", layout="wide")

st.sidebar.image("C:\\Users\\Dell\\Downloads\\Multi-Agent Auto ML.png")

st.sidebar.subheader("API Health Status")
try:
    health = requests.get("http://127.0.0.1:8000/health")
    if health.status_code == 200:
        st.sidebar.success("API Running")
    else:
        st.sidebar.error("API Not Responding")
except:
    st.sidebar.error("API Offline")

page = st.sidebar.radio("Navigation", ["Home", "MultiAgent Auto-Ml + NLP System"])

if page == "Home":
    st.header("Multi Agent Auto ML")
    st.image("C:\\Users\\Dell\\Downloads\\Multi-Agent Auto ML.png")
    st.write("This project combines Automated Machine Learning (AutoML) with Natural Language Processing (NLP) to build an intelligent end-to-end text analysis pipeline. The system automatically performs data preprocessing, feature extraction, model selection and evaluation without requiring manual intervention.It supports tasks like text classification, sentiment analysis, and topic modeling, using techniques such as TF-IDF and modern optimization strategies. The AutoML engine selects the best NLP model based on performance metrics, while the pipeline is structured for scalability, reproducibility.")
    st.sidebar.markdown("---")
    st.sidebar.subheader("About App")
    st.sidebar.info("A project that uses AutoML with NLP to automatically clean text, extract features, try multiple models, and select the best-performing one for tasks like text classification or sentiment analysis")

# Footer
    st.sidebar.markdown("---")
    st.sidebar.write("💡 *Created by Vaishnavi Barolia*")
    

elif page == "MultiAgent Auto-Ml + NLP System":
    st.title("🤖 Multi-Agent AutoML + NLP System")
    st.write("Upload a CSV file and get complete analysis, EDA, modeling, and report.")

    # Sidebar upload
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        # Preview dataset
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        # Select target column
        target = st.sidebar.selectbox("Select Target Column", df.columns)

        # Run pipeline button
        if st.sidebar.button("Run AutoML Pipeline"):
            st.info("⏳ Processing your dataset... Please wait.")

            # Send request to API
            response = requests.post(
                FASTAPI_URL,
                files={"file": ("uploaded.csv", uploaded_file.getvalue(), "text/csv")},
                data={"target": target}
            )

            if response.status_code == 200:
                result = response.json()
                st.success("✔ Completed!")

             
                st.subheader("📊 Problem Type")
                st.write(result.get("problem_type"))

          
                st.subheader("🏆 Best Model")
                st.write(result.get("best_model_name"))

               
                st.subheader("📈 Metrics")
                st.json(result.get("metrics"))

               
                plots = result.get("plots", {})
                if plots:
                    st.subheader("📊 Visualizations")
                    for name, img_b64 in plots.items():
                        if img_b64:
                            img_data = base64.b64decode(img_b64)
                            img = Image.open(io.BytesIO(img_data))
                            st.image(img, caption=name, use_container_width=True)
                        else:
                            st.warning(f"⚠ Could not display {name}")
                else:
                    st.warning("⚠ No plots received from API.")

                # 🔹 Summary
                st.subheader("📄 Summary Report")
                st.write(result.get("summary"))

                # 🔹 All Models
                st.subheader("📚 All Model Comparison")
                st.json(result.get("results"))

            else:
                st.error("Error from FastAPI: " + response.text)

    else:
        st.info("⬅ Upload a CSV file from the sidebar to begin.")

