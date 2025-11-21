import streamlit as st
import pandas as pd
import requests
import io

FASTAPI_URL = "http://127.0.0.1:8000/run-automl"   # <-- your FastAPI endpoint



st.set_page_config(page_title="AutoML + NLP System", layout="wide",page_icon="Hey")

st.title("🤖 Multi-Agent AutoML + NLP System")
st.write("Upload a CSV file and get complete analysis, EDA, modeling, and report.")


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"],
    help="Upload your dataset in CSV format."
)


# -------------------------
# If file uploaded
# -------------------------
if uploaded_file:
    # Read file
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # Choose target column
    target = st.sidebar.selectbox(
        "Select Target Column",
        options=df.columns,
        help="Choose the target column for AutoML/NLP."
    )

    # Run button
    if st.sidebar.button(" Run AutoML Pipeline"):
        st.info("⏳ Processing your dataset... Please wait.")

        # prepare file for FastAPI
        file_data = uploaded_file.getvalue()

        # send request to FastAPI
        response = requests.post(
            FASTAPI_URL,
            files={"file": ("uploaded.csv", file_data, "text/csv")},
            data={"target": target}
        )

        if response.status_code == 200:
            result = response.json()

            st.success("✔ Completed!")

            # --------------------
            # Show results
            # --------------------
            st.subheader("📊 Problem Type")
            st.write(result.get("problem_type"))

            st.subheader("🏆 Best Model")
            st.write(result.get("best_model_name"))

            st.subheader("📈 Metrics")
            st.json(result.get("metrics"))

            st.subheader("📄 Summary Report")
            st.write(result.get("summary"))

            st.subheader("📚 All Model Comparison")
            st.json(result.get("results"))

        else:
            st.error("❌ Error from FastAPI: " + response.text)

else:
    st.info("⬅ Upload a CSV file from the sidebar to begin.")
