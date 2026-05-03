import streamlit as st
import requests
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Medical AI Cancer Detection", layout="centered")

# ── Medical info cards ────────────────────────────────────────────────────────
CANCER_INFO = {
    "glioma": {
        "description": "Malignant tumour arising from glial cells in the brain or spinal cord.",
        "symptoms"   : "Persistent headache, seizures, nausea, cognitive changes.",
        "next_steps" : "Urgent neurosurgery / neurology referral. MRI with contrast required.",
    },
    "meningioma": {
        "description": "Usually benign tumour forming in the meninges (brain/spinal membranes).",
        "symptoms"   : "Headaches, vision disturbances, hearing loss, weakness.",
        "next_steps" : "Neurosurgery consultation; watchful waiting for slow-growing cases.",
    },
    "pituitary tumor": {
        "description": "Abnormal growth in the pituitary gland affecting hormone regulation.",
        "symptoms"   : "Hormonal imbalance, visual field defects, fatigue.",
        "next_steps" : "Endocrinologist + neurosurgeon consultation.",
    },
    "no tumor": {
        "description": "No tumour detected in this scan.",
        "symptoms"   : "None identified.",
        "next_steps" : "Continue routine monitoring as directed by your physician.",
    },
    "benign lung lesion": {
        "description": "A non-cancerous lung lesion — may include granulomas or benign nodules.",
        "symptoms"   : "Often asymptomatic; may cause mild cough or chest discomfort.",
        "next_steps" : "Pulmonology follow-up. Serial CT scans to monitor for changes.",
    },
    "malignant lung cancer": {
        "description": "Malignant lung tumour confirmed on CT imaging.",
        "symptoms"   : "Persistent cough, haemoptysis, shortness of breath, weight loss.",
        "next_steps" : "Urgent oncology and thoracic surgery referral. Staging required.",
    },
    "normal lung": {
        "description": "No malignant or suspicious changes detected on CT scan.",
        "symptoms"   : "None identified.",
        "next_steps" : "Continue regular health check-ups as directed by your physician.",
    },
    # Colon — Kather 2016 eight tissue classes
    "colorectal adenocarcinoma": {
        "description": "Malignant tumour epithelium; primary tissue of colorectal cancer.",
        "symptoms"   : "Rectal bleeding, change in bowel habits, abdominal pain, weight loss.",
        "next_steps" : "Urgent colorectal surgery / oncology referral. Full staging required.",
    },
    "cancer-associated stroma": {
        "description": "Reactive stromal tissue surrounding colorectal tumour cells.",
        "symptoms"   : "Usually found alongside tumour tissue — indicates active cancer.",
        "next_steps" : "Oncology referral; correlate with tumour epithelium findings.",
    },
    "complex glandular epithelium": {
        "description": "Architecturally complex glands — may be high-grade dysplasia or adenoma.",
        "symptoms"   : "Often pre-cancerous; may progress to invasive carcinoma.",
        "next_steps" : "Gastroenterology review; biopsy and close endoscopic surveillance.",
    },
    "lymphocytic infiltrate": {
        "description": "Dense aggregates of lymphocytes — inflammatory or immune response.",
        "symptoms"   : "Associated with inflammatory bowel disease or tumour immunoreactivity.",
        "next_steps" : "Gastroenterology review to rule out Crohn's / IBD.",
    },
    "cellular debris": {
        "description": "Necrotic material and cellular breakdown products.",
        "symptoms"   : "Indicates tissue necrosis — often associated with high-grade tumours.",
        "next_steps" : "Correlate with other tissue findings; oncology review.",
    },
    "normal mucosa": {
        "description": "Normal colonic mucosal glands — no pathological changes detected.",
        "symptoms"   : "None identified.",
        "next_steps" : "Continue routine colorectal screening as directed by your physician.",
    },
    "adipose tissue": {
        "description": "Normal adipose (fat) tissue — no pathological significance.",
        "symptoms"   : "None.",
        "next_steps" : "No action required; routine follow-up.",
    },
    "background / empty": {
        "description": "Background glass or empty tissue region — no diagnostic tissue present.",
        "symptoms"   : "N/A — image region contains no tissue.",
        "next_steps" : "Rescan with a tissue-containing region of the slide.",
    },
}


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Multi-Cancer AI Detection System")
st.markdown("Analyse brain MRI, lung CT, and colon histopathology images.")

cancer_type = st.selectbox("Select scan type", ["brain", "lung", "colon"])
uploaded    = st.file_uploader("Upload medical image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    st.image(Image.open(uploaded), caption="Uploaded image", width=380)

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running AI analysis …"):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    data={"cancer_type": cancer_type},
                    timeout=60,
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend. Is the FastAPI server running?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. The server may be loading models.")
                st.stop()

        if response.status_code != 200:
            st.error(f"Server error {response.status_code}: {response.text}")
            st.stop()

        result = response.json()
        status = result.get("status")

        # ── Rejected / Error ────────────────────────────────────────────────
        if status == "rejected":
            reason = result.get("reason", "")
            msg    = result.get("message", "Request rejected.")
            st.error(f"**Rejected** ({reason}): {msg}")
            st.stop()

        if status == "error":
            st.error(f"**Error** ({result.get('reason')}): {result.get('message')}")
            st.stop()

        # ── Successful prediction ────────────────────────────────────────────
        predicted_class = result["predicted_class"]

        # ── Ambiguous / Invalid Scan ─────────────────────────────────────────
        if predicted_class == "Invalid Scan":
            st.warning(
                "**Invalid Scan** — the model could not produce a reliable prediction. "
                "Please upload a clearer or higher-quality image."
            )
            st.stop()

        st.success("Analysis complete")
        st.metric("Prediction", predicted_class)

        # Medical info card
        key = predicted_class.lower()
        if key in CANCER_INFO:
            info = CANCER_INFO[key]
            st.markdown("---")
            st.subheader("Medical Information")
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Common symptoms:** {info['symptoms']}")
            st.write(f"**Recommended next steps:** {info['next_steps']}")

        st.info(
            "⚠️ This system is a research tool only. "
            "It does not replace professional medical diagnosis. "
            "Always consult a qualified physician."
        )
