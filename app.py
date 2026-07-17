import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="BrainVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ---------------- THEME ----------------

dark = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html,body,[class*="css"]{
font-family:Poppins,sans-serif;
}

.stApp{
background:
linear-gradient(135deg,#020617,#0f172a,#1e3a8a);
background-attachment:fixed;
}

section[data-testid="stSidebar"]{
background:rgba(15,23,42,.65);
backdrop-filter:blur(15px);
}

.hero{
background:linear-gradient(135deg,#2563eb,#0f172a);
padding:35px;
border-radius:25px;
text-align:center;
box-shadow:0 10px 40px rgba(0,0,0,.45);
animation:fade 1s;
}

.hero h1{
font-size:55px;
color:white;
margin:0;
}

.hero p{
font-size:20px;
color:#dbeafe;
}

.glass{
background:rgba(255,255,255,.08);
backdrop-filter:blur(20px);
padding:25px;
border-radius:20px;
border:1px solid rgba(255,255,255,.12);
transition:.3s;
box-shadow:0 8px 25px rgba(0,0,0,.25);
}

.glass:hover{
transform:translateY(-8px);
box-shadow:0 15px 35px rgba(59,130,246,.45);
}

.metric{
font-size:28px;
font-weight:700;
color:#38bdf8;
}

.metric2{
color:white;
}

.resultGood{
background:#14532d;
padding:30px;
border-radius:20px;
text-align:center;
color:white;
animation:fade .6s;
}

.resultBad{
background:#7f1d1d;
padding:30px;
border-radius:20px;
text-align:center;
color:white;
animation:fade .6s;
}

.footer{
text-align:center;
padding:20px;
color:#cbd5e1;
}

@keyframes fade{
from{
opacity:0;
transform:translateY(30px);
}
to{
opacity:1;
transform:translateY(0px);
}
}

</style>
"""

light = """
<style>

.stApp{
background:#eef4ff;
}

.hero{
background:linear-gradient(135deg,#2563eb,#60a5fa);
padding:35px;
border-radius:25px;
text-align:center;
}

.hero h1{
color:white;
}

.hero p{
color:white;
}

.glass{
background:white;
padding:25px;
border-radius:20px;
box-shadow:0 10px 30px rgba(0,0,0,.15);
}

.metric{
font-size:28px;
font-weight:bold;
color:#2563eb;
}

.metric2{
color:#111827;
}

.resultGood{
background:#dcfce7;
padding:30px;
border-radius:20px;
text-align:center;
}

.resultBad{
background:#fee2e2;
padding:30px;
border-radius:20px;
text-align:center;
}

.footer{
text-align:center;
padding:20px;
}

</style>
"""

if st.session_state.theme=="Dark":
    st.markdown(dark,unsafe_allow_html=True)
else:
    st.markdown(light,unsafe_allow_html=True)
    # ---------------- SIDEBAR ----------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/240/brain.png",
        width=120
    )

    st.title("🧠 BrainVision AI")

    st.markdown("---")

    theme = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.theme=="Dark"
    )

    if theme:
        st.session_state.theme="Dark"
    else:
        st.session_state.theme="Light"

    st.markdown("---")

    st.subheader("📋 System Information")

    st.info("""
CNN Architecture

TensorFlow

MRI Classification

224 × 224 Input

Binary Classification
""")

    st.markdown("---")

    st.success("""
Version : 2.0

Developer

Uradi Narsimulu
""")

    st.warning(
        "Educational Use Only.\n\nNot a Medical Diagnosis."
    )

# ---------------- HERO ----------------

st.markdown("""
<div class="hero">

<h1>🧠 BrainVision AI</h1>

<p>
Artificial Intelligence Powered Brain Tumor Detection
</p>

</div>
""",unsafe_allow_html=True)

st.write("")

# ---------------- DASHBOARD ----------------

c1,c2,c3,c4=st.columns(4)

with c1:

    st.markdown("""
<div class="glass">

<center>

<h1>🧠</h1>

<div class="metric">
CNN
</div>

<div class="metric2">
Deep Learning
</div>

</center>

</div>
""",unsafe_allow_html=True)

with c2:

    st.markdown("""
<div class="glass">

<center>

<h1>📷</h1>

<div class="metric">
224×224
</div>

<div class="metric2">
MRI Input
</div>

</center>

</div>
""",unsafe_allow_html=True)

with c3:

    st.markdown("""
<div class="glass">

<center>

<h1>⚡</h1>

<div class="metric">
AI
</div>

<div class="metric2">
Fast Prediction
</div>

</center>

</div>
""",unsafe_allow_html=True)

with c4:

    st.markdown("""
<div class="glass">

<center>

<h1>🩺</h1>

<div class="metric">
Medical
</div>

<div class="metric2">
Diagnosis Support
</div>

</center>

</div>
""",unsafe_allow_html=True)

st.write("")

# ---------------- MAIN LAYOUT ----------------

left,right=st.columns([1,1])

with left:

    st.markdown("## 📤 Upload MRI Scan")

    uploaded=st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

with right:

    st.markdown("""
<div class="glass">

<h3>📖 Instructions</h3>

✔ Upload MRI Brain Scan

✔ AI analyzes MRI

✔ Confidence Score

✔ Download Report

✔ Prediction History

</div>
""",unsafe_allow_html=True)

st.write("")

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "brain_tumor_model.h5"
    )

model=load_model()

# ---------------- PLOTLY GAUGE ----------------

def draw_gauge(value):

    fig=go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,

            number={
                "suffix":"%"
            },

            title={
                "text":"Confidence"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":"royalblue"
                },

                "steps":[

                    {
                        "range":[0,50],
                        "color":"lightgray"
                    },

                    {
                        "range":[50,80],
                        "color":"orange"
                    },

                    {
                        "range":[80,100],
                        "color":"green"
                    }

                ]
            }
        )
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=40,
            b=20
        )
    )

    return fig
# ---------------- PREDICTION ----------------

if uploaded is not None:

    img = Image.open(uploaded).convert("RGB")

    st.markdown("## 🖼 Uploaded MRI Scan")

    c1, c2, c3 = st.columns([1,2,1])

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.image(
            img,
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PREPROCESS ----------------

    img_resize = img.resize((224,224))

    img_array = image.img_to_array(img_resize)

    img_array = np.expand_dims(img_array,axis=0)

    img_array = img_array/255.0

    st.write("")

    st.markdown("## 🧠 AI Analysis")

    progress = st.progress(0)

    status = st.empty()

    scan_steps = [

        "📥 Loading MRI image...",

        "🧠 Extracting features...",

        "🔬 Running CNN Model...",

        "📊 Calculating confidence...",

        "✅ Preparing report..."

    ]

    for i, step in enumerate(scan_steps):

        status.info(step)

        for j in range(20):

            progress.progress(i*20+j+1)

            time.sleep(0.02)

    status.success("Prediction Complete")

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    confidence = prediction if prediction>0.5 else 1-prediction

    confidence_percent = confidence*100

    st.write("")

    col1,col2=st.columns([1,1])

    # ---------------- GAUGE ----------------

    with col1:

        st.plotly_chart(
            draw_gauge(confidence_percent),
            use_container_width=True
        )

    # ---------------- RESULT ----------------

    with col2:

        if prediction>0.5:

            st.markdown(f"""

<div class="resultBad">

<h1>🚨 Tumor Detected</h1>

<h2>{confidence_percent:.2f}% Confidence</h2>

<hr>

<h4>Recommendation</h4>

Consult a Neurologist or Radiologist for
further clinical evaluation.

</div>

""",unsafe_allow_html=True)

            result = "Tumor Detected"

        else:

            st.markdown(f"""

<div class="resultGood">

<h1>✅ No Tumor Detected</h1>

<h2>{confidence_percent:.2f}% Confidence</h2>

<hr>

<h4>Recommendation</h4>

No tumor identified by the AI model.
Clinical consultation is still recommended.

</div>

""",unsafe_allow_html=True)

            result = "No Tumor"

    # ---------------- METRICS ----------------

    st.write("")

    m1,m2,m3=st.columns(3)

    with m1:

        st.metric(
            "Prediction",
            result
        )

    with m2:

        st.metric(
            "Confidence",
            f"{confidence_percent:.2f}%"
        )

    with m3:

        st.metric(
            "Model",
            "CNN"
        )

    # ---------------- SAVE HISTORY ----------------

    st.session_state.history.append({

        "Time":datetime.now().strftime("%d-%m-%Y %H:%M:%S"),

        "Prediction":result,

        "Confidence":f"{confidence_percent:.2f}%"

    })

    st.success("✔ Prediction saved to history.")
    # =====================================================
# PDF REPORT GENERATOR
# =====================================================

def create_pdf(prediction_result, confidence):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    story = []

    title = Paragraph(
        "<b><font size=20 color='blue'>BrainVision AI Report</font></b>",
        styles["Title"]
    )

    story.append(title)
    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            f"<b>Date :</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,12))

    story.append(
        Paragraph(
            "<b>Model :</b> CNN (TensorFlow)",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,12))

    story.append(
        Paragraph(
            f"<b>Prediction :</b> {prediction_result}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,12))

    story.append(
        Paragraph(
            f"<b>Confidence :</b> {confidence:.2f}%",
            styles["Normal"]
        )
    )

    story.append(Spacer(1,25))

    if prediction_result=="Tumor Detected":

        recommendation="""
<b>Recommendation</b><br/><br/>

AI indicates the possible presence of a brain tumor.

Please consult a Neurologist or Radiologist
for further diagnosis.

"""

    else:

        recommendation="""
<b>Recommendation</b><br/><br/>

No brain tumor detected by the AI model.

Continue regular medical follow-up if required.

"""

    story.append(
        Paragraph(
            recommendation,
            styles["BodyText"]
        )
    )

    story.append(Spacer(1,25))

    disclaimer="""
<b>Disclaimer</b><br/><br/>

This application is intended only for educational
and research purposes.

It should never replace a certified medical diagnosis.

"""

    story.append(
        Paragraph(
            disclaimer,
            styles["BodyText"]
        )
    )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf


# =====================================================
# PDF DOWNLOAD
# =====================================================

if uploaded is not None:

    pdf = create_pdf(
        result,
        confidence_percent
    )

    st.download_button(

        label="📄 Download PDF Report",

        data=pdf,

        file_name="BrainVision_Report.pdf",

        mime="application/pdf"

    )


# =====================================================
# PREDICTION HISTORY
# =====================================================

st.write("")
st.markdown("## 📊 Prediction History")

if len(st.session_state.history)>0:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(

        history_df,

        use_container_width=True,

        hide_index=True

    )

    csv = history_df.to_csv(
        index=False
    ).encode()

    st.download_button(

        "⬇ Download History CSV",

        csv,

        "prediction_history.csv",

        "text/csv"

    )

    if st.button("🗑 Clear History"):

        st.session_state.history=[]

        st.rerun()

else:

    st.info("No predictions yet.")


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""

<div class="footer">

<h3>🧠 BrainVision AI</h3>

Deep Learning Brain Tumor Detection System

<br><br>

Built with ❤️ using

TensorFlow • Keras • Streamlit • Plotly

<br><br>

© 2026 All Rights Reserved

</div>

""",unsafe_allow_html=True)
# =====================================================
# PREMIUM UI ENHANCEMENTS (PART 5)
# =====================================================

st.markdown("""
<style>

/* Smooth animation */
*{
transition:0.35s ease;
}

/* Glass Card */
.glass-card{
    background:rgba(255,255,255,.08);
    backdrop-filter:blur(18px);
    border-radius:20px;
    padding:20px;
    border:1px solid rgba(255,255,255,.12);
    box-shadow:0 10px 30px rgba(0,0,0,.25);
}

.glass-card:hover{
    transform:translateY(-6px);
    box-shadow:0 18px 40px rgba(37,99,235,.35);
}

/* Upload Box */

.upload-box{
    border:2px dashed #3b82f6;
    border-radius:20px;
    padding:30px;
    text-align:center;
    color:white;
    background:rgba(255,255,255,.05);
}

/* Result Animation */

.resultGood{

animation:pop .6s;

}

.resultBad{

animation:pop .6s;

}

@keyframes pop{

0%{

transform:scale(.8);
opacity:0;

}

100%{

transform:scale(1);
opacity:1;

}

}

/* Floating animation */

.hero{

animation:floating 5s ease-in-out infinite;

}

@keyframes floating{

0%{

transform:translateY(0px);

}

50%{

transform:translateY(-8px);

}

100%{

transform:translateY(0px);

}

}

/* Buttons */

.stButton>button{

width:100%;

border-radius:12px;

font-size:17px;

font-weight:600;

background:linear-gradient(90deg,#2563eb,#3b82f6);

color:white;

border:none;

padding:12px;

}

.stButton>button:hover{

background:linear-gradient(90deg,#1d4ed8,#2563eb);

transform:scale(1.03);

}

/* Download Button */

.stDownloadButton>button{

width:100%;

border-radius:12px;

background:#16a34a;

color:white;

font-weight:bold;

padding:12px;

border:none;

}

.stDownloadButton>button:hover{

background:#15803d;

}

/* Metric Cards */

[data-testid="metric-container"]{

background:rgba(255,255,255,.08);

padding:20px;

border-radius:15px;

box-shadow:0 8px 20px rgba(0,0,0,.25);

}

/* Progress Bar */

.stProgress>div>div{

background:linear-gradient(90deg,#2563eb,#06b6d4);

}

/* Scrollbar */

::-webkit-scrollbar{

width:8px;

}

::-webkit-scrollbar-thumb{

background:#2563eb;

border-radius:20px;

}

/* Responsive */

@media(max-width:900px){

.hero h1{

font-size:34px;

}

.metric{

font-size:20px;

}

}

</style>
""",unsafe_allow_html=True)

# =====================================================
# AI SCANNING EFFECT
# =====================================================

def ai_scan():

    text = st.empty()

    bar = st.progress(0)

    steps = [

        "🧠 Initializing AI...",

        "📷 Loading MRI...",

        "🔬 Extracting Features...",

        "⚡ Running CNN Layers...",

        "📊 Calculating Confidence...",

        "✅ Finalizing Prediction..."

    ]

    value = 0

    for step in steps:

        text.info(step)

        for i in range(15):

            value += 1

            if value > 100:
                value = 100

            bar.progress(value)

            time.sleep(0.02)

    text.success("Analysis Completed")
    # =====================================================
# PART 6 - HISTORY DASHBOARD & STATISTICS
# =====================================================

from openpyxl import Workbook
from io import BytesIO

st.markdown("---")
st.header("📊 Prediction Dashboard")

# ---------------- HISTORY ----------------

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    left, right = st.columns([3,1])

    with left:

        search = st.text_input(
            "🔍 Search Prediction",
            placeholder="Tumor or No Tumor"
        )

    with right:

        sort_order = st.selectbox(
            "Sort",
            ["Newest First","Oldest First"]
        )

    df = history_df.copy()

    if search:

        df = df[
            df["Prediction"].str.contains(
                search,
                case=False
            )
        ]

    if sort_order == "Newest First":

        df = df.iloc[::-1]

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    # ---------------- METRICS ----------------

    total = len(history_df)

    tumor = len(
        history_df[
            history_df["Prediction"]=="Tumor Detected"
        ]
    )

    normal = total - tumor

    c1,c2,c3 = st.columns(3)

    with c1:

        st.metric(
            "Total Predictions",
            total
        )

    with c2:

        st.metric(
            "Tumor Cases",
            tumor
        )

    with c3:

        st.metric(
            "Healthy Cases",
            normal
        )

    # ---------------- PIE CHART ----------------

    pie = go.Figure(
        data=[
            go.Pie(
                labels=[
                    "Tumor",
                    "Healthy"
                ],
                values=[
                    tumor,
                    normal
                ],
                hole=.45
            )
        ]
    )

    pie.update_layout(
        height=420,
        title="Prediction Distribution"
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    # ---------------- BAR CHART ----------------

    bar = go.Figure()

    bar.add_bar(
        x=["Tumor","Healthy"],
        y=[tumor,normal]
    )

    bar.update_layout(
        title="Prediction Count",
        height=420
    )

    st.plotly_chart(
        bar,
        use_container_width=True
    )

    # ---------------- CSV ----------------

    csv = history_df.to_csv(
        index=False
    ).encode()

    st.download_button(
        "⬇ Download CSV",
        csv,
        "Prediction_History.csv",
        "text/csv"
    )

    # ---------------- EXCEL ----------------

    wb = Workbook()

    ws = wb.active

    ws.title = "Prediction History"

    ws.append(history_df.columns.tolist())

    for row in history_df.values.tolist():

        ws.append(row)

    excel = BytesIO()

    wb.save(excel)

    excel.seek(0)

    st.download_button(

        "📄 Download Excel",

        excel,

        "Prediction_History.xlsx",

        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    )

    # ---------------- CLEAR ----------------

    if st.button("🗑 Clear Prediction History"):

        st.session_state.history=[]

        st.success("History Cleared")

        st.rerun()

else:

    st.info("No prediction history available.")

# =====================================================
# QUICK INSIGHTS
# =====================================================

if len(st.session_state.history)>0:

    st.markdown("---")

    st.subheader("📈 AI Insights")

    if tumor > normal:

        st.warning(
            "More Tumor cases than Healthy predictions."
        )

    elif normal > tumor:

        st.success(
            "Healthy predictions are higher."
        )

    else:

        st.info(
            "Tumor and Healthy predictions are equal."
        )
        # =====================================================
# PART 7 - PREMIUM UI ENHANCEMENTS
# =====================================================

# ---------------- TOAST NOTIFICATIONS ----------------

def notify_success(msg):
    try:
        st.toast(msg, icon="✅")
    except:
        st.success(msg)

def notify_error(msg):
    try:
        st.toast(msg, icon="⚠️")
    except:
        st.warning(msg)

# ---------------- HERO PARTICLES ----------------

st.markdown("""
<style>

.hero{
position:relative;
overflow:hidden;
}

.hero::before{

content:"";

position:absolute;

width:220px;
height:220px;

background:rgba(255,255,255,.08);

border-radius:50%;

top:-70px;
left:-70px;

animation:float1 10s infinite linear;

}

.hero::after{

content:"";

position:absolute;

width:180px;
height:180px;

background:rgba(255,255,255,.06);

border-radius:50%;

bottom:-60px;
right:-60px;

animation:float2 12s infinite linear;

}

@keyframes float1{

0%{
transform:translateY(0px);
}

50%{
transform:translateY(25px);
}

100%{
transform:translateY(0px);
}

}

@keyframes float2{

0%{
transform:translateY(0px);
}

50%{
transform:translateY(-25px);
}

100%{
transform:translateY(0px);
}

}

/* MRI Image */

img{

border-radius:18px;

transition:.35s;

}

img:hover{

transform:scale(1.03);

box-shadow:0 12px 35px rgba(37,99,235,.45);

}

/* Floating Metrics */

[data-testid="metric-container"]{

transition:.35s;

}

[data-testid="metric-container"]:hover{

transform:translateY(-6px);

}

.badge{

padding:12px;

border-radius:12px;

font-weight:bold;

background:#2563eb;

color:white;

text-align:center;

font-size:18px;

margin-bottom:15px;

}

</style>

""",unsafe_allow_html=True)

# ---------------- CONFIDENCE BADGE ----------------

if uploaded is not None:

    st.markdown(

        f"""

<div class="badge">

🎯 AI Confidence :
<b>{confidence_percent:.2f}%</b>

</div>

""",

unsafe_allow_html=True

    )

# ---------------- SUCCESS ANIMATION ----------------

if uploaded is not None:

    if confidence_percent >= 95:

        st.balloons()

        notify_success(
            "High Confidence Prediction Completed!"
        )

    elif confidence_percent >= 80:

        notify_success(
            "Prediction Completed Successfully."
        )

    else:

        notify_error(
            "Prediction confidence is moderate."
        )

# ---------------- MODEL INFO ----------------

with st.expander("🧠 AI Model Details"):

    st.markdown("""

### CNN Model

- Framework : TensorFlow/Keras

- Image Size : **224 × 224**

- Classification : Binary

- Output :

    - Tumor

    - No Tumor

This model predicts whether an MRI image
contains signs of a brain tumor.

""")

# ---------------- SYSTEM STATUS ----------------

with st.expander("⚙️ System Status"):

    st.success("✔ TensorFlow Loaded")

    st.success("✔ CNN Model Loaded")

    st.success("✔ Streamlit Running")

    st.success("✔ Prediction Engine Ready")

# ---------------- COPYRIGHT ----------------

st.markdown("""

<br>

<center>

<h5>

🧠 BrainVision AI Version 2.0

</h5>

<p>

Developed using

TensorFlow • Streamlit • Plotly

</p>

</center>

""",unsafe_allow_html=True)
# =====================================================
# PART 8 - FINAL POLISH & DEPLOYMENT READY
# =====================================================

import os

# ---------------- MODEL CHECK ----------------

MODEL_PATH = "brain_tumor_model.h5"

if not os.path.exists(MODEL_PATH):
    st.error("❌ brain_tumor_model.h5 not found.")
    st.info("Place brain_tumor_model.h5 in the same folder as app.py")
    st.stop()

# ---------------- WELCOME SCREEN ----------------

if uploaded is None:

    st.markdown("""
    <div class="glass-card">
    <h2 style="text-align:center;">👋 Welcome to BrainVision AI</h2>

    <p style="text-align:center;font-size:18px;">
    Upload an MRI image to begin AI analysis.
    </p>

    <hr>

    <h4>Features</h4>

    <ul>
    <li>🧠 CNN Deep Learning Model</li>
    <li>📈 Confidence Gauge</li>
    <li>📄 PDF Report</li>
    <li>📊 Prediction History</li>
    <li>🌙 Dark / Light Theme</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------

st.markdown("---")

st.warning("""
### ⚠ Medical Disclaimer

This AI application is intended only for educational
and research purposes.

The prediction generated by this software should
NOT be considered as a medical diagnosis.

Always consult a qualified neurologist or
radiologist before making medical decisions.
""")



with st.expander("💻 Application Information"):

    st.write("**Application :** BrainVision AI")
    st.write("**Version :** 2.0")
    st.write("**Framework :** Streamlit")
    st.write("**Deep Learning :** TensorFlow / Keras")
    st.write("**Input Size :** 224 × 224")
    st.write("**Classification :** Binary")


with st.sidebar:

    st.markdown("---")

    st.caption("BrainVision AI")

    st.caption("Version 2.0")

    st.caption("Developed by")

    st.success("Uradi Narsimulu")

# ---------------- FINAL FOOTER ----------------

st.markdown("""
<hr>

<div style="text-align:center;padding:20px;">

<h3>🧠 BrainVision AI</h3>

<p>
Artificial Intelligence Based Brain Tumor Detection
</p>

<p>

Built using

TensorFlow • Keras • Streamlit • Plotly

</p>

<p>

© 2026 All Rights Reserved

</p>

</div>

""", unsafe_allow_html=True)



