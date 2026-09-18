# app.py
import os
import tempfile
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import onnxruntime as ort
import streamlit as st
import json
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
import time

# ==== Constants ====
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
SPEC_LEN = 128

INSTR_FAMILY_MAP = {
    0: 'Accordion', 1: 'Acoustic_Guitar', 2: 'Banjo', 3: 'Bass_Guitar', 4: 'Clarinet',
    5: 'Cymbals', 6: 'Dobro', 7: 'Drum_set', 8: 'Electro_Guitar', 9: 'Floor_Tom',
    10: 'Harmonica', 11: 'Harmonium', 12: 'Hi_Hats', 13: 'Horn', 14: 'Keyboard',
    15: 'Mandolin', 16: 'Organ', 17: 'Piano', 18: 'Saxophone', 19: 'Shakers',
    20: 'Tambourine', 21: 'Trombone', 22: 'Trumpet', 23: 'Ukulele', 24: 'Violin',
    25: 'cowbell', 26: 'flute', 27: 'vibraphone'
}

# Custom CSS for enhanced design and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3rem !important;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInUp 1s ease-out;
    }
    
    .hero-text {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 3rem;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    .upload-area {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(20px);
        border: 2px dashed rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: floatIn 1.2s ease-out;
    }
    
    .upload-area:hover {
        background: rgba(255,255,255,0.25);
        border-color: rgba(255,255,255,0.6);
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.2);
    }
    
    .predict-btn {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        border: none;
        border-radius: 15px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        color: white;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        animation: slideInUp 1s ease-out 0.4s both;
    }
    
    .predict-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 40px rgba(255,107,107,0.4);
    }
    
    .predict-btn:active {
        transform: translateY(-1px);
    }
    
    .results-container {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 0 30px 60px rgba(0,0,0,0.25);
        animation: slideInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        margin-top: 2rem;
    }
    
    .top-prediction {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
    }
    
    .prediction-card {
        background: rgba(255,255,255,0.8);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .prediction-card:hover {
        transform: translateX(10px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
    }
    
    .progress-bar {
        background: rgba(255,255,255,0.2);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        border-radius: 4px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .audio-player {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes floatIn {
        0% { opacity: 0; transform: scale(0.9) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .stMetric {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-family: 'Inter', sans-serif;
    }
    
    .download-section {
        background: linear-gradient(135deg, #ff6b6b, #feca57);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: bounceIn 1s ease-out;
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ==== ONNX wrapper ====
class NsynthOnnxWrapper:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            center=True,
            power=2.0,
        )
        self.to_db = AmplitudeToDB(stype="power")

    def preprocess_wav(self, wav_path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        spec = self.to_db(self.mel(waveform)).squeeze(0)

        if spec.size(1) < SPEC_LEN:
            spec = torch.nn.functional.pad(spec, (0, SPEC_LEN - spec.size(1)))
        else:
            spec = spec[:, :SPEC_LEN]

        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.unsqueeze(0).unsqueeze(0)

        return spec.numpy().astype("float32")

    def predict(self, wav_path: str):
        x = self.preprocess_wav(wav_path)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        return probs

# ==== PDF Generator ====
def generate_pdf(results: dict, file_path: str):
    c = canvas.Canvas(file_path, pagesize=LETTER)
    width, height = LETTER

    c.setFont("Helvetica-Bold", 20)
    c.setFillColorRGB(0.2, 0.4, 0.8)
    c.drawString(50, height - 60, "🎵 Instrument Classification Report")

    c.setFont("Helvetica-Bold", 14)
    c.setFillColorRGB(1,1,1)
    y = height - 120
    c.drawString(60, y, f"🎼 Predicted: {results['predicted_instrument']}")

    c.setFont("Helvetica", 12)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    y -= 50
    c.drawString(60, y, "Top 5 Predictions:")

    for i, item in enumerate(results["top_predictions"]):
        y -= 25
        c.setFillColorRGB(0.4, 0.6, 1.0 if i == 0 else 0.7)
        c.drawString(80, y, f"{item['instrument']}: {item['probability']:.1%}")

    c.save()

# ==== Enhanced Streamlit UI ====
st.set_page_config(
    page_title="🎵 Instrument Identifier Pro", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header with animation
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>🎵 Instrument Identifier Pro</h1>
    <p class='hero-text'>Upload any WAV file and discover its musical instrument with AI precision</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = "nsynth_instrument_family_cnn.onnx"

@st.cache_resource
def load_model():
    return NsynthOnnxWrapper(MODEL_PATH)

wrapper = load_model()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class='upload-area'>
        <h3 style='color: white; margin-bottom: 1rem;'>📁 Upload WAV File</h3>
        <p style='color: rgba(255,255,255,0.8); margin-bottom: 2rem;'>
            Drop your audio file here or click to browse. Supports all WAV formats.
        </p>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["wav"], key="main_uploader")

with col2:
    st.markdown("""
        <div style='background: rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem; text-align: center; height: 300px;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>🎹</div>
            <h3 style='color: white;'>Ready to Analyze</h3>
            <p style='color: rgba(255,255,255,0.7);'>AI Model Loaded</p>
        </div>
    """, unsafe_allow_html=True)

if uploaded_file:
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Audio player with enhanced styling
    st.markdown("""
    <div class='audio-player'>
    """, unsafe_allow_html=True)
    
    st.audio(uploaded_file, format="audio/wav")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Predict button with animation
    if st.button("🎯 Analyze Instrument", key="predict", help="Click to run AI analysis"):
        with st.spinner("🔮 AI is listening..."):
            time.sleep(1)  # Visual feedback
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Analyzing audio features... {i+1}%")
                time.sleep(0.02)
            
            status_text.text("🎉 Generating predictions!")
            time.sleep(0.5)
        
        # Run prediction
        probs = wrapper.predict(tmp_path)
        top_k = 5
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = {
            "predicted_class_index": int(top_indices[0]),
            "predicted_instrument": INSTR_FAMILY_MAP[int(top_indices[0])],
            "top_predictions": [
                {
                    "class_index": int(i),
                    "instrument": INSTR_FAMILY_MAP[int(i)],
                    "probability": float(probs[i])
                }
                for i in top_indices
            ]
        }
        
        # Results section
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # Top prediction highlight
        st.markdown(f"""
        <div class='top-prediction'>
            <h2 style='margin: 0;'>🎼 {results['predicted_instrument']}</h2>
            <p style='margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.9;'>
                Confidence: {probs[top_indices[0]]:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top predictions list
        st.markdown("### 📊 Top 5 Predictions")
        for i, pred in enumerate(results["top_predictions"]):
            conf_color = "#ff6b6b" if i == 0 else "#feca57" if i == 1 else "#667eea"
            st.markdown(f"""
            <div class='prediction-card'>
                <h4 style='margin: 0 0 0.5rem 0; color: {conf_color};'>
                    #{i+1} {pred['instrument']}
                </h4>
                <div class='progress-bar'>
                    <div class='progress-fill' style='width: {pred["probability"]*100}%'></div>
                </div>
                <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem; font-weight: 600;'>
                    {pred['probability']:.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # JSON output
        with st.expander("📋 Raw JSON Output", expanded=False):
            st.json(results)
        
        # Download section
        st.markdown("""
        <div class='download-section'>
            <h3 style='color: white; margin-bottom: 1rem;'>📄 Download Report</h3>
        """, unsafe_allow_html=True)
        
        # Generate and download PDF
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        generate_pdf(results, pdf_path)
        
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="💾 Download Professional PDF Report",
                data=f,
                file_name=f"instrument_report_{results['predicted_instrument'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # Close results-container

else:
    # Empty state with enhanced design
    st.markdown("""
    <div class='upload-area' style='min-height: 400px; display: flex; flex-direction: column; justify-content: center;'>
        <div style='font-size: 5rem; margin-bottom: 2rem;'>🎵</div>
        <h2 style='color: white; margin-bottom: 1rem;'>No file selected</h2>
        <p style='color: rgba(255,255,255,0.8); max-width: 400px; margin: 0 auto;'>
            Drag & drop your WAV file here or click to browse. 
            Our AI will instantly identify the musical instrument!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 3rem 0 1rem 0; color: rgba(255,255,255,0.7);'>
    <p>Powered by NSynth AI Model • Built with ❤️ for music lovers</p>
</div>
""", unsafe_allow_html=True)
