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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from io import BytesIO
import pandas as pd

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

# Custom CSS (same as before)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    
    h1 { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 3rem !important;
         background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%); -webkit-background-clip: text;
         -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1rem;
         animation: fadeInUp 1s ease-out; }
    
    .hero-text { font-family: 'Inter', sans-serif; font-weight: 400; font-size: 1.3rem; color: rgba(255,255,255,0.9);
                 text-align: center; margin-bottom: 3rem; animation: fadeInUp 1s ease-out 0.2s both; }
    
    .upload-area { background: rgba(255,255,255,0.15); backdrop-filter: blur(20px); border: 2px dashed rgba(255,255,255,0.3);
                   border-radius: 20px; padding: 3rem 2rem; text-align: center; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                   animation: floatIn 1.2s ease-out; }
    
    .upload-area:hover { background: rgba(255,255,255,0.25); border-color: rgba(255,255,255,0.6);
                         transform: translateY(-5px); box-shadow: 0 25px 50px rgba(0,0,0,0.2); }
    
    .predict-btn { background: linear-gradient(135deg, #ff6b6b, #feca57); border: none; border-radius: 15px;
                   padding: 1rem 3rem; font-size: 1.2rem; font-weight: 600; font-family: 'Inter', sans-serif;
                   color: white; width: 100%; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden;
                   animation: slideInUp 1s ease-out 0.4s both; }
    
    .predict-btn:hover { transform: translateY(-3px); box-shadow: 0 20px 40px rgba(255,107,107,0.4); }
    
    .results-container { background: rgba(255,255,255,0.95); backdrop-filter: blur(20px); border-radius: 25px;
                        padding: 2.5rem; box-shadow: 0 30px 60px rgba(0,0,0,0.25); animation: slideInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
                        margin-top: 2rem; }
    
    .spectrogram-container { background: rgba(0,0,0,0.3); border-radius: 20px; padding: 2rem; margin: 2rem 0;
                            backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2); }
    
    .file-list-item { background: rgba(255,255,255,0.1); border-radius: 12px; padding: 1rem; margin: 0.5rem 0;
                      border-left: 4px solid #ff6b6b; transition: all 0.3s ease; cursor: pointer; }
    
    .file-list-item:hover { background: rgba(255,255,255,0.2); transform: translateX(5px); }
    
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes floatIn { 0% { opacity: 0; transform: scale(0.9) translateY(20px); } 100% { opacity: 1; transform: scale(1) translateY(0); } }
    @keyframes slideInUp { from { opacity: 0; transform: translateY(50px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ==== FIXED ONNX wrapper ====
class NsynthOnnxWrapper:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, center=True, power=2.0
        )
        self.to_db = AmplitudeToDB(stype="power")

    def preprocess_wav(self, wav_path: str):
        """Returns ONLY the model input tensor (fixed)"""
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

    def get_mel_spectrogram(self, wav_path: str):
        """Separate method for spectrogram visualization (NEW)"""
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        spec = self.to_db(self.mel(waveform)).squeeze(0)
        return spec.detach().numpy()

    def predict(self, wav_path: str):
        """Returns ONLY probabilities (fixed)"""
        x = self.preprocess_wav(wav_path)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        return probs

# ==== Spectrogram visualization ====
@st.cache_data
def create_spectrogram_image(mel_spec):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Mel Spectrogram', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Frequency Bins')
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='black')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==== PDF Generator ====
def generate_pdf(results_list, file_path: str):
    c = canvas.Canvas(file_path, pagesize=LETTER)
    width, height = LETTER

    c.setFont("Helvetica-Bold", 20)
    c.setFillColorRGB(0.2, 0.4, 0.8)
    c.drawString(50, height - 60, "🎵 Batch Instrument Classification Report")

    y = height - 120
    for i, results in enumerate(results_list):
        c.setFont("Helvetica-Bold", 16)
        c.setFillColorRGB(1,1,1)
        c.drawString(60, y, f"File {i+1}: {results['filename']}")
        y -= 25
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(80, y, f"🎼 {results['predicted_instrument']}")
        y -= 35
        
        c.setFont("Helvetica", 11)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        for j, pred in enumerate(results["top_predictions"][:3]):
            c.drawString(100, y, f"{pred['instrument']}: {pred['probability']:.1%}")
            y -= 20
        
        y -= 10
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(50, y, width-50, y)
        y -= 30
    
    c.save()

# ==== Main App ====
st.set_page_config(page_title="🎵 Instrument Identifier Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>🎵 Instrument Identifier Pro</h1>
    <p class='hero-text'>Upload multiple WAV files and analyze with AI-powered Mel Spectrogram visualization</p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = "nsynth_instrument_family_cnn.onnx"
@st.cache_resource
def load_model():
    return NsynthOnnxWrapper(MODEL_PATH)

wrapper = load_model()

# Session state for uploaded files
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar for file management
st.sidebar.header("📁 File Manager")
new_files = st.sidebar.file_uploader("Upload WAV files", type=["wav"], accept_multiple_files=True, key="multi_uploader")

if new_files:
    for file in new_files:
        if file not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(file)
    st.success(f"✅ Added {len(new_files)} new files!")
    st.rerun()

if st.session_state.uploaded_files:
    st.sidebar.markdown("**📋 Current Files:**")
    for i, file in enumerate(st.session_state.uploaded_files):
        st.sidebar.markdown(f"""
        <div class='file-list-item'>
            📄 {file.name}
            <span style='float: right; color: #ff6b6b;'>✓</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Remove file button
    if st.sidebar.button("🗑️ Clear All Files"):
        st.session_state.uploaded_files = []
        st.rerun()
    
    selected_file_idx = st.sidebar.selectbox("Select file for preview:", 
                                           range(len(st.session_state.uploaded_files)),
                                           format_func=lambda x: st.session_state.uploaded_files[x].name)
else:
    selected_file_idx = None

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.uploaded_files:
        st.success(f"✅ {len(st.session_state.uploaded_files)} files ready for analysis!")
        
        # File tabs for individual preview
        tab1, tab2 = st.tabs(["🎵 Audio Preview", "📊 Spectrogram Preview"])
        
        with tab1:
            if selected_file_idx is not None:
                selected_file = st.session_state.uploaded_files[selected_file_idx]
                st.audio(selected_file, format="audio/wav")
            else:
                st.info("👈 Select a file from sidebar to preview")
        
        with tab2:
            if selected_file_idx is not None:
                selected_file = st.session_state.uploaded_files[selected_file_idx]
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(selected_file.read())
                    tmp_path = tmp.name
                
                # FIXED: Use separate spectrogram method
                mel_spec = wrapper.get_mel_spectrogram(tmp_path)
                spectrogram_img = create_spectrogram_image(mel_spec)
                
                st.markdown("""
                <div class='spectrogram-container'>
                    <h3 style='color: white; text-align: center; margin-bottom: 1rem;'>🔬 Mel Spectrogram</h3>
                """, unsafe_allow_html=True)
                
                st.image(spectrogram_img, caption="Mel Spectrogram Visualization", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("👈 Select a file from sidebar to see spectrogram")
    else:
        st.markdown("""
        <div class='upload-area' style='min-height: 500px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 6rem; margin-bottom: 2rem;'>🎵</div>
            <h2 style='color: white; margin-bottom: 1rem;'>Upload WAV Files</h2>
            <p style='color: rgba(255,255,255,0.8); max-width: 500px; margin: 0 auto;'>
                Drag & drop multiple WAV files or click to browse. View Mel Spectrograms and get AI predictions!
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem; 
                text-align: center; height: 350px; display: flex; flex-direction: column; justify-content: center;'>
        <div style='font-size: 5rem; margin-bottom: 1rem;'>🎹</div>
        <h3 style='color: white;'>Ready to Analyze</h3>
        <p style='color: rgba(255,255,255,0.7);'>AI Model Loaded • Multiple file support</p>
    </div>
    """, unsafe_allow_html=True)

# Batch prediction - FIXED
if st.session_state.uploaded_files and st.button("🚀 Analyze All Files", key="batch_predict"):
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(st.session_state.uploaded_files):
        progress_bar.progress((i + 1) / len(st.session_state.uploaded_files))
        status_text.text(f"Analyzing {uploaded_file.name}... {i+1}/{len(st.session_state.uploaded_files)}")
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        # FIXED: Use correct methods separately
        probs = wrapper.predict(tmp_path)
        top_k = 5
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = {
            "filename": uploaded_file.name,
            "predicted_class_index": int(top_indices[0]),
            "predicted_instrument": INSTR_FAMILY_MAP[int(top_indices[0])],
            "top_predictions": [
                {"class_index": int(idx), "instrument": INSTR_FAMILY_MAP[int(idx)], "probability": float(probs[idx])}
                for idx in top_indices
            ]
        }
        all_results.append(results)
    
    status_text.text("🎉 Batch analysis complete!")
    
    # Results display (same as before)
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    st.markdown("### 📈 Batch Results Summary")
    summary_data = []
    for results in all_results:
        top_prob = next(p['probability'] for p in results["top_predictions"] if p['class_index'] == results['predicted_class_index'])
        summary_data.append({
            "File": results["filename"][:20] + "..." if len(results["filename"]) > 20 else results["filename"],
            "Predicted": results["predicted_instrument"],
            "Confidence": f"{top_prob:.1%}"
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Download batch report
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    generate_pdf(all_results, pdf_path)
    
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📄 Download Batch PDF Report",
            data=f,
            file_name="batch_instrument_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 3rem 0 1rem 0; color: rgba(255,255,255,0.7);'>
    <p>✅ Fixed! Powered by NSynth AI • Mel Spectrogram Visualization • Multi-file Support</p>
</div>
""", unsafe_allow_html=True)
