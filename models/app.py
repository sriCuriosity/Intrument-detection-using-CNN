# app.py - FULLY FIXED VERSION
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
matplotlib.use('Agg')
from io import BytesIO
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal

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

# Custom CSS
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
                      border-left: 4px solid #ff6b6b; transition: all 0.3s ease; }
    
    .file-list-item:hover { background: rgba(255,255,255,0.2); transform: translateX(5px); }
    
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes floatIn { 0% { opacity: 0; transform: scale(0.9) translateY(20px); } 100% { opacity: 1; transform: scale(1) translateY(0); } }
    @keyframes slideInUp { from { opacity: 0; transform: translateY(50px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ==== ONNX Wrapper ====
class NsynthOnnxWrapper:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mel = MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, center=True, power=2.0
        )
        self.to_db = AmplitudeToDB(stype="power")

    def load_audio(self, wav_path: str):
        """Load and preprocess audio using librosa (more reliable on Windows)"""
        try:
            # Use librosa instead of torchaudio.load to avoid FFmpeg issues
            waveform, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            # Convert to torch tensor and add channel dimension
            waveform = torch.from_numpy(waveform).unsqueeze(0).float()
            return waveform
        except Exception as e:
            # Fallback to soundfile if librosa fails
            try:
                waveform, sr = sf.read(wav_path)
                if len(waveform.shape) > 1:  # Multi-channel
                    waveform = np.mean(waveform, axis=1)  # Convert to mono
                
                # Resample if needed (simple linear interpolation)
                if sr != SAMPLE_RATE:
                    waveform = signal.resample(waveform, int(len(waveform) * SAMPLE_RATE / sr))
                
                # Convert to torch tensor and add channel dimension
                waveform = torch.from_numpy(waveform).unsqueeze(0).float()
                return waveform
            except Exception as e2:
                raise Exception(f"Failed to load audio with both librosa and soundfile: {e}, {e2}")

    def compute_mel_spec(self, waveform):
        """Compute mel spectrogram"""
        spec = self.to_db(self.mel(waveform)).squeeze(0)
        return spec

    def preprocess_for_model(self, spec):
        """Prepare spectrogram for model input"""
        if spec.size(1) < SPEC_LEN:
            spec = torch.nn.functional.pad(spec, (0, SPEC_LEN - spec.size(1)))
        else:
            spec = spec[:, :SPEC_LEN]
        
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec = spec.unsqueeze(0).unsqueeze(0)
        return spec.numpy().astype("float32")

    def predict(self, wav_path: str):
        """Get predictions"""
        waveform = self.load_audio(wav_path)
        spec = self.compute_mel_spec(waveform)
        model_input = self.preprocess_for_model(spec)
        
        logits = self.session.run([self.output_name], {self.input_name: model_input})[0]
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]
        return probs

    def get_mel_spectrogram_for_display(self, wav_path: str):
        """Get mel spectrogram for visualization"""
        waveform = self.load_audio(wav_path)
        spec = self.compute_mel_spec(waveform)
        return spec.detach().numpy()

# ==== Visualization functions (NO CACHING) ====
def create_spectrogram_image(mel_spec):
    """Create spectrogram image - NOT CACHED"""
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

def create_waveform_image(waveform, sample_rate=SAMPLE_RATE):
    """Create amplitude vs time waveform visualization"""
    # Convert to numpy if tensor
    if torch.is_tensor(waveform):
        waveform_np = waveform.squeeze().numpy()
    else:
        waveform_np = waveform
    
    # Create time axis in seconds
    duration = len(waveform_np) / sample_rate
    time_axis = np.linspace(0, duration, len(waveform_np))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Full waveform
    ax1.plot(time_axis, waveform_np, color='#00d4ff', linewidth=0.5, alpha=0.8)
    ax1.set_title('Full Audio Waveform', fontsize=14, fontweight='bold', pad=15, color='white')
    ax1.set_xlabel('Time (seconds)', fontsize=11, color='white')
    ax1.set_ylabel('Amplitude', fontsize=11, color='white')
    ax1.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax1.set_facecolor('#1a1a1a')
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('white')
    ax1.spines['left'].set_color('white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Zoomed portion (first 0.1 seconds or available)
    zoom_duration = min(0.1, duration)
    zoom_samples = int(zoom_duration * sample_rate)
    zoom_time = time_axis[:zoom_samples]
    zoom_wave = waveform_np[:zoom_samples]
    
    ax2.plot(zoom_time, zoom_wave, color='#ff6b6b', linewidth=1.2, alpha=0.9)
    ax2.fill_between(zoom_time, zoom_wave, alpha=0.3, color='#ff6b6b')
    ax2.set_title(f'Zoomed View (First {zoom_duration:.2f} seconds)', fontsize=14, fontweight='bold', pad=15, color='white')
    ax2.set_xlabel('Time (seconds)', fontsize=11, color='white')
    ax2.set_ylabel('Amplitude', fontsize=11, color='white')
    ax2.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax2.set_facecolor('#1a1a1a')
    ax2.tick_params(colors='white')
    ax2.spines['bottom'].set_color('white')
    ax2.spines['left'].set_color('white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Add statistics text
    stats_text = f'Duration: {duration:.2f}s | Samples: {len(waveform_np):,} | Sample Rate: {sample_rate}Hz\n'
    stats_text += f'Max: {waveform_np.max():.3f} | Min: {waveform_np.min():.3f} | Mean: {waveform_np.mean():.3f}'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, color='white', 
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.patch.set_facecolor('#0a0a0a')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
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
        if y < 100:  # New page if running out of space
            c.showPage()
            y = height - 60
            
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0, 0, 0)
        c.drawString(60, y, f"File {i+1}: {results['filename']}")
        y -= 25
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(80, y, f"Predicted: {results['predicted_instrument']}")
        y -= 25
        
        c.setFont("Helvetica", 10)
        c.drawString(100, y, "Top 3 Predictions:")
        y -= 15
        
        for pred in results["top_predictions"][:3]:
            c.drawString(120, y, f"• {pred['instrument']}: {pred['probability']:.1%}")
            y -= 15
        
        y -= 10
        c.setStrokeColorRGB(0.8, 0.8, 0.8)
        c.line(50, y, width-50, y)
        y -= 25
    
    c.save()

# ==== Helper function to save uploaded file ====
def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp location and return path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# ==== Main App ====
st.set_page_config(page_title="🎵 Instrument Identifier Pro", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <h1>🎵 Instrument Identifier Pro</h1>
    <p class='hero-text'>Upload multiple WAV files • Waveform Analysis • Mel Spectrogram • AI-powered Classification</p>
</div>
""", unsafe_allow_html=True)

# Load model
MODEL_PATH = r"D:\Sri Nithilan\Documents\GitHub\Intrument-detection-using-CNN\models\nsynth_instrument_family_cnn.onnx"

@st.cache_resource
def load_model():
    return NsynthOnnxWrapper(MODEL_PATH)

try:
    wrapper = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_loaded = False

# Initialize session state
if 'file_data' not in st.session_state:
    st.session_state.file_data = {}  # {filename: {'file_obj': UploadedFile, 'temp_path': str}}
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Sidebar for file management
st.sidebar.header("📁 File Manager")

# File uploader
new_files = st.sidebar.file_uploader(
    "Upload WAV files", 
    type=["wav"], 
    accept_multiple_files=True, 
    key="multi_uploader"
)

# Add new files to session state
if new_files:
    for file in new_files:
        if file.name not in st.session_state.file_data:
            temp_path = save_uploaded_file(file)
            if temp_path:
                st.session_state.file_data[file.name] = {
                    'file_obj': file,
                    'temp_path': temp_path
                }
    
    if len(new_files) > 0:
        st.sidebar.success(f"✅ Added {len(new_files)} file(s)!")

# Display current files
if st.session_state.file_data:
    st.sidebar.markdown("**📋 Current Files:**")
    
    file_names = list(st.session_state.file_data.keys())
    
    for i, filename in enumerate(file_names):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            if st.button(f"📄 {filename}", key=f"select_{i}", use_container_width=True):
                st.session_state.selected_file = filename
        with col2:
            if st.button("🗑️", key=f"delete_{i}"):
                # Clean up temp file
                temp_path = st.session_state.file_data[filename]['temp_path']
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
                del st.session_state.file_data[filename]
                if st.session_state.selected_file == filename:
                    st.session_state.selected_file = None
                st.rerun()
    
    st.sidebar.markdown("---")
    
    # Clear all button
    if st.sidebar.button("🗑️ Clear All Files", use_container_width=True):
        # Clean up all temp files
        for data in st.session_state.file_data.values():
            try:
                if os.path.exists(data['temp_path']):
                    os.unlink(data['temp_path'])
            except:
                pass
        st.session_state.file_data = {}
        st.session_state.selected_file = None
        st.session_state.results = None
        st.rerun()
    
    # Set default selection if none
    if st.session_state.selected_file is None and file_names:
        st.session_state.selected_file = file_names[0]

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.file_data and st.session_state.selected_file:
        selected_data = st.session_state.file_data[st.session_state.selected_file]
        
        st.success(f"📂 Selected: **{st.session_state.selected_file}**")
        
        # File tabs - ADDED WAVEFORM TAB
        tab1, tab2, tab3 = st.tabs(["🎵 Audio Preview", "📊 Waveform", "🌈 Spectrogram"])
        
        with tab1:
            # Audio player - use file object directly
            st.audio(selected_data['file_obj'], format="audio/wav")
        
        with tab2:
            if model_loaded:
                try:
                    with st.spinner("Generating waveform..."):
                        waveform = wrapper.load_audio(selected_data['temp_path'])
                        waveform_img = create_waveform_image(waveform)
                    
                    st.markdown("""
                    <div class='spectrogram-container'>
                        <h3 style='color: white; text-align: center; margin-bottom: 1rem;'>📈 Amplitude vs Time</h3>
                    """, unsafe_allow_html=True)
                    
                    st.image(waveform_img, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating waveform: {e}")
            else:
                st.warning("⚠️ Model not loaded. Cannot generate waveform.")
        
        with tab3:
            if model_loaded:
                try:
                    with st.spinner("Generating spectrogram..."):
                        mel_spec = wrapper.get_mel_spectrogram_for_display(selected_data['temp_path'])
                        spectrogram_img = create_spectrogram_image(mel_spec)
                    
                    st.markdown("""
                    <div class='spectrogram-container'>
                        <h3 style='color: white; text-align: center; margin-bottom: 1rem;'>🔬 Mel Spectrogram</h3>
                    """, unsafe_allow_html=True)
                    
                    st.image(spectrogram_img, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating spectrogram: {e}")
            else:
                st.warning("⚠️ Model not loaded. Cannot generate spectrogram.")
    
    elif st.session_state.file_data:
        st.info("👈 Select a file from the sidebar to preview")
    
    else:
        st.markdown("""
        <div class='upload-area' style='min-height: 500px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='font-size: 6rem; margin-bottom: 2rem;'>🎵</div>
            <h2 style='color: white; margin-bottom: 1rem;'>Upload WAV Files</h2>
            <p style='color: rgba(255,255,255,0.8); max-width: 500px; margin: 0 auto;'>
                Use the sidebar to upload multiple WAV files. View waveforms, spectrograms, and get AI predictions!
            </p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    file_count = len(st.session_state.file_data)
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.1); border-radius: 20px; padding: 2rem; 
                text-align: center; min-height: 350px; display: flex; flex-direction: column; justify-content: center;'>
        <div style='font-size: 5rem; margin-bottom: 1rem;'>🎹</div>
        <h3 style='color: white;'>{'Ready to Analyze' if model_loaded else 'Model Error'}</h3>
        <p style='color: rgba(255,255,255,0.7);'>
            {file_count} file(s) loaded<br/>
            {'✅ Model Ready' if model_loaded else '❌ Model Failed'}
        </p>
    </div>
    """, unsafe_allow_html=True)

# Batch analysis button
if st.session_state.file_data and model_loaded:
    st.markdown("---")
    
    if st.button("🚀 Analyze All Files", key="batch_predict", use_container_width=True):
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        file_list = list(st.session_state.file_data.items())
        
        for i, (filename, data) in enumerate(file_list):
            progress_bar.progress((i + 1) / len(file_list))
            status_text.text(f"Analyzing {filename}... {i+1}/{len(file_list)}")
            
            try:
                # Get predictions
                probs = wrapper.predict(data['temp_path'])
                
                top_k = 5
                top_indices = np.argsort(probs)[::-1][:top_k]

                results = {
                    "filename": filename,
                    "predicted_class_index": int(top_indices[0]),
                    "predicted_instrument": INSTR_FAMILY_MAP[int(top_indices[0])],
                    "top_predictions": [
                        {
                            "class_index": int(idx), 
                            "instrument": INSTR_FAMILY_MAP[int(idx)], 
                            "probability": float(probs[idx])
                        }
                        for idx in top_indices
                    ]
                }
                all_results.append(results)
                
            except Exception as e:
                st.error(f"Error analyzing {filename}: {e}")
                continue
        
        progress_bar.empty()
        status_text.success("🎉 Batch analysis complete!")
        
        # Store results in session state
        st.session_state.results = all_results

# Display results if available
if st.session_state.results:
    st.markdown("---")
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    st.markdown("### 📈 Batch Results Summary")
    
    # Create summary table
    summary_data = []
    for results in st.session_state.results:
        top_prob = results["top_predictions"][0]["probability"]
        summary_data.append({
            "File": results["filename"],
            "Predicted Instrument": results["predicted_instrument"],
            "Confidence": f"{top_prob:.1%}"
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Detailed results expander
    with st.expander("📊 View Detailed Results"):
        for i, results in enumerate(st.session_state.results):
            st.markdown(f"#### File {i+1}: {results['filename']}")
            st.markdown(f"**🎼 Predicted: {results['predicted_instrument']}**")
            
            st.markdown("**Top 5 Predictions:**")
            for j, pred in enumerate(results["top_predictions"], 1):
                st.write(f"{j}. {pred['instrument']}: {pred['probability']:.2%}")
            
            st.markdown("---")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        json_str = json.dumps(st.session_state.results, indent=2)
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name="batch_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # PDF download
        try:
            pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            generate_pdf(st.session_state.results, pdf_path)
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name="batch_instrument_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            # Clean up PDF
            try:
                os.unlink(pdf_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 3rem 0 1rem 0; color: rgba(255,255,255,0.7);'>
    <p>✅ Enhanced! • Waveform Analysis • Mel Spectrogram • NSynth AI • Multi-file Support</p>
</div>
""", unsafe_allow_html=True)