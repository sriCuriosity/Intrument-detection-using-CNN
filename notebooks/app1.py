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

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Instrument Classification Report")

    c.setFont("Helvetica", 12)
    y = height - 100

    c.drawString(50, y, f"Predicted Instrument: {results['predicted_instrument']}")
    y -= 30

    c.drawString(50, y, "Top Predictions:")
    y -= 20

    for item in results["top_predictions"]:
        c.drawString(
            70, y,
            f"{item['instrument']} — Probability: {item['probability']:.3f}"
        )
        y -= 18

    c.save()

# ==== Streamlit UI ====
st.set_page_config(page_title="Instrument Identifier", layout="centered")
st.title("🎵 Musical Instrument Identifier")
st.write("Upload a WAV file and get JSON + PDF results.")

MODEL_PATH = "nsynth_instrument_family_cnn.onnx"

@st.cache_resource
def load_model():
    return NsynthOnnxWrapper(MODEL_PATH)

wrapper = load_model()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)

    if st.button("Predict Instrument"):
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

        # ✅ Show JSON output
        st.subheader("Prediction Output (JSON)")
        st.json(results)

        # Save PDF
        pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
        generate_pdf(results, pdf_path)

        # ✅ Download PDF
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name="instrument_prediction.pdf",
                mime="application/pdf"
            )
else:
    st.info("Please upload a WAV file to begin.")
