# 🎵 InstruNet AI 🎷

> **Unleashing the Power of CNNs to Decode the Symphony of Sounds**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**InstruNet AI** is a cutting-edge Deep Learning project designed to identify musical instruments from audio clips. By transforming soundwaves into visual spectrograms and analyzing them with a custom Convolutional Neural Network (CNN), InstruNet can distinguish between the soulful strums of a guitar, the rhythmic beats of drums, and the elegant keys of a piano.

---

## 🚀 Features

-   **🧠 Deep Learning Brain**: Powered by a multi-layer CNN architecture optimized for pattern recognition in audio spectrograms.
-   **👁️ Visual Audio Intelligence**: Converts raw audio processing into Mel-Spectrograms, allowing the model to "see" sound frequencies.
-   **🎹 Multi-Instrument Support**: Currently supports detection of **Piano**, **Guitar**, **Drums**, **Violin**, and **Flute**.
-   **📊 Interactive Dashboard**: Built with Streamlit for a seamless, drag-and-drop user experience. Visualize waveforms and heatmaps in real-time!

---

## 🛠️ Tech Stack

-   **Core**: Python, NumPy, Pandas
-   **Audio Processing**: Librosa
-   **Machine Learning**: TensorFlow / Keras
-   **Visualization**: Matplotlib, Seaborn
-   **Web App**: Streamlit

---

## 📸 Snapshots

*Visualize the sound before you analyze it.*

| Waveform | Mel Spectrogram |
|:---:|:---:|
| *(Signal Amplitude over Time)* | *(Frequency Intensity Heatmap)* |

*(Run the app to see these live!)*

---

## 🏃‍♂️ Getting Started

### 1. Clone the Conservatory
```bash
git clone https://github.com/Sri-Nithilan/Intrument-detection-using-CNN.git
cd Intrument-detection-using-CNN
```

### 2. Tune Your Environment
Install the necessary dependencies to get the band together.
```bash
pip install -r requirements.txt
```

### 3. Training Mode (Backstage)
Want to train the model from scratch? Run the training script:
```bash
python src/train.py
```
*This will train the CNN on your dataset and save the weights to `models/instrunet_model.h5`.*

### 4. Showtime (Run the App) 🎤
Launch the Streamlit dashboard to test the model:
```bash
streamlit run src/app.py
```
Open your browser to the local URL provided (usually `http://localhost:8501`) and upload your favorite audio clips!

---

## 📂 Project Structure

```bash
Intrument-detection-using-CNN/
├── 📂 data/             # Raw audio samples
├── 📂 models/           # Saved trained models (.h5)
├── 📂 notebooks/        # Jupyter notebooks for experimentation
├── 📂 src/              # Source code
│   ├── app.py           # Streamlit Web Application
│   ├── model.py         # CNN Architecture Definition
│   ├── preprocessing.py # Audio loading & Spectrogram generation
│   └── train.py         # Model training loop
├── 📂 tests/            # Unit tests
└── requirements.txt     # Dependencies
```

---

## 🔮 Future Roadmap

- [ ] 🎤 Real-time microphone input for live instrument detection.
- [ ] 🎻 Support for identifying multiple instruments playing simultaneously (Polyphonic detection).
- [ ] ☁️ Cloud deployment for universal access.

---

## 🤝 Contributing

Got a new instrument to add? Found a bug? Feel free to open an issue or submit a pull request!

---

*Crafted with 🎧 and 💻 by [Sri Nithilan]*
