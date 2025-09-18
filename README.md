# Autoformer-Keras

Keras implementation of [Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008).

## ✨ Features
- Series Decomposition block (trend + seasonal separation)
- Auto-Correlation mechanism (FFT-based, replaces self-attention)
- Encoder-Decoder architecture in pure Keras/TensorFlow
- Ready for long-term time series forecasting

## 📦 Installation
```bash
git clone https://github.com/ksra7900/Autoformer-Keras.git
cd Autoformer-Keras
pip install -r requirements.txt

