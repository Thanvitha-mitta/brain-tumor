# 🧠 Brain Tumor Classification with Explainable AI (Streamlit)

A deep learning web application that classifies brain MRI images into four tumor types — **glioma**, **meningioma**, **pituitary**, and **no tumor** — using a pre-trained **CNN model**.  
It also provides **Grad-CAM visualizations** for explainability and generates **personalized PDF reports**.

---

## 🚀 Features

- 🧩 Multi-class brain tumor classification  
- 🌈 Explainable AI with Grad-CAM visualizations  
- 🖥️ Interactive, modern web interface using Streamlit  
- 🧾 Downloadable prediction reports in PDF  
- 🎉 Celebration animation for **No Tumor** results  
- ⚡ Lightweight, fast, and easy to use  

---

## 🗂️ Project Structure
```
brain-tumor/
├── dataset/
│   ├── Training/              # Training dataset
│   └── Testing/               # Testing dataset
├── brain_tumor_app.py         # Streamlit web app
├── train_brain_tumor_model.py # Model training script
├── grad_cam_utils.py          # Grad-CAM utility (optional)
├── brain_tumor_model.h5       # Pre-trained CNN model
├── requirements.txt           # Dependencies list
└── README.md                  # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create and activate a virtual environment

```powershell
# Create virtual environment
python -m venv .venv

# Allow script execution (Windows only)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force

# Activate environment
.venv\Scripts\Activate.ps1
```

---

### 2️⃣ Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:

```bash
pip install streamlit tensorflow opencv-python pillow matplotlib numpy reportlab
```

---

## ▶️ Running the App

```powershell
streamlit run brain_tumor_app.py
```

Then open the provided local URL (usually `http://localhost:8501`) in your browser.

---

## 🧠 How to Use

1. Upload an **MRI image** (`.jpg`, `.jpeg`, or `.png`)  
2. Click **🔍 Predict**  
3. The app will:
   - 🧠 Predict the tumor type  
   - 📊 Show the confidence score  
   - 🔥 Display a **Grad-CAM heatmap**  
4. You can also **download a PDF report** of the prediction.

> 💥 If **no tumor** is detected, the app displays a 🎉 **party popper animation** to celebrate good news!

---

## 🏗️ Model Architecture

- **Input size:** 150×150×3 RGB  
- **Layers:**
  - 3× Convolution + ReLU + MaxPooling  
  - Dropout for regularization  
  - Dense layers for classification  
- **Output:** 4 classes  
  (`glioma`, `meningioma`, `no tumor`, `pituitary`)

---

## 🧩 Explainability with Grad-CAM

The application uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize which regions of the MRI influenced the model’s decision.  
This improves **trust and interpretability** in medical AI systems.

---

## 📄 Example Output

| MRI Image                 | Grad-CAM Heatmap                | Prediction | Confidence |
| -------------------------- | ------------------------------- | ----------- | ----------- |
| ![MRI](example_input.jpg) | ![GradCAM](example_heatmap.jpg) | Glioma      | 97.8%       |

---

## ⚠️ Troubleshooting

### 🧩 TensorFlow not installing?

- Use Python **3.8–3.11**
- Run:
  ```bash
  pip install --upgrade pip
  ```
- For CPU-only:
  ```bash
  pip install tensorflow-cpu
  ```

### 📦 ReportLab not found?

If you see:
```
ModuleNotFoundError: No module named 'reportlab'
```

Run:
```bash
pip install reportlab
```

### 🧠 Model file missing?

Ensure `brain_tumor_model.h5` exists in the same directory as `brain_tumor_app.py`.  
If not, retrain it using:
```bash
python train_brain_tumor_model.py
```

---

## 🧰 Dependencies

- Streamlit  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Pillow  
- Matplotlib  
- ReportLab  


## 🌟 Acknowledgments

- MRI datasets from **Kaggle Brain Tumor Dataset**  
- TensorFlow & Keras open-source frameworks  
- Streamlit community for interactive UI tools  

---
