# DepHNN: Depression Detection using Hybrid CNN-LSTM

**DepHNN** is a deep learning-based system designed to detect Major Depressive Disorder (MDD) using Electroencephalogram (EEG) signals. By employing a hybrid architecture combining Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal analysis, the system achieves high accuracy in distinguishing between depressed and healthy subjects.

## 🚀 Features
* **Automated EEG Analysis:** Upload raw `.edf` files directly via a web interface.
* **Hybrid Deep Learning Model:** Uses CNN to detect brain region anomalies and LSTM to track signal changes over time.
* **Advanced Preprocessing:** Includes artifact removal, filtering (0.5Hz - 80Hz), and Z-Score normalization.
* **Clinical Biomarker Detection:** Analyzing Frontal Alpha Asymmetry and Theta/Beta ratios.
* **PDF Reporting:** Generates a downloadable medical report with the diagnosis and confidence score.
* **Secure User System:** Django-based authentication for patient/doctor data privacy.

## 🛠️ Tech Stack
* **Frontend:** HTML, CSS, Bootstrap, JavaScript
* **Backend:** Django (Python)
* **Database:** SQLite
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** MNE-Python, NumPy, Scikit-learn
* **Visualization:** Matplotlib

## 🧠 Model Architecture
The core of the project is a **Hybrid CNN-LSTM** model:
1.  **Input Layer:** Receives 19-channel EEG signals.
2.  **1D Convolutional Layers:** Extract spatial features and local patterns from brain regions.
3.  **LSTM Layers:** Capture long-term temporal dependencies in the brainwaves.
4.  **Dropout Layers:** Applied to prevent overfitting.
5.  **Dense Layers & Softmax:** Perform the final binary classification (Depressed vs. Healthy).

**Optimization:**
* **Optimizer:** Adam (Adaptive Moment Estimation) for fast convergence.
* **Loss Function:** Huber Loss to handle noise and outliers in EEG data robustly.

## 📂 Dataset Details
The model was trained on EEG data with the following specifications:
* **Format:** standard `.edf` (European Data Format)
* **Channels:** 19 electrodes (10-20 international system)
* **Sampling Rate:** 256 Hz
* **Units:** Microvolts ($\mu V$)
* **Filtering:** High-pass (0.5 Hz) and Low-pass (80 Hz) applied.

## 🔧 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/DepHNN-Depression-Detection.git](https://github.com/YOUR-USERNAME/DepHNN-Depression-Detection.git)
    cd DepHNN-Depression-Detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    # Windows
    .\env\Scripts\activate
    # Mac/Linux
    source env/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run migrations:**
    ```bash
    python manage.py migrate
    ```

5.  **Start the server:**
    ```bash
    python manage.py runserver
    ```
    Access the app at `http://127.0.0.1:8000/`

## 🔮 Future Enhancements
* **Severity Estimation:** Upgrading the model to classify Mild, Moderate, and Severe depression.
* **Longitudinal Monitoring:** Tracking patient progress over weeks to evaluate treatment efficacy.
* **Multi-Modal Integration:** Combining EEG with Heart Rate Variability (HRV) and facial micro-expressions for holistic diagnosis.

## 👥 Authors
* **Mareddy Charan** - *Lead Developer*