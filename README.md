# CodingBara 🦫🍊

> *An emotional support coding companion that helps aspiring programmers debug code, stay productive, and stay calm—wrapped in a cute Capybara package.*

---

## 📌 Project Overview

Coding is hard, and debugging often leads to cognitive fatigue, frustration, and burnout. **CodingBara** is an AI-powered physical companion and web app designed to act as an intelligent coding buddy to help aspiring coders with debugging with emotional support[cite: 1]. By monitoring your vocal tone and facial expressions, CodingBara can know emotions through audio & video and respond accordingly[cite: 1].

### Key Features
* **Dual-Modal Emotion Sensing:** ML includes SER & FER training[cite: 1].
* **Capybara AI Persona:** Calls the Gemini API to respond with soothing, constructive debugging guidance[cite: 1].
* **Embedded Hardware Unit:** Robotics component features an RPi w/ speaker, mic, LCD[cite: 1].
* **Companion Web App:** WebApp includes a Django backend w/ HTML/CSS/JS frontend to help w/ productivity[cite: 1]. It features a Pomodoro timer in a cute capybara package[cite: 1].

---

## 🛠 System Architecture & Components

The project consists of three core engineering pillars[cite: 1]:

```text
                     ┌─────────────────────────────┐
                     │   Physical Hardware Unit    │
                     │  (RPi, Mic, PiCam, Speaker) │
                     └──────────────┬──────────────┘
                                    │ 3s Audio + Photo
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ML Pipeline                                   │
│  ┌─────────────────────────────────┐   ┌─────────────────────────────┐  │
│  │ Speech Emotion Recognition (SER)│   │ Facial Emotion Recog. (FER) │  │
│  │  Custom Mel-Spectrogram CNN     │   │  Custom 4-Block 96x96 CNN   │  │
│  └────────────────┬────────────────┘   └──────────────┬──────────────┘  │
└───────────────────┼───────────────────────────────────┼─────────────────┘
                    └─────────────────┬─────────────────┘
                                      │ Emotion Class Vectors (0-6)
                                      ▼
                      ┌─────────────────────────────┐
                      │    Gemini API Orchestrator  │
                      │  (Capybara Debugging Prompt)│
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │ Companion Django Web Application│
                      │  (Pomodoro Timer & To-Do)   │
                      └─────────────────────────────┘

```

---

## 🧠 Machine Learning Development

CodingBara classifies human emotional states across 7 primary categories:
`0: Angry`, `1: Disgust`, `2: Fear`, `3: Happy`, `4: Neutral`, `5: Sad`, `6: Surprise`

### 1. Speech Emotion Recognition (SER)

* **Pipeline:** Audio recorded via the microphone is standardized, converted to 2D Mel Spectrograms using `librosa`, and passed to a CNN classifier.
* **Datasets (~39k total audio samples):** CREMA-D, RAVDESS, TESS, JL-Corpus, MELD.
* **Engineering Pivots & Debugging:**
* **RAM Optimization:** Switched training from Google Colab (due to memory limits when flattening spectrograms) to local GPU/VS Code execution.
* **Overfitting Resolution:** Initial models overfit heavily (98% train vs. 66% test). Added dropout layers, augmented audio data, and reduced model width to improve generalization.
* Achieved **~85% test accuracy** across all 7 emotions.



### 2. Facial Emotion Recognition (FER)

* **Pipeline:** 96x96 grayscale face images processed through a custom 4-block CNN.
* **Dataset:** Combined FER2013 + RAF-DB datasets (~49.8k samples).
* **Architecture Highlights:**
* Integrated `GlobalAveragePooling2D()` to eliminate standard parameter explosion caused by `Flatten()` layers, keeping total model size around **35.5 MB**.
* Dynamic on-the-fly augmentation (`RandomFlip`, `RandomRotation`, `RandomTranslation`) with `fill_mode="nearest"` to prevent boundary distortion.
* Achieved **~73% test accuracy** across all 7 emotions.



---

## 🤖 Physical Companion (Robotics)

* **Hardware Components:**
* Raspberry Pi (Main compute module)


* USB Microphone & Mini Speaker (Voice interaction)


* Raspberry Pi Camera Module (PiCam - Visual capture)
* OLED / LCD Screen (Status & expressive face animations)


* Push Button (Interaction trigger)


* **Chassis Iterations:**
1. *CAD Concept:* 3D model designed in Blender/Onshape for 3D printing.
2. *Clay/Airbrush:* Prototype shell modeling.
3. *Final Build:* Lightweight cardboard structural base with paper and acrylic paint covering crafted into a capybara shape.

<img width="480" height="600" alt="IMG_8165" src="https://github.com/user-attachments/assets/147ad219-fb6a-46a8-8632-4dd9c5e127bd" />

---

## 💻 Web Application

Built using a **Django backend w/ HTML/CSS/JS frontend** with **Firebase** connectivity:

* **Pomodoro Productivity Timer:** Custom study session timer configured specifically for coding focus intervals.


* **To-Do List:** Integrated task tracker for managing debugging steps and session goals.

<img width="1470" height="800" alt="Screenshot 2026-09-27 at 10 01 08 AM" src="https://github.com/user-attachments/assets/2a0991b0-a868-4d64-83b0-5f7ca6a88ef8" />

* Code to the website + more details here: https://github.com/Cici3939/CodingBaraWeb

---

## 🚀 Quickstart & Setup

### Prerequisites

* Python 3.9+
* TensorFlow 2.x & PyTorch
* Librosa, OpenCV, NumPy, Pillow, Scikit-learn

### Installation

```bash
# Clone the repository
git clone [https://github.com/Cici3939/CodingBara.git](https://github.com/Cici3939/CodingBara.git)
cd CodingBara

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install tensorflow torch librosa opencv-python pillow scikit-learn django

```

### Running Emotion Classification (Inference Test)

```bash
# Preprocess audio/image input and run emotion prediction
python classify.py

```

### Starting the Web App

```bash
cd webapp/
python manage.py runserver

```

---

## 📄 License & Acknowledgments

Developed by **Cici Xing** and **Sarah Xu**.

Special thanks to open-source speech and vision emotion datasets (CREMA-D, RAVDESS, TESS, JL-Corpus, FER2013, RAF-DB).

```
