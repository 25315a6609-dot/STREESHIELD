# 🛡️ STREESHIELD

## AI-Powered Deepfake Detection System

STREESHIELD is an AI-assisted deepfake detection system that analyzes both images and videos and provides prediction, confidence, AI-assisted interpretation, risk assessment, reporting, and conversational assistance.

---

## 🚀 Features

- Image-based deepfake detection using Basic CNN
- Video-based deepfake detection using 3D CNN
- Face detection and preprocessing using OpenCV
- 16-frame video sequence processing
- REAL / FAKE prediction
- Confidence score
- AI-powered analysis and explanation
- Risk assessment
- AI detection reports
- AI Assistant / chatbot
- Professional Streamlit interface
- Light and dark theme support

---

## 🏗️ System Architecture

```text
                    STREESHIELD
                         │
                ┌────────┴────────┐
                │                 │
             IMAGE              VIDEO
                │                 │
             Basic CNN          3D CNN
                │                 │
                └────────┬────────┘
                         ↓
                 REAL / FAKE
                 + Confidence
                         ↓
                  AI Analysis
                         ↓
                  Risk Assessment
                         ↓
                    AI Report
                         ↓
                  AI Assistant
                         ↓
                 Streamlit UI