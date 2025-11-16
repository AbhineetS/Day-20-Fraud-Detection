# 🛡️ Day 20 — Credit Card Fraud Detection using Autoencoders (Deep Learning)

This project focuses on detecting fraudulent transactions using an **unsupervised deep learning Autoencoder**.  
Fraud datasets are typically **highly imbalanced**, so instead of predicting fraud directly, the model learns the pattern of *normal transactions* and flags anomalies based on reconstruction error — a technique widely used in **banking and fintech**.

---

## 🚀 Overview
- Trained a **Deep Autoencoder** for anomaly detection  
- Used synthetic data simulating a **real-world imbalance** (5000 normal, 200 fraud)  
- Measured performance using **ROC-AUC, Precision, Recall, and F1-score**  
- Visualized learned representations using **t-SNE**  
- Saved model as `.keras` file  
- Achieved **ROC-AUC = 1.00** on synthetic dataset  

---

## 🧠 Workflow

1. **Dataset Loading** — Uses `creditcard.csv` if available, otherwise generates synthetic data  
2. **Preprocessing** — Normalizes features using StandardScaler  
3. **Autoencoder Training** — Learns patterns of non-fraudulent transactions  
4. **Reconstruction Error Calculation** — Higher error = more likely fraud  
5. **Thresholding** — Classifies fraud using optimized cutoff  
6. **Evaluation** — Generates classification report + ROC-AUC  
7. **Visualization** — t-SNE plot to show fraud vs normal separation  

---

## 📊 Results

### **Autoencoder Performance**
| Metric | Value |
|--------|--------|
| **ROC-AUC** | 1.00 |
| **Accuracy** | 98.85% |
| **Fraud Recall** | 1.00 |
| **Fraud Precision** | 0.77 |

🎯 *Perfect recall means the model didn’t miss any fraudulent transactions.*

---

## 🧩 Tech Stack
Python | Pandas | NumPy | Scikit-learn | TensorFlow/Keras | Matplotlib | Seaborn  

---

## 🧠 Key Concepts

- **Autoencoders:** Neural networks trained to reconstruct input → useful for anomaly detection  
- **Reconstruction Error:** Large difference = anomalous transaction  
- **Class Imbalance:** Fraud cases form <1% of real-world datasets  
- **ROC-AUC:** Measures model discrimination capability  
- **t-SNE Visualization:** Shows latent separation of fraud vs normal  

---

## 🔗 Connect

💼 **LinkedIn:** https://www.linkedin.com/in/abhineet-s  
📁 **GitHub Repository:** https://github.com/AbhineetS/Day-20-Fraud-Detection