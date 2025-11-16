import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras import layers, models


# -------------------------------------------------------
# 1. Load Dataset (from creditcard.csv or generate synthetic)
# -------------------------------------------------------
def load_or_create_data():
    if os.path.exists("creditcard.csv"):
        print("📦 Loading local dataset: creditcard.csv")
        df = pd.read_csv("creditcard.csv")
    else:
        print("⚠️ No dataset found — generating synthetic demo data...")
        normal = np.random.normal(0, 1, (5000, 30))
        fraud = np.random.normal(3, 1, (200, 30))
        df = pd.DataFrame(np.vstack([normal, fraud]))
        df["Class"] = [0] * 5000 + [1] * 200

    print(df.head())
    return df


# -------------------------------------------------------
# 2. Preprocessing
# -------------------------------------------------------
def preprocess(df):
    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train ONLY on normal transactions
    X_normal = X_scaled[y == 0]

    print(f"Normal samples: {X_normal.shape[0]}, Fraud samples: {sum(y)}")

    return X_scaled, X_normal, y, scaler


# -------------------------------------------------------
# 3. Create Autoencoder Model
# -------------------------------------------------------
def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(input_dim, activation="linear")
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()
    return model


# -------------------------------------------------------
# 4. Train Autoencoder
# -------------------------------------------------------
def train_autoencoder(autoencoder, X_normal):
    history = autoencoder.fit(
        X_normal, X_normal,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    # Save training curve
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.close()

    return autoencoder


# -------------------------------------------------------
# 5. Evaluate with Reconstruction Error
# -------------------------------------------------------
def evaluate_model(autoencoder, X_scaled, y):
    recon = autoencoder.predict(X_scaled)
    errors = np.mean(np.abs(X_scaled - recon), axis=1)

    auc = roc_auc_score(y, errors)
    print(f"\n🎯 ROC-AUC: {auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, errors)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    # Thresholding
    threshold = np.percentile(errors, 95)
    preds = (errors > threshold).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y, preds, digits=4))

    return errors, preds


# -------------------------------------------------------
# 6. Optional t-SNE Visualization
# -------------------------------------------------------
def plot_tsne(X_scaled, y):
    print("🔍 Running t-SNE (may take 20–30s)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X2 = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X2[:, 0], y=X2[:, 1], hue=y, palette=["blue", "red"], s=10)
    plt.title("t-SNE Visualization of Transactions")
    plt.tight_layout()
    plt.savefig("tsne_plot.png")
    plt.close()


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    df = load_or_create_data()
    X_scaled, X_normal, y, scaler = preprocess(df)

    autoencoder = build_autoencoder(input_dim=X_scaled.shape[1])
    autoencoder = train_autoencoder(autoencoder, X_normal)

    errors, preds = evaluate_model(autoencoder, X_scaled, y)
    plot_tsne(X_scaled, y)

    autoencoder.save("fraud_autoencoder.keras")
    print("\n💾 Saved model: fraud_autoencoder.keras")
    print("🎉 Done!")


if __name__ == "__main__":
    main()