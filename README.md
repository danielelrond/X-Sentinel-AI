# MultiversX AI Fraud Detection

## Introduction
Fraud detection in blockchain transactions is a critical challenge as malicious actors exploit vulnerabilities to perform fraudulent activities. Our project leverages AI-powered models to detect fraudulent transactions in real-time on the MultiversX blockchain. This system integrates an autoencoder-based anomaly detection model and a Graph Neural Network (GNN) to classify fraudulent transactions based on transaction attributes and graph-based representations.

This project was developed for the **AI MegaWave Hackathon**, a prestigious event focused on the intersection of **AI & MultiversX**, featuring cutting-edge projects in AI, blockchain, and Web3 technologies.

---

## Problem Statement
The rise of blockchain transactions brings opportunities and risks. Fraudulent transactions, including money laundering, transaction spoofing, and unauthorized access, threaten the integrity of the ecosystem. Traditional rule-based fraud detection systems are ineffective against adaptive attackers, necessitating an AI-driven approach to detect fraudulent behavior.

Our **MultiversX AI Fraud Detection System** addresses these challenges using:
- **Autoencoder for Anomaly Detection**: Identifies unusual transaction patterns.
- **Graph Neural Networks (GNN)**: Detects fraud based on transaction relationships and node interactions in the blockchain.

---

## AI Models Used
### 1. **Autoencoder Model**
The autoencoder is a deep learning model trained to reconstruct normal transaction data. Anomalies are detected when the reconstruction error exceeds a predefined threshold.
- **Input features**: `gasLimit`, `gasPrice`, `gasUsed`, `nonce`, `receiver`, `sender`, `value`, `fee`, `timestamp`
- **Architecture**: Three-layer encoder-decoder structure
- **Loss function**: Mean Squared Error (MSE)

### 2. **Graph Neural Network (GNN)**
A Graph Neural Network (SAGEConv) processes transactions as a graph where each transaction represents a node, and edges denote sender-receiver relationships.
- **Features**: `gasLimit`, `gasPrice`, `gasUsed`, `value`, `fee`
- **Hidden layers**: Two-layer SAGEConv architecture
- **Activation function**: ReLU
- **Output**: Fraud probability score

---

## How It Works
### **1. Data Preprocessing**
- Transactions are received via API.
- Data normalization and encoding for numerical stability.
- Conversion into PyTorch tensor for ML model inference.

### **2. Autoencoder-Based Anomaly Detection**
- Transactions are passed through the autoencoder.
- Reconstruction loss is computed.
- Transactions with a high reconstruction loss are flagged as anomalous.

### **3. Graph Neural Network (GNN) for Fraud Classification**
- Transactions are structured as a directed graph.
- GNN model classifies nodes as fraudulent or legitimate.
- Fraud probability is assigned based on node features.

### **4. Decision Logic**
- A transaction is classified as **fraudulent** if both models detect an anomaly.
- Predictions are returned via API.

---

## Running the Project
### **1. Setting Up the Backend (Flask API)**
#### Install Dependencies:
```bash
pip install flask flask-cors torch torch-geometric pandas numpy scikit-learn networkx
```
#### Run the Flask Server:
```bash
python server.py
```

### **2. Running the Frontend (React Application)**
#### Install Dependencies:
```bash
cd multiversx-live-transactions
npm install
```
#### Start the React App:
```bash
npm start
```

The React app will display **live transactions** with fraud detection results in real-time.

---

## Future Work
- **Integration with MultiversX smart contracts** for on-chain fraud detection.
- **Optimization of GNN model** using additional blockchain-based features.
- **Enhancing fraud detection accuracy** through reinforcement learning.

