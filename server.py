from flask import Flask, request, jsonify
from flask_cors import CORS 
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

features = ['gasLimit', 'gasPrice', 'gasUsed', 'nonce', 'receiver', 'sender', 
            'value', 'fee', 'timestamp']

input_dim = len(features)
autoencoder = Autoencoder(input_dim)
autoencoder.load_state_dict(torch.load("autoencoder_multiversx.pth"))
autoencoder.eval()

gnn_model = GNN(5, hidden_dim=16)
gnn_model.load_state_dict(torch.load("gnn_multiversx.pth"))
gnn_model.eval()

def preprocess_transactions(transactions):
    df = pd.DataFrame(transactions)
    print("df: ", df.columns.tolist())
    print(type(df))
    label_encoders = {}
    for col in ['sender', 'receiver']:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce")
    # Normalize data
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Convert to PyTorch tensor
    test_tensor = torch.tensor(df[features].values, dtype=torch.float32)
    return df, test_tensor

# 🔹 Create transaction graph for GNN
def create_graph(df):
    G = nx.DiGraph()
    encoder = LabelEncoder()
    df['sender'] = encoder.fit_transform(df['sender'])
    df['receiver'] = encoder.fit_transform(df['receiver'])
    for _, row in df.iterrows():
        G.add_edge(row['sender'], row['receiver'], weight=row['value'])
    return from_networkx(G)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        #print("--", data)
        if "transactions" not in data:
            return jsonify({"error": "Missing 'transactions' key in request data"}), 400

        transactions = data.get("transactions", [])
        #print("\ntransakcije: ", transactions)
        df, test_tensor = preprocess_transactions(transactions)
        graph_data = create_graph(df)
        node_features = torch.tensor(df[['gasLimit', 'gasPrice', 'gasUsed', 'value', 'fee']].values, dtype=torch.float32)
        graph_data.x = node_features
        print("gnn created")
        with torch.no_grad():
            reconstructed = autoencoder(test_tensor)
            reconstruction_loss = torch.mean((test_tensor - reconstructed) ** 2, dim=1)

        # Autoencoder threshold (95th percentile)
        threshold_autoencoder = np.percentile(reconstruction_loss.numpy(), 95)
        df['autoencoder_anomaly'] = reconstruction_loss.numpy() > threshold_autoencoder
        print("autoencoder prediction: ", reconstruction_loss.numpy())
        # Run GNN inference
        with torch.no_grad():
            fraud_scores = gnn_model(graph_data).squeeze().numpy()

        # GNN threshold
        threshold_gnn = 0.5  # Adjust based on training
        df['gnn_anomaly'] = fraud_scores > threshold_gnn
        print("gnn prediction: ", fraud_scores)

        df['isFraud'] = df['autoencoder_anomaly'] & df['gnn_anomaly']
        print(df['isFraud'])

        df_records = df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries

        results = [{"txHash": tx["txHash"], "isFraud": pred["isFraud"]} for tx, pred in zip(transactions, df_records)]

        print("results:" ,results)
        return jsonify({"predictions": results})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
