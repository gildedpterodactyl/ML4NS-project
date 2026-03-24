import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. Define the MLP Architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, output_dim=1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

def main(args):
    # Load and prepare data
    if not os.path.exists(args.latent_csv):
        print(f"Latent dataset not found: {args.latent_csv}")
        return
        
    df = pd.read_csv(args.latent_csv)
    
    X = df.drop(columns=[args.property_column]).values
    y = df[args.property_column].values.reshape(-1, 1)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    input_dim = X_train.shape[1]
    model = MLP(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Starting training for property: {args.property_column}")
    print(f"Input dimension: {input_dim}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{args.epochs}.. Train Loss: {train_loss:.4f}.. Val Loss: {val_loss:.4f}")

    # Save the model and scaler
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"regressor_{args.property_column}.pt")
    scaler_path = os.path.join(args.output_dir, f"scaler_{args.property_column}.pkl")
    
    torch.save(model.state_dict(), model_path)
    pd.to_pickle(scaler, scaler_path)
    
    print(f"\nTraining complete. Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP regressor on latent vectors.")
    parser.add_argument("--latent_csv", type=str, required=True, help="Path to the latent dataset CSV.")
    parser.add_argument("--property_column", type=str, required=True, help="Name of the property column to predict.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the trained model.")
    
    args = parser.parse_args()
    main(args)