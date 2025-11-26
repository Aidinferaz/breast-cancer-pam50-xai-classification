import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from interpret.glassbox import ExplainableBoostingClassifier  # Pastikan library 'interpret' terinstall
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(ide="manual", file_path: str = None):
  if ide == "local":
    df = pd.read_csv("G:\My Drive\ITS\Tugas\Semester_5\Biomedical Engineering\Final Project Req\Dataset.csv")
  elif ide == "local-linux":
        df = pd.read_csv("/home/ferazzio/Dataset/Dataset.csv")
  elif ide == "colab":
    from google.colab import drive
    drive.mount('/content/drive')
    if file_path is None:
        file_path = "/content/drive/MyDrive/ITS/Tugas/Semester_5/Biomedical Engineering/Final Project Req/Dataset.csv"
    df = pd.read_csv(file_path)
  else:
    if file_path is None:
        raise ValueError("file_path must be provided for manual ide.")
    df = pd.read_csv(file_path)

  return df

def preprocess_data(X_train, X_val, X_test=None):
    """
    Melakukan Standard Scaling pada data fitur.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_lasso(X_train, y_train, C=1.0, random_state=42):
    """
    Melatih model White-Box (Lasso Logistic Regression).
    """
    print(f"Training Lasso (White-Box) dengan C={C}...")
    model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        multi_class='ovr',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Melatih model Black-Box (Random Forest).
    """
    print(f"Training Random Forest (Black-Box)...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', random_state=42):
    """
    Melatih model SVM (Support Vector Machine).
    probability=True agar kompatibel dengan SHAP.
    """
    print(f"Training SVM ({kernel} kernel)...")
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=True,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


# --- BAGIAN YANG DIPERBAIKI ---
def train_ebm(X_train, y_train, interactions=0, random_state=42):
    """
    Melatih model Glass-Box (Explainable Boosting Machine).
    Parameter 'interactions' ditambahkan untuk mengontrol pencarian interaksi.
    - interactions=0: Cepat (hanya efek main)
    - interactions=10: Mencari 10 interaksi terkuat (lebih lambat)
    """
    print(f"Training EBM (Glass-Box)...")
    print(f"Note: Interaksi diset ke {interactions}")

    ebm = ExplainableBoostingClassifier(
        interactions=interactions,
        random_state=random_state,
        n_jobs=-1
    )
    ebm.fit(X_train, y_train)
    print("Training Selesai.")
    return ebm


def evaluate(model, X, y, class_names=None, model_name="Model"):
    """
    Menampilkan akurasi dan classification report.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\n--- Evaluasi Performa: {model_name} ---")
    print(f"Akurasi: {acc:.4f}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y, y_pred, target_names=class_names))
    return acc

def get_top_features_from_lasso(lasso_model, feature_names, top_n=200):
    """
    Mengambil N fitur terpenting berdasarkan koefisien Lasso.
    Untuk multi-class, kita ambil nilai absolut maksimum di semua kelas.
    """
    # 1. Ambil Koefisien (n_classes, n_features)
    # Kita ambil nilai absolut karena koefisien negatif (-0.5) sama pentingnya dengan positif (0.5)
    abs_coefs = np.abs(lasso_model.coef_)

    # 2. Agregasi: Ambil nilai max per fitur di semua kelas
    # Artinya: "Apakah gen ini penting untuk setidaknya SATU jenis kanker?"
    importances = np.max(abs_coefs, axis=0)

    # 3. Urutkan dari yang terbesar
    # argsort mengembalikan indeks, [::-1] membaliknya jadi descending
    sorted_indices = np.argsort(importances)[::-1]

    # 4. Ambil Top N
    top_indices = sorted_indices[:top_n]
    top_feature_names = [feature_names[i] for i in top_indices]

    print(f"✅ Berhasil menyeleksi {top_n} fitur terbaik dari {len(feature_names)} fitur awal.")
    return top_indices, top_feature_names

class BreastCancerNN(nn.Module):
    """
    Arsitektur Neural Network untuk klasifikasi PAM50.
    """
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], num_classes=5, dropout=0.3):
        super(BreastCancerNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NeuralNetworkWrapper:
    """
    Wrapper untuk Neural Network agar kompatibel dengan API sklearn (predict, predict_proba).
    """
    def __init__(self, model, device='cpu', classes=None):
        self.model = model
        self.device = device
        self.classes_ = classes
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()


def train_neural_network(X_train, y_train, X_val=None, y_val=None,
                         hidden_sizes=[256, 128, 64], dropout=0.3,
                         epochs=100, batch_size=32, learning_rate=0.001,
                         early_stopping_patience=10, random_state=42):
    """
    Melatih model Neural Network untuk klasifikasi.
    
    Parameters:
    -----------
    X_train : array-like, Data training
    y_train : array-like, Label training
    X_val : array-like, Data validasi (opsional, untuk early stopping)
    y_val : array-like, Label validasi (opsional)
    hidden_sizes : list, Ukuran hidden layers [256, 128, 64]
    dropout : float, Dropout rate
    epochs : int, Jumlah epoch maksimum
    batch_size : int, Ukuran batch
    learning_rate : float, Learning rate
    early_stopping_patience : int, Patience untuk early stopping
    random_state : int, Random seed
    
    Returns:
    --------
    NeuralNetworkWrapper : Model yang sudah dilatih dengan API sklearn-like
    """
    print(f"Training Neural Network (Black-Box)...")
    print(f"  Architecture: {hidden_sizes}, Dropout: {dropout}")
    print(f"  Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    
    # Set random seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Prepare data
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    classes = np.unique(y_train)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation data
    if X_val is not None and y_val is not None:
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Initialize model
    model = BreastCancerNN(input_size, hidden_sizes, num_classes, dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        else:
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print("Training Neural Network Selesai.")
    
    # Wrap model untuk kompatibilitas sklearn
    wrapper = NeuralNetworkWrapper(model, device, classes)
    wrapper.model_raw = model  # Simpan model asli untuk akses langsung
    wrapper.device = device
    
    return wrapper