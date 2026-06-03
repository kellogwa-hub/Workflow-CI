import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    # Di environment GitHub Actions (CI), kita serahkan pembuatan Run sepenuhnya ke 'mlflow run'
    # Cukup panggil autolog, MLflow otomatis mencatat semuanya ke Run yang sedang aktif
    mlflow.autolog()

    # 2. Muat data bersih (pastikan folder dan file ini ada di dalam folder MLProject)
    df = pd.read_csv("loan_dataset_preprocessing/loan_clean.csv")

    # 3. Pisahkan Fitur (X) dan Target (y)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # 4. Data Splitting (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Scaling Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Inisiasi dan Pelatihan Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 7. Evaluasi Model
    predictions = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)

    print(f"Model selesai dilatih dengan Akurasi: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, predictions))

if __name__ == "__main__":
    train_model()