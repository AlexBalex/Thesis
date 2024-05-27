import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC

class EmotionDetector:
    def __init__(self, model_path=None):
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=0.99)
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data['features'], data['labels']

    def preprocess_data(self, features):
        max_cols = max(df.shape[1] for df in features)
        padded_features = [df.join(pd.DataFrame(0, index=df.index, columns=range(df.shape[1], max_cols))) for df in features]
        flattened_features = [df.values.flatten() for df in padded_features]
        normalized_features = self.scaler.fit_transform(flattened_features)
        return normalized_features

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)
        
        # Print class distribution before SMOTE
        # print("Class distribution before SMOTE:", np.bincount(y_train))

        smote = SMOTE(random_state=51)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Print class distribution after SMOTE
        # print("Class distribution after SMOTE:", np.bincount(y_train_smote))

        self.model = SVC(kernel='linear', C=1, probability=True)
        self.model.fit(X_train_smote, y_train_smote)

        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        self.save_model("processed_eeg_data_de_LDS.pkl")

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.scaler, self.pca, self.model), f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.scaler, self.pca, self.model = pickle.load(f)

    def predict_emotion(self, data):
        data_normalized = self.scaler.transform([data])
        # print(data_normalized)
        # data_reduced = self.pca.transform(data_normalized)
        prediction = self.model.predict(data_normalized)[0]
        
        return prediction


if __name__ == "__main__":
    detector = EmotionDetector()
    features, labels = detector.load_data('playground/model_training/processed_eeg_data_de_LDS.pkl')
    X = detector.preprocess_data(features)
    detector.train_model(X, labels)
