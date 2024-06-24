import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
import json

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
        return data

    def preprocess_data(self, features):
        features = [pd.DataFrame(feature).fillna(0) for feature in features]
        max_cols = max(df.shape[1] for df in features)
        padded_features = [df.join(pd.DataFrame(0, index=df.index, columns=range(df.shape[1], max_cols))) for df in features]
        flattened_features = [df.values.flatten() for df in padded_features]
        normalized_features = self.scaler.fit_transform(flattened_features)
        reduced_data = self.pca.fit_transform(normalized_features)
        return reduced_data
    
    def preprocess_data_for_prediction(self, feature, max_cols=102):
        feature = pd.DataFrame(feature).fillna(0)
        padded_feature = feature.join(pd.DataFrame(0, index=feature.index, columns=range(feature.shape[1], max_cols)))
        flattened_feature = padded_feature.values.flatten()
        normalized_feature = self.scaler.transform([flattened_feature])
        reduced_data = self.pca.transform(normalized_feature)
        return reduced_data
    
    def train_model(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        smote = SMOTE()

        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        self.model = SVC(kernel='rbf', C=10) 
        self.model.fit(X_train_smote, y_train_smote)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        with open('accuracy.json', 'w') as f:
            json.dump({'accuracy': accuracy}, f)

        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))

        self.save_model("processed_eeg_data.pkl")


    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.scaler, self.pca, self.model), f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.scaler, self.pca, self.model = pickle.load(f)

    def predict_emotion(self, data):
        prediction = self.model.predict(data)[0]
        
        return prediction

if __name__ == "__main__":
    detector = EmotionDetector()
    data = detector.load_data('/home/alex/UVT/Thesis/playground/model_training/processed_eeg_data_opt.pkl')
    features = [pd.DataFrame(np.array(feature).T) for feature in data['features']]
    labels = data['labels']
    X = np.array(detector.preprocess_data(features))
    y = np.array(labels)
    detector.train_model(X, y)