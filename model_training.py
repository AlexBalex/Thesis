import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
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

        # print("Number of components retained:", self.pca.n_components_)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


        # Print class distribution before SMOTE
        # print("Class distribution before SMOTE:", np.bincount(y_train))

        # smote = SMOTE(random_state=51)
        smote = SMOTE()

        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Print class distribution after SMOTE
        # print("Class distribution after SMOTE:", np.bincount(y_train_smote))

        self.model = SVC(kernel='rbf', C=10) 
        self.model.fit(X_train_smote, y_train_smote)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        with open('accuracy.json', 'w') as f:
            json.dump({'accuracy': accuracy}, f)

        print("Accuracy:", accuracy)
        # print(classification_report(y_test, y_pred))

        # self.save_model("processed_eeg_data.pkl")

        return accuracy, precision, recall, fscore

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.scaler, self.pca, self.model), f)

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.scaler, self.pca, self.model = pickle.load(f)

    def predict_emotion(self, data):
        prediction = self.model.predict(data)[0]
        
        return prediction

    def multiple_trainings(self, detector, data, repetitions=100):
        accuracies = []
        precisions = []
        recalls = []
        fscores = []
        
        features = [pd.DataFrame(np.array(feature).T) for feature in data['features']]
        labels = data['labels']
        X = np.array(detector.preprocess_data(features))
        y = np.array(labels)

        for _ in range(repetitions):
            accuracy, precision, recall, fscore = detector.train_model(X, y)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)

        # Calculate the average of each metric
        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_fscore = np.mean(fscores)

        print(f"Average Accuracy: {avg_accuracy:.2f}")
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")
        print(f"Average F1-Score: {avg_fscore:.2f}")

        return avg_accuracy, avg_precision, avg_recall, avg_fscore


if __name__ == "__main__":
    detector = EmotionDetector()
    data = detector.load_data('/home/alex/UVT/Thesis/playground/model_training/processed_eeg_data_opt.pkl')
    features = [pd.DataFrame(np.array(feature).T) for feature in data['features']]
    labels = data['labels']
    X = np.array(detector.preprocess_data(features))
    y = np.array(labels)
    detector.train_model(X, y)
    detector.multiple_trainings(detector, data, 100)