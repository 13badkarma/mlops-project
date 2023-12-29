import numpy as np
import pandas as pd
import hydra
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import subprocess
import skops.io as sio


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import hydra
from omegaconf import DictConfig

class TrainModel:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data = None
        self.model = None

    def load_data(self):
        # Загрузка данных из csv файла с помощью DVC и Google Drive
        self.data = pd.read_csv(self.cfg.data_path)

    def preprocess_data(self):
        # Очистка данных и масштабирование через StandardScaler
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def fit_data(self):
        # Обучение модели логистической регрессии
        self.model = LogisticRegression()
        self.model.fit(self.data)

    def save_model(self):
        # Сохранение модели в DVC
        pass  # Здесь должен быть ваш код для сохранения модели

@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig):
    model = TrainModel(cfg)
    model.load_data()
    model.preprocess_data()
    model.fit_data()
    model.save_model()

if __name__ == "__main__":
    main()



dataset = pd.read_csv('../data/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

joblib.dump(classifier, 'model.pkl')

# Добавление модели в DVC
subprocess.run(['dvc', 'add', 'model.pkl'])
subprocess.run(['dvc', 'push', 'model.pkl.dvc'])

# Извлечение модели из DVC
subprocess.run(['dvc', 'pull', 'model.pkl.dvc'])

# Загрузка модели
model = joblib.load('model.pkl')

# Загрузка тестовых данных
test_data = pd.read_csv('data/test.csv')

# Предсказание модели
predictions = model.predict(test_data)

# Сохранение предсказаний
pd.DataFrame(predictions, columns=['prediction']).to_csv('predictions.csv', index=False)

# Добавление предсказаний в DVC
subprocess.run(['dvc', 'add', 'predictions.csv'])
subprocess.run(['dvc', 'push', 'predictions.csv.dvc'])
if __name__ == "__main__":
    main()