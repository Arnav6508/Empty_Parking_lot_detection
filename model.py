import os
import pickle
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_path = './data_img'

categories = ['empty', 'not_empty']

data = []
labels = []

for cat_idx, cat in enumerate(categories):
    for file in os.listdir(os.path.join(data_path, cat)):
        img_path = os.path.join(data_path, cat, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(cat_idx)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, stratify = labels, shuffle = True)

classifier = SVC()
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

model = grid_search.best_estimator_

y_prediction = model.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print(f'{score*100}% accuracy on test set')

pickle.dump(model, open('./model.p', 'wb'))