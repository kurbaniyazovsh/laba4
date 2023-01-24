import csv
import math
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pylab


#загрузка данных из сsv-файла
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        for row in lines:
            if not row:
                continue
            dataset.append(row)
    return dataset

#вычисление расстояние по метрике Манхэттена
def dist(point1, point2):
    return math.fabs(int(point1[1]) - int(point2[1])) + math.fabs(int(point1[2]) - int(point2[2]))

#получение отсортированной по возрастанию коллекции расстояний и индексов
def getNeighbors(trainData, testData, k):
    neighborsData = []
    distData = []
    for i in range(len(trainData)):
        distance = dist(testData, trainData[i])
        distData.append((trainData[i], distance))
    distData.sort(key = lambda x: x[1])

    #выбор первых k записей из отсортированной коллекции
    for x in range(k):
        neighborsData.append(distData[x][0])
    return neighborsData

#выбор наиболее часто встречающегося значения (одного) выбранныч ранее меток (расстояний) k
def getLabel(neighborsData):
    labels = {}
    for x in range(len(neighborsData)):
        labels[neighborsData[x][-1]] = labels.get(neighborsData[x][-1], 0) + 1
    labels = sorted(labels.items(), key = lambda x: x[1], reverse = True)
    return labels[0][0]

#функция оценки точности (сравнения исходных значений и предсказанных)
def getAccuracy(testData, predictions):
    correct = 0
    for x in range(len(testData)):
        if int(testData[x][-1]) == int(predictions[x]):
            correct += 1
    return (correct/float(len(testData)))*100

#knn-классификатор с помощью бибилиотеки sklearn
def knn_sklearn(trainData, testData, trainClasses, testClasses, k):

    x_train = trainData
    x_test = testData
    y_train = trainClasses
    y_test = testClasses

    #стандартизация и нормализация данных
    scaler = StandardScaler().fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)
    scaler = Normalizer().fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    # Предсказывание
    predictions = model.predict(x_test)

    print('Классы тестовой выборки')
    print(y_test)
    print('Предсказанные классы для тестовой выборки')
    print(predictions)

    return x_train, x_test, y_train, y_test, predictions

#функция построения графика по двум параметрам
def graph(sweetness, crunch, classes, color):

    color_list = [color[str(i)] for i in classes]

    plt.scatter(sweetness, crunch, c = color_list)

#загрузка обучающей выборки
trainData = load_csv("data.csv")
#загрузка тестовой выборки
testData = load_csv("data1example.csv")

k = 3

#1 задание
print("Значение k = ", k)
#предсказывание класса
predictions = []
for x in range(len(testData)):
    print("Значение: ", testData[x])
    neighbors = getNeighbors(trainData, testData[x], k)
    print("Наилучшие соседи: ", neighbors)
    result = getLabel(neighbors)
    predictions.append(result)
    print("предсказываемый класс " + str(result) + " актуальный класс " + repr(testData[x][-1]))
accuracy = getAccuracy(testData, predictions)
print("Точность " + str(accuracy) + "%")

#визуализация данных
colors = {'0': 'red', '1': 'green', '2': 'blue'}

#подготовка данных для визуализации
classes = []
for i in range(len(testData)):
    classes.append(testData[i][-1])

sweetness = []
for i in range(len(testData)):
    sweetness.append(int(testData[i][1]))

crunch = []
for i in range(len(testData)):
    crunch.append(int(testData[i][2]))

pylab.subplot(1,2,1)
graph(sweetness, crunch, classes, colors)
pylab.subplot(1,2,2)
graph(sweetness, crunch, predictions, colors)
plt.show()

#подготовка обучающих данных и целевого столбца
classes_train = []
for i in range(len(trainData)):
    classes_train.append(trainData[i][-1])
    del trainData[i][-1]
    del trainData[i][0]

#подготовка тестовых данных и целевого столбца
classes_test = []
for i in range(len(testData)):
    classes_test.append(testData[i][-1])
    del testData[i][-1]
    del testData[i][0]

X_train, X_test, y_train, y_test, predictions = knn_sklearn(trainData, testData, classes_train, classes_test, k)
print("Точность - " + str((accuracy_score(y_test, predictions))*100.0) + " %")

#визуализация данных
pylab.subplot(1,2,1)
graph(sweetness, crunch, y_test, colors)
pylab.subplot(1,2,2)
graph(sweetness, crunch, predictions, colors)
plt.show()

#повторение эксперимента с новым классом
#загрузка обучающей выборки
trainData = load_csv("trainallergien.csv")
#загрузка тестовой выборки
testData = load_csv("testallergien.csv")

classes = []
for i in range(len(testData)):
    classes.append(testData[i][-1])

sweetness = []
for i in range(len(testData)):
    sweetness.append(int(testData[i][1]))

allergic = []
for i in range(len(testData)):
    allergic.append(int(testData[i][2]))

#1
k = 1

print("Значение k = ", k)
#предсказывание класса
predictions = []
for x in range(len(testData)):
    print("Значение выборки: ", testData[x])
    neighbors = getNeighbors(trainData, testData[x], k)
    print("Наилучшие соседи: ", neighbors)
    result = getLabel(neighbors)
    predictions.append(result)
    print("предсказываемый класс " + str(result) + " актуальный класс " + repr(testData[x][-1]))
accuracy = getAccuracy(testData, predictions)
print("Точность " + str(accuracy) + "%")

colors = {'0': 'red', '1': 'blue', '2': 'green', '3': 'yellow'}

pylab.subplot(1,2,1)
graph(sweetness, allergic, classes, colors)
pylab.subplot(1,2,2)
graph(sweetness, allergic, predictions, colors)
plt.show()

#2
classes_train = []
for i in range(len(trainData)):
    classes_train.append(trainData[i][-1])
    del trainData[i][-1]
    del trainData[i][0]

classes_testing = []
for i in range(len(testData)):
    classes_testing.append(testData[i][-1])
    del testData[i][-1]
    del testData[i][0]

X_train, X_testing, y_train, y_testing, predictions = knn_sklearn(trainData, testData, classes_train, classes_testing, k)
print("Точность - " + str((accuracy_score(y_testing, predictions))*100.0) + " %")

pylab.subplot(1,2,1)
graph(sweetness, allergic, y_testing, colors)
pylab.subplot(1,2,2)
graph(sweetness, allergic, predictions, colors)
plt.show()
