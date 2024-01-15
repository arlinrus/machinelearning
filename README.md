# machinelearning

# Направления методов машинного обучения: методов главных компонентов 
МГК - это уменьшение размерности с минимальными потерями в информативности 

Рассмотрим пример с таблицей:

<img width="484" alt="Снимок экрана 2024-01-10 в 15 23 54" src="https://github.com/arlinrus/machinelearning/assets/111064731/a45e8110-171c-4395-bd90-28ece3ca447d">

Для начала найдем среднее арифметичнское для каждого из объектов и произведем центрирование для объектов x1 и x2

<img width="769" alt="Снимок экрана 2024-01-10 в 15 29 06" src="https://github.com/arlinrus/machinelearning/assets/111064731/0abc8dbe-f3d2-4a51-8545-57f527df92f9">

Где новые координат векторов будут иметь данные заданные значения: 0,591 и 0,807 или тоже самое с минусами.

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/arlinrus/PycharmProjects/randomNumbers/venv/14_16 (2).csv" , header = None)

# With the help of PCA we can work with our dataset
pca = PCA(n_components= 2 , svd_solver= 'full')
pca.fit(df)
pca_data = pca.transform(df)
print(pca_data[0])

#Try to find explained_variance
pca.components_.shape #матрица весов хранится в поле компнентс экземпляра класса pca и имеет размеры
pca.components_

pca = PCA(n_components= 10, svd_solver = "full")
pca_data = pca.fit_transform(df)
print(pca_data)

explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_),2)
print(explained_variance)

# Make a plot
# plt.plot(np.arange(1 , 11) ,explained_variance , ls = "-")
#
# plt.scatter(pca_data[:, 0]  ,pca_data[:, 1])
# plt.scatter

