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

