import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift

X = np.loadtxt('Data_Clustering.txt', delimiter=',')

bandwidth = MeanShift(bandwidth=2).fit(X)
ms_centers = bandwidth.cluster_centers_

scores = []
values = np.arange(2, 16)
for num_clusters in values:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    score = silhouette_score(X, kmeans.labels_, metric='euclidean')
    print("Кількість кластерів =", num_clusters, "\nОцінка силуета =", score)
    scores.append(score)

num_clusters = np.argmax(scores) + values[0]
print("\nОптимальна кількість кластерів =", num_clusters)

kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.subplot(2, 2, 1)
plt.scatter(X[:,0], X[:,1], color='black', s=50, marker='o')
plt.title('Вихідні точки на площині')

plt.subplot(2, 2, 2)
plt.scatter(X[:,0], X[:,1], color='black', s=50, marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, color='red', linewidths=3, marker='x')
plt.title('Центри кластерів')

plt.subplot(2, 2, 3)
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.xticks(values)
plt.title('Кількість кластерів vs оцінка силуета')
plt.xlabel('Кількість кластерів')
plt.ylabel('Оцінка силуета')

plt.subplot(2, 2, 4)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.5)
plt.scatter(X[:,0], X[:,1], color='black', s=50, marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=150, color='red', linewidths=3, marker='x')
plt.title('Кластерні дані')

plt.show()
