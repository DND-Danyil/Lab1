import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

def shift_method(data):
    model = KMeans(n_clusters=1, n_init=10)
    model.fit(data)
    wcss = model.inertia_
    scores = []
    for k in range(2, 15):
        model = KMeans(n_clusters=k, n_init=10)
        model.fit(data)
        wcss_new = model.inertia_
        if (wcss_new / wcss) < 0.5:
            scores.append(model.score(data))
            break
        wcss = wcss_new
        scores.append(model.score(data))
    optimal_k = len(scores) + 1
    kmeans = KMeans(n_clusters=optimal_k, n_init=10)
    kmeans.fit(data)
    return optimal_k, scores, kmeans.labels_, kmeans.cluster_centers_

data = np.array([[0, 0], [0, 2], [0, 4], [0, 6], [0, 8], [0, 10], [2, 0], [2, 2], [2, 4], [2, 6], [2, 8], [2, 10], [4, 0], [4, 2], [4, 4], [4, 6], [4, 8], [4, 10], [6, 0], [6, 2], [6, 4], [6, 6], [6, 8], [6, 10], [8, 0], [8, 2], [8, 4], [8, 6], [8, 8], [8, 10], [10, 0], [10, 2], [10, 4], [10, 6], [10, 8], [10, 10]])
clusters, scores, marks, centers = shift_method(data)

plt.subplot(2, 2, 1)
plt.scatter(data[:, 0], data[:, 1])
plt.title("Вихідні точки на площині")

plt.subplot(2, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=marks)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='r', s=200, linewidths=3)
plt.title("Центри кластерів")

plt.subplot(2, 2, 3)
plt.bar(range(2, clusters+1), scores)
plt.title("Оцінка для різної кількості кластерів")
plt.xlabel("Кількість кластерів")
plt.ylabel("Оцінка")

plt.subplot(2, 2, 4)
cmap = ListedColormap(['r', 'g', 'b', 'y', 'c', 'm'])
plt.scatter(data[:, 0], data[:, 1], c=marks, cmap=cmap)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='k', s=200, linewidths=3)
plt.title("Кластерні дані")

plt.tight_layout()
plt.show()
