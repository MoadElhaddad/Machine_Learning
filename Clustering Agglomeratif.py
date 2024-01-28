import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import time
import pandas as pd
import scipy.cluster.hierarchy as shc
from scipy.io import arff
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_dataset(path, filenames):
    datasets = []
    for filename in filenames:
        full_path = os.path.join(path, filename + '.arff')
        data, _ = arff.loadarff(full_path)
        df = pd.DataFrame(data)
        datasets.append((df.values, df.columns))
    return datasets


def display_dataset(dataset, shouldBlock=True):
    data, columns = dataset
    f0 = data[:,0] 
    f1 = data[:,1] 
    
    plt.scatter(f0, f1, s=8)
    plt.title("Your selected dataset")
    if shouldBlock:
        plt.show()

def display_dendrogramme(dataset, option, shouldBlock=True):
    data, _ = dataset  
    linked_mat = shc.linkage(data, option)

    print(f"Dendrogramme {option} donnees initiales")

    plt.figure(figsize=(12, 12))
    shc.dendrogram(linked_mat,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=False)
    plt.title(f"Your selected dataset with {option} linkage dendogramme")
    plt.ylabel("Seuil de distance")

    if shouldBlock:
        plt.show()
    

def display_clusters(data, distance, k, thresh, shouldBlock=True):
    model = cluster.AgglomerativeClustering(distance_threshold=thresh, n_clusters=k, linkage=distance)
    model = model.fit(data)

    labels = model.labels_
    kres = model.n_clusters_
    leaves = model.n_leaves_

    f0 = data[:,0] 
    f1 = data[:,1] 
    plt.figure(figsize=(12,12))
    
    plt.scatter(f0, f1, c=labels, s=8)
    plt.title(f"Result of clustering {distance}")
    plt.text(0.05, 0.95, f"Number of clusters: {kres}", transform=plt.gca().transAxes)
    plt.show(block=shouldBlock)


def compute_scores_agglomerative(data,thresh,distance,k):
    model = cluster.AgglomerativeClustering(distance_threshold = thresh,n_clusters=k,linkage = distance)
    model.fit(data)

    labels = model.labels_
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)

    return silhouette, davies_bouldin, calinski_harabasz, labels

def compute_scores(data, thresh, distance, k):
    model = cluster.AgglomerativeClustering(distance_threshold=thresh, n_clusters=k, linkage=distance).fit(data)
    labels = model.labels_
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    return silhouette, davies_bouldin, calinski_harabasz, labels
def plot_silhouette_scores(silhouette_scores):
    k = len(silhouette_scores) + 2
    plt.figure()
    plt.plot(range(2, k), silhouette_scores, label='Silhouette')
    plt.title(f'Silhouette_scores')
    plt.grid()
    plt.legend()
    plt.show()

def plot_davies_bouldin_scores(davies_bouldin_scores):
    k = len(davies_bouldin_scores) + 2
    plt.figure()
    plt.plot(range(2, k), davies_bouldin_scores, label='Davies-Bouldin')
    plt.title(f'Davies_Bouldin_scores')
    plt.grid()
    plt.legend()
    plt.show()

def plot_calinski_harabasz_scores(calinski_harabasz_scores):
    k = len(calinski_harabasz_scores) + 2
    plt.figure()
    plt.plot(range(2, k), calinski_harabasz_scores, label='Calinski-Harabasz')
    plt.title(f'calinski_harabasz_scores')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    full_path = input("Please provide the full path to your desired dataset using the following format \n /your/path/<yourfile>.arff :")
    directory_path, filename_with_ext = os.path.split(full_path)
    filename, _ = os.path.splitext(filename_with_ext)
    datasets = load_dataset(directory_path, [filename])
    first_dataset = datasets[0]
    display_dataset(first_dataset, True)

    silhouette_tab = [0,0]
    davies_bouldin_tab = [0,0]
    calinski_harabasz_tab = [0,0]
    i = 2
    while not((silhouette_tab[-2] > silhouette_tab[-1]) and (silhouette_tab[-3] > silhouette_tab[-1])) and (i<50):
        data, _ = first_dataset
        [silhouette, davies_bouldin, calinski_harabasz, labels] = compute_scores(data, None, "ward", i)
        silhouette_tab.append(silhouette)
        davies_bouldin_tab.append(davies_bouldin)
        calinski_harabasz_tab.append(calinski_harabasz)
        i = i+1
    data, _ = first_dataset
    display_dendrogramme(first_dataset, "ward", True)
    display_clusters(data, "ward", i-2, None, True)
    plot_silhouette_scores(silhouette_tab[2:])
    plot_davies_bouldin_scores(davies_bouldin_tab[2:])
