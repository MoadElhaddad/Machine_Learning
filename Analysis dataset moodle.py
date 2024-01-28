 
from scipy.io import arff
import pandas as pd
from sklearn import cluster
import time
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_data(path, filenames):
    datasets = []
    for filename in filenames:
        df = pd.read_csv(path + filename + '.txt', sep="\s+", header=None)  
        datasets.append((df.values, filename))  
    return datasets

def compute_scores(data, k):
    model = cluster.KMeans(n_clusters=k, random_state=0).fit(data)
    labels = model.labels_
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    return silhouette, davies_bouldin, calinski_harabasz, labels

def plot_data(data, labels, dataset_name, k, method):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title(f'{method} Clustering Results for \'{dataset_name}\' (k={k})')
    plt.show()

def plot_scores(silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores, dataset_name):
    k = len(silhouette_scores) + 2
    fig, axs = plt.subplots(3, 1, figsize=(3, 6)) 

    # Plot Silhouette scores
    axs[0].plot(range(2, k), silhouette_scores, label='Silhouette')
    axs[0].set_title(f'Silhouette scores vs number of clusters ({dataset_name})')
    axs[0].legend()

    # Plot Davies-Bouldin scores
    axs[1].plot(range(2, k), davies_bouldin_scores, label='Davies-Bouldin')
    axs[1].set_title(f'Davies-Bouldin scores vs number of clusters ({dataset_name})')
    axs[1].legend()

    # Plot Calinski-Harabasz scores
    axs[2].plot(range(2, k), calinski_harabasz_scores, label='Calinski-Harabasz')
    axs[2].set_title(f'Calinski-Harabasz scores vs number of clusters ({dataset_name})')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
    
    
def display_dataset(data,shouldBlock = True):
    f0 = data[:,0] 
    f1 = data[:,1] 
    
    plt.scatter(f0,f1, s=8)
    plt.title("Your selected dataset")
    plt.show(block = shouldBlock) 
    
def display_dendrogramme(data,option,shouldBlock = True):

    print(f" Dendrogramme {option} donnees initiales")
    linked_mat = shc.linkage(data ,option)

    plt.figure(figsize = (12,12))
    shc.dendrogram(linked_mat ,
    orientation = 'top' ,
    distance_sort = 'descending' ,
    show_leaf_counts = False)
    plt.title(f"Your selected dataset with {option} linkage dendogramme")
    plt.ylabel("Seuil de distance")
    plt.show(block = shouldBlock)
    
def display_clusters(data,distance,k,thresh,shouldBlock = True):

    model = cluster.AgglomerativeClustering(distance_threshold = thresh,n_clusters=k,linkage = distance)
    model = model.fit(data)

    labels = model.labels_
    kres = model.n_clusters_
    leaves = model.n_leaves_
    
    f0 = data[:,0] 
    f1 = data[:,1] 
    plt.figure(figsize = (12,12))
    
    plt.scatter(f0,f1, c= labels, s=8)
    plt.title(f"Resultat du clustering {distance}")
    plt.show(block = shouldBlock)
def compute_scores_agglomerative(data,thresh,distance,k):
    model = cluster.AgglomerativeClustering(distance_threshold = thresh,n_clusters=k,linkage = distance)
    model.fit(data)

    labels = model.labels_
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)

    return silhouette, davies_bouldin, calinski_harabasz, labels


def main():
    path = '/home/moad/Downloads/dataset-rapport/'  # replace with your actual dataset path
    filenames = ['x1', 'x2', 'x3', 'x4', 'y1', 'zz1', 'zz2']
    datasets = load_data(path, filenames)

    for i, (data, dataset_name) in enumerate(datasets):
        print(f"\nProcessing dataset '{dataset_name}'")
        display_dataset(data)

        if dataset_name in ['x1', 'x2', 'zz1', 'zz2']:  
            print("\nApplying Agglomerative Clustering...")
            silhouette_scores = []
            davies_bouldin_scores = []  
            calinski_harabasz_scores = []  
            labels = None
            best_k = None
            best_silhouette = -1
            k=2
            while not(len(silhouette_scores) > 2 and silhouette_scores[-1] < silhouette_scores[-2] and silhouette_scores[-2] < silhouette_scores[-3]) and (k < 31):  
                silhouette, davies_bouldin, calinski_harabasz, labels_tmp = compute_scores_agglomerative(data, None, 'ward', k)
                silhouette_scores.append(silhouette)
                davies_bouldin_scores.append(davies_bouldin)  
                calinski_harabasz_scores.append(calinski_harabasz)  
                if silhouette > best_silhouette:
                    best_k = k
                    best_silhouette = silhouette
                k = k+1
            labels = compute_scores_agglomerative(data, None, 'ward', best_k)[3]
            display_dendrogramme(data, 'ward')  # display dendrogram after clustering
            plot_data(data, labels, dataset_name, best_k, 'Agglomerative')  
            plot_scores(silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores, dataset_name)

        if dataset_name not in ['x2']:
            print("\nApplying K-means Clustering...")
            silhouette_scores = []
            davies_bouldin_scores = []  
            calinski_harabasz_scores = []  
            labels = None
            best_k = None
            best_silhouette = -1
            k = 2
            while not(len(silhouette_scores) > 2 and silhouette_scores[-1] < silhouette_scores[-2] and silhouette_scores[-2] < silhouette_scores[-3]) and (k < 31):  
                silhouette, davies_bouldin, calinski_harabasz, labels_tmp = compute_scores(data, k)
                silhouette_scores.append(silhouette)
                davies_bouldin_scores.append(davies_bouldin)  
                calinski_harabasz_scores.append(calinski_harabasz) 
                if silhouette > best_silhouette:
                    best_k = k
                    best_silhouette = silhouette
                k = k+1
            labels = compute_scores(data, best_k)[3]
            plot_data(data, labels, dataset_name, best_k, 'K-means')  
            plot_scores(silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores, dataset_name)
if __name__ == '__main__':
    main()
