from scipy.io import arff
import pandas as pd
from sklearn import cluster
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



def load_data(path, filenames):
    datasets = []
    for filename in filenames:
        data, _ = arff.loadarff(path + filename + '.arff')
        df = pd.DataFrame(data)
        datasets.append((df.values, df.columns))
    return datasets

def compute_scores(data, k):
    model = cluster.KMeans(n_clusters=k, random_state=0).fit(data)
    labels = model.labels_
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    return silhouette, davies_bouldin, calinski_harabasz, labels

def plot_data(data, labels, dataset_name, k):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.title(f'k-Means Clustering Results for \'{dataset_name}\' (k={k})')
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



       

        

def main():
    path = '/home/moad/clustering-benchmark/src/main/resources/datasets/artificial//' #replace with your path
    dataset_names = ['xclara', 'square4', 'zelnik1', 'xor']
    datasets = load_data(path, dataset_names)

    for i, (data, target) in enumerate(datasets):
        print(f"Dataset {i+1}: {dataset_names[i]}")
        
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        k = 2
        decrease_count = 0
        labels = None

        while decrease_count < 2:
            tps1 = time.time()
            silhouette, davies_bouldin, calinski_harabasz, labels = compute_scores(data, k)
            tps2 = time.time()

            if silhouette_scores and silhouette < silhouette_scores[-1]:
             decrease_count += 1
            silhouette_scores.append(silhouette)
            davies_bouldin_scores.append(davies_bouldin)
            calinski_harabasz_scores.append(calinski_harabasz)

            print("nb clusters =", k, ", runtime =", round((tps2 - tps1) * 1000, 2))
            print("Silhouette score =", silhouette)
            print("Davies-Bouldin score =", davies_bouldin)
            print("Calinski-Harabasz score =", calinski_harabasz)

            k += 1

        num_clusters_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2
        num_clusters_davies_bouldin = davies_bouldin_scores.index(min(davies_bouldin_scores)) + 2
        num_clusters_calinski_harabasz = calinski_harabasz_scores.index(max(calinski_harabasz_scores)) + 2

        
        _, _, _, labels = compute_scores(data, num_clusters_silhouette)        
        plot_data(data, labels, dataset_names[i], num_clusters_silhouette)

        plot_scores(silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores, dataset_names[i])

        print("\nOptimal number of clusters (Silhouette) =", num_clusters_silhouette)
        print("Optimal number of clusters (Davies-Bouldin) =", num_clusters_davies_bouldin)
        print("Optimal number of clusters (Calinski-Harabasz) =", num_clusters_calinski_harabasz)

if __name__ == "__main__":
    main()

   
