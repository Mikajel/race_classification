from sklearn.cluster import KMeans
import numpy as np
import Utilities as util
import Properties as prop
import statistics as stat


def classify_kmeans(images_races, info=False):

    print('-------- Classification using K-means ----------')

    train_data, label_order = kmeans_create_data(images_races)
    kmeans_fit_clusters(data=train_data, label_order=label_order, info=info)


# Fits data into space and finds clusters
# Default number of clusters is given by number of races we search for
def kmeans_fit_clusters(data, label_order, n_cluster=len(prop.race_class_labels), info=False):

    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(data)

    if info:
        clusters_accuracy = []

        for index in range(0, n_cluster):

            current_subset = kmeans.labels_[index*prop.unsupervised_train_size:(index+1)*prop.unsupervised_train_size]
            print(label_order[index])

            # print occurrence of every element in cluster and percent of highest occurrence element in cluster
            for index2 in range(0, n_cluster):
                print('%d: %d' % (index2, current_subset.tolist().count(index2)))

            major_occurrence = max(set(current_subset), key=current_subset.tolist().count)
            major_occurence_count = current_subset.tolist().count(major_occurrence)
            print('Major cluster: %d - %d occurrences - %.2f%%' % (
                major_occurrence,
                major_occurence_count,
                100*major_occurence_count/prop.unsupervised_train_size
                ), end='\n\n'
            )

            clusters_accuracy.append(100*major_occurence_count/prop.unsupervised_train_size)

        print('Average result hit rate: %.2f' % stat.mean(clusters_accuracy))
        print('Standard deviation of results: %.2f' % stat.stdev(clusters_accuracy))


# Returns an array of vectors for train and order of labels in which arrays contain elements
# No test dataset because why, we are clustering FTW, I don't even know what ends up where
def kmeans_create_data(vectors_races):

    label_order = []
    train_vectors = []

    for key in vectors_races.keys():

        label_order.append(key)
        train_vectors += vectors_races[key]

    return np.array(train_vectors), label_order

