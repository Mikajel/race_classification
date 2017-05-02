import numpy as np
import cv2 as opencv
import Utilities as util
import statistics as stat
import Properties as prop


def classify_knn(vectors_races: {}, info=False):

    test_data = []
    print('-------- Classification using K-nearest --------')
    print('\t K: %d \n\t' % prop.knn_neighbors)

    # subset train and test vectors
    train_vectors_races, test_vectors_races = util.subset_data_train_test(vectors_races)

    if info:
        print('Loaded KNN training samples:\n')
        for race in train_vectors_races.keys():
            print('\tRace: %*s \t Amount: %*s' % (10, race, 5, str(len(train_vectors_races[race]))))

    trained_model = train_knn_classifier(train_vectors_races, info=info)

    races_order = []

    # create test data
    for race in vectors_races.keys():

        races_order.append(race)
        # add next race images into test data, save race name into races_order
        test_data += test_vectors_races[race]

    # testing
    race_results = []

    for race in range(0, len(races_order)):

        print(races_order[race][0].upper() + ': ', end='')

        results_matrix_row = {}

        for index in range(0, prop.test_size):

            results, neighbours = classify_vector(
                trained_model,
                test_data[race*prop.test_size + index],
                k=prop.knn_neighbors
            )

            if str(chr(results)) in results_matrix_row.keys():
                results_matrix_row[str(chr(results))] += 1
            else:
                results_matrix_row[str(chr(results))] = 1

        print(results_matrix_row)
        try:
            print('%.2f %% hit rate' %
                  (100*int(results_matrix_row[races_order[race][0].upper()])/prop.test_size))
            race_results.append(100*int(results_matrix_row[races_order[race][0].upper()])/prop.test_size)
        except KeyError:
            print('%.2f %% hit rate' % 0)
            race_results.append(0)

    print('Average result hit rate: %.2f %%' % (stat.mean(race_results)))
    print('Standard deviation of results: %.2f' % stat.stdev(race_results), end='\n\n\n')


# Train KNN model with training data - images
def train_knn_classifier(train_data_vectors, info=False):

    train_data = []
    responses = []

    knn = opencv.ml.KNearest_create()

    for race in train_data_vectors.keys():
        for sample in train_data_vectors[race]:

            train_data.append(np.asarray(sample).astype(np.float32))

            if race == 'asian':
                responses.append(65)

            if race == 'negroid':
                responses.append(78)

            if race == 'caucasian':
                responses.append(67)

            if race == 'hispanic':
                responses.append(72)

    train_data = np.asarray(train_data)
    responses = np.asarray(responses)

    print(len(train_data), len(responses))

    if info:
        for index in range(0, len(train_data)):
            print(train_data[index], ' == ', responses[index])

    knn.train(train_data, opencv.ml.ROW_SAMPLE, responses)

    return knn


# Classifies an input image inside pre-counted KNN space
# K represents number of neighbours, default is 10
def classify_vector(knn_model, sample, k=10):

    ret, results, neighbours, dist = knn_model.findNearest(np.asarray([np.asarray(sample).astype(np.float32)]), k)

    return results, neighbours
