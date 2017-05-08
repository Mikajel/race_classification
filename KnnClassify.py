import numpy as np
import cv2 as opencv
import Utilities as util
import statistics as stat
import Properties as prop


def classify_knn(vectors_races: {}, info=False):

    race_to_label, label_to_race = create_race_label_mapping()

    print('-------- Classification using K-nearest --------')
    print('\t K: %d \n\t' % prop.knn_neighbors)

    # subset train and test vectors
    train_vectors_races, test_vectors_races = util.subset_data_train_test(vectors_races)

    if info:
        print('Loaded KNN training samples:\n')
        for race in train_vectors_races.keys():
            print('\tRace: %*s \t Amount: %*s' % (10, race, 5, str(len(train_vectors_races[race]))))

    # train knn classifier
    trained_model = train_knn_classifier(train_vectors_races, race_to_label, info=info)

    test_data = []
    test_labels = []

    # create test data
    for race in vectors_races.keys():

        # add next race images into test data, save race name into races_order
        test_data += test_vectors_races[race]
        test_labels += len(test_vectors_races[race])*[race_to_label[race]]

    test_predictions = []
    # list of tuples (sample vector, label, prediction)
    failed_samples = []

    hit = 0
    test_amount = len(test_data)

    for index in range(test_amount):

        sample = test_data[index]

        prediction, neighbours = classify_vector(
            trained_model,
            sample,
            k=prop.knn_neighbors
        )

        prediction = int(prediction[0])

        test_predictions.append(prediction)

        if prediction == test_labels[index]:
            hit += 1

    cm_predictions, cm_labels = util.conf_matrix_mapping(label_to_race, test_labels, test_predictions)
    util.visualize_conf_matrix(cm_labels, cm_predictions, prop.race_class_labels)

    # collect all incorrectly classified samples for display and analytics
    for index in range(test_amount):

        if cm_predictions[index] != cm_labels[index]:
            failed_sample = (test_data[index], cm_predictions[index], cm_labels[index])
            failed_samples.append(failed_sample)

    print('Knn hit rate: %.2f %%' % (100*hit/test_amount), end='\n\n\n')

    return failed_samples


# Train KNN model with training data - images
def train_knn_classifier(
        train_data_vectors: [],
        race_to_label_mapping: dict,
        info=False) -> opencv.ml.KNearest_create():

    train_data = []
    responses = []

    knn = opencv.ml.KNearest_create()

    for race in train_data_vectors.keys():
        for sample in train_data_vectors[race]:

            train_data.append(np.asarray(sample).astype(np.float32))
            responses.append(race_to_label_mapping[race])

    train_data = np.asarray(train_data)
    responses = np.asarray(responses)

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


# Create mapping between races and their integer labels for classification
def create_race_label_mapping():

    race_to_label_mapping = dict()
    label_to_race_mapping = dict()
    index = 0

    for race in prop.race_class_labels:
        race_to_label_mapping[race] = index
        label_to_race_mapping[index] = race
        index += 1

    return race_to_label_mapping, label_to_race_mapping
