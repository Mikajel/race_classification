import tensorflow as tf
import numpy as np
import Properties as prop
from random import shuffle
from Utilities import subset_data_train_test


def classify_fnn(vectors_races, info=False):

    train_data, test_data = subset_data_train_test(vectors_races)
    train_vectors, train_labels, train_mapping = create_vectors_and_labels(train_data)
    test_vectors, test_labels, test_mapping = create_vectors_and_labels(test_data)

    train_fnn(train_vectors, train_labels, test_vectors, test_labels)


def train_fnn(
        train_vectors: [],
        train_labels: [],
        test_vectors: [],
        test_labels: [],
        learning_rate: float = prop.fnn_learning_rate,
        hidden_size: int = prop.fnn_hidden_size):

    batches_of_vectors, batches_of_labels = split_to_batches(train_vectors, train_labels)

    session_graph = tf.Graph()
    with session_graph.as_default():

        # variables.
        batch_size = prop.fnn_batch_size
        num_of_labels_session = len(prop.race_class_labels)

        # input data; for the training data, use a placeholder that will be fed at run time with a training minibatch
        tf_train_vectors = tf.placeholder(tf.float32, shape=(batch_size, len(train_vectors[0])))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, len(train_labels[0])))
        tf_test_vectors = tf.constant(test_vectors)

        # weights and biases
        weights_h1 = tf.Variable(tf.truncated_normal([len(train_vectors[0]), hidden_size]))
        biases_h1 = tf.Variable(tf.zeros([hidden_size]))
        h1 = tf.nn.relu(tf.matmul(tf_train_vectors, weights_h1) + biases_h1)

        weights = tf.Variable(tf.truncated_normal([hidden_size, num_of_labels_session]))
        biases = tf.Variable(tf.zeros([num_of_labels_session]))

        # training computation
        logits = tf.matmul(h1, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(
            tf.matmul(
                tf.nn.relu(
                    tf.matmul(tf_test_vectors, weights_h1) + biases_h1), weights) + biases)

    with tf.Session(graph=session_graph) as session:
        tf.initialize_all_variables().run()

        last_test_accuracy = 0.0
        stagnation = 0
        stagnation_tolerance = prop.fnn_stagnation_tolerance
        stop_training = False
        stagnation_definition = prop.fnn_stagnation_definition

        dataset_walks = 0

        while not stop_training:
            for batch_index in range(len(batches_of_vectors)):

                if not stop_training:
                    # Generate a batch.
                    batch_vectors = batches_of_vectors[batch_index]
                    batch_labels = batches_of_labels[batch_index]

                    # Feed the dictionary and start training
                    feed_dict = {tf_train_vectors: batch_vectors, tf_train_labels: batch_labels}

                    _, l, predictions = session.run(
                        [optimizer, loss, train_prediction],
                        feed_dict=feed_dict
                    )

                    print("Batch accuracy: %.1f%%" % evaluate_accuracy(predictions, batch_labels))

                    test_accuracy = evaluate_accuracy(
                        test_prediction.eval(), test_labels, print_confusion_matrix=False)

                    print("Test accuracy: %.1f%%\n" % test_accuracy)

                    # Count the difference of validation set accuracy
                    diff = test_accuracy - last_test_accuracy
                    if diff < stagnation_definition:
                        stagnation += 1
                        print('Accuracy improvement only %.3f, stagnation increased to %d' % (diff, stagnation))
                        if stagnation > stagnation_tolerance:
                            stop_training = True

                    if diff > stagnation_definition:
                        if stagnation > 0:
                            stagnation -= 1
                            print('Accuracy improvement %.3f, stagnation decreased to %d' % (diff, stagnation))

                    last_test_accuracy = test_accuracy

                # increase the number of dataset walks and decrease stagnation tolerance
                dataset_walks += 1
                # stagnation_tolerance -= 1

        print('Number of dataset walks: %d' % dataset_walks)


# take predictions of neural networks and return hit rate
def evaluate_accuracy(predictions: [], labels: [], print_confusion_matrix: bool =False) -> float:

    hit = 0
    total = len(predictions)

    # evaluate every prediction
    for index in range(len(predictions)):

        # get the index of highest element - prediction
        pred_index_max = max(range(len(predictions[index])), key=predictions[index].__getitem__)
        label_index_max = max(range(len(labels[index])), key=labels[index].__getitem__)

        if label_index_max == pred_index_max:
            hit += 1

    if print_confusion_matrix:
        print('This is a trial version. Purchase a full version to see confusion matrix.')

    return 100*(hit/total)


# Separates training data into batches, returns list of batches for vectors and labels
def split_to_batches(train_vectors, train_labels, batch_size=prop.fnn_batch_size):

    batches_vectors = []
    batches_labels = []

    if len(train_vectors) == len(train_labels):
        sample_amount = len(train_vectors)
    else:
        print('ERROR: Amount of vectors and labels not equal')
        return

    batch_amount = int(sample_amount/batch_size)

    for index in range(0, batch_amount):

        offset_lower = index*batch_size
        offset_upper = (index+1)*batch_size

        batches_vectors.append(train_vectors[offset_lower:offset_upper])
        batches_labels.append(train_labels[offset_lower:offset_upper])

    return batches_vectors, batches_labels


# Create labels for neural network output, return vectors, labels and mapping from races to labels
# example: { 'asian' : [0, 0, 1, 0] }
def create_vectors_and_labels(vectors_races: dict):

    label_mapping = dict()
    counter = 0

    # create mapping from output labels to race labels
    for race in vectors_races.keys():

        label = [0] * len(prop.race_class_labels)
        label[counter] = 1
        counter += 1
        label_mapping[race] = label

    created_vectors = []
    created_labels = []

    # add vectors with corresponding labels to lists
    for race in vectors_races.keys():

        label = label_mapping[race]
        created_vectors += vectors_races[race]
        created_labels += [label] * len(vectors_races[race])

    tuples_to_shuffle = []

    # join them into tuples
    for index in range(0, len(created_vectors)):

        sample_tuple = (created_vectors[index], created_labels[index])
        tuples_to_shuffle.append(sample_tuple)

    # shuffle tuples
    shuffle(tuples_to_shuffle)

    # split them back into lists
    final_vectors = [tup[0] for tup in tuples_to_shuffle]
    final_labels = [tup[1] for tup in tuples_to_shuffle]

    return final_vectors, final_labels, label_mapping


