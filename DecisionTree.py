from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import subprocess
import Properties as prop
from Utilities import subset_data_train_test, visualize_conf_matrix, conf_matrix_mapping


def classify_tree(vectors_races, info=False):

    train_vectors, test_vectors = subset_data_train_test(vectors_races)

    df_train, train_race_to_label, train_label_to_race = tree_data(train_vectors)

    trained_decision_tree = train_tree(dataframe=df_train)

    control_labels = []
    control_predictions = []

    for race in test_vectors.keys():

        actual_race_vectors = test_vectors[race]
        actual_label = train_race_to_label[race]

        # predict samples of race and get hit rate
        for sample in actual_race_vectors:
            sample_predict = tree_predict(trained_decision_tree, sample)

            control_predictions.append(sample_predict)
            control_labels.append(actual_label)

    cm_predictions, cm_labels = conf_matrix_mapping(train_label_to_race, control_labels, control_predictions)

    visualize_conf_matrix(cm_labels, cm_predictions, prop.race_class_labels)


# Predict vector race
# Return prediction label
def tree_predict(tree: DecisionTreeClassifier, sample: []) -> int:

    predict_label = tree.predict(sample)

    return predict_label[0]


# Fit decision tree to the dataframe on the input
# Return trained decision tree
def train_tree(dataframe: pd.DataFrame) -> DecisionTreeClassifier:

    # take all columns except for the last one
    features = list(dataframe.columns[:len(dataframe.columns) - 1])
    labels = dataframe['target']
    data = dataframe[features]

    decision_tree = DecisionTreeClassifier(min_samples_split=20, random_state=99)
    decision_tree.fit(data, labels)

    return decision_tree


# Create a pandas dataframe of classification samples
# Return dataframe, mapping from race to label and vice versa as dictionaries
def tree_data(vectors_races: dict) -> pd.DataFrame:

    dataframe_labels = dataframe_feature_labels()
    race_to_label_mapping = dict()
    label_to_race_mapping = dict()

    # create mapping for dataframe
    race_label = 0
    for race_name in vectors_races.keys():
        race_to_label_mapping[race_name] = race_label
        label_to_race_mapping[race_label] = race_name
        race_label += 1

    dataframe_samples = []

    # create dictionaries of samples with 'target' column attached
    for race in vectors_races.keys():

        race_samples = vectors_races[race]
        race_label = race_to_label_mapping[race]

        for sample in race_samples:
            dataframe_samples.append(
                create_dataframe_sample_dict(sample, dataframe_labels, race_label)
            )

    return pd.DataFrame(dataframe_samples), race_to_label_mapping, label_to_race_mapping


# Create a single dictionary from feature vector with labels
# Add 'target' as a classification value
def create_dataframe_sample_dict(vector: [], df_labels: [], race_label) -> dict:

    number_of_features = len(df_labels)
    sample_dict = dict()

    for index in range(number_of_features):

        key_label = df_labels[index]
        value_feature = vector[index]

        sample_dict[key_label] = value_feature

    sample_dict['target'] = race_label

    return sample_dict


# Create labels for features
# Used for labeling in pandas dataframe
def dataframe_feature_labels() -> []:

    df_labels = []

    # order of features should be the same as in Utilities.count_features
    if prop.features_histogram:

        # order of color channels should be the same in Utilities.encode_histogram
        for index in range(prop.hist_bin_per_color):
            label = "hist_b_" + str(index + 1) + "_" + str(prop.hist_bin_per_color)
            df_labels.append(label)

        for index in range(prop.hist_bin_per_color):
            label = "hist_g_" + str(index + 1) + "_" + str(prop.hist_bin_per_color)
            df_labels.append(label)

        for index in range(prop.hist_bin_per_color):
            label = "hist_r_" + str(index + 1) + "_" + str(prop.hist_bin_per_color)
            df_labels.append(label)

    if prop.features_stdev:
        # in order corresponding to Utilities.color_deviations
        df_labels += ['stdev_b', 'stdev_g', 'stdev_r']

    if prop.features_eyes_distance:
        df_labels.append('eye_dist')

    if prop.features_eyes_size:
        df_labels.append('eye_size')

    return df_labels


# Taken from:
# http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")