import os
import sys

# base source directory
dir_src_base = os.path.dirname(sys.modules['__main__'].__file__)
dir_resource_base = os.path.join(os.path.dirname(sys.modules['__main__'].__file__), 'resources')
# base dataset directory used to store images
dir_data_base = '/home/michal/Documents/CV_face_dataset'
dir_data_type = 'bmp_'
dir_data_dirty_mark = '_dirty'

# set to use unfiltered dataset
dirty_dataset = False

# audio file names
audio_finished = 'classification_finished.wav'

# base classes available for classification
race_class_labels = [
    'asian',
    'negroid',
    'caucasian',
    'hispanic'
]

# select desired method of classification
use_knn = False
use_fnn = True
use_kmeans = False
use_tree = False

# displays one image per race in loaded dataset
test_load = False
# displays incorrectly classified samples for each method
display_incorrect = True

# size of a train and test dataset per race for supervised learning
# should be divisible by fnn_batch_size to maximize amount of training samples for fnn
train_size = 100
test_size = 50

# K means train size (no test dataset, just clusters)
unsupervised_train_size = 150

# knn range of nearest neighbors
knn_neighbors = 5

# feed-forward NN parameters
fnn_learning_rate = 0.01
fnn_hidden_size = 32
fnn_batch_size = 50
fnn_stagnation_definition = 0.0
fnn_stagnation_tolerance = 8

# color of the background in BGR for masking purposes, experimentally discovered
background = [255, 255, 255]
# settings for ignoring/accepting white background in KNN histograms
filter_background = False

# setting for ignoring/accepting hair color in KNN histograms
filter_hair = False
# what is acceptable for hair color compared to base found pixel in terms of BGR values
hair_tolerance = 35

# select features to use for knn
features_histogram = True
# bin amount - number of groups for color values in histograms, 250 instead of 256 ignores background
hist_bin_amount = 250
# number of bins for histogram feature
hist_bin_per_color = 4

features_stdev = False
features_eyes_distance = False
features_eyes_size = False

# what is acceptable horizontal difference between eye positions
eye_horiz_diff_tolerance = 15

eye_dist_default = 57.5
eye_size_default = 35

eye_size_max = 50
eye_size_min = 20

eye_dist_max = 85
eye_dist_min = 30

# Values of max and min for normalization of input values
max_color = 150.0
min_color = 0.0

max_dev = 100.0
min_dev = 20.0
