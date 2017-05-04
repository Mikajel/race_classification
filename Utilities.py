import Properties as prop
import cv2 as opencv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import itertools


# Switch labels and predictions from numbers to races; decrypt races
def conf_matrix_mapping(label_to_race_mapping: dict(), labels: [], predictions: []):

    labels_decrypted = []
    predictions_decrypted = []

    for index in range(len(predictions)):
        labels_decrypted.append(label_to_race_mapping[labels[index]])
        predictions_decrypted.append(label_to_race_mapping[predictions[index]])

    return predictions_decrypted, labels_decrypted


# as taken from scipy manual for confusion matrix usage
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def visualize_conf_matrix(labels, predictions, classes, precision=2):

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels, predictions)
    np.set_printoptions(precision=precision)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def color_deviations(img) -> []:

    averages = []

    # mean of bgr channels, counting 'average pixel' without background color
    b = img[:, :, 0]
    averages.append(np.mean(np.ma.masked_array(b, mask=(b == 255))))

    g = img[:, :, 1]
    averages.append(np.mean(np.ma.masked_array(g, mask=(g == 255))))

    r = img[:, :, 2]
    averages.append(np.mean(np.ma.masked_array(r, mask=(r == 255))))

    img_width, img_height, _ = img.shape

    b_dev, g_dev, r_dev = 0, 0, 0
    counted_pixels = 0

    for row in range(0, img_width):
        for col in range(0, img_height):

            pixel = img[row][col]

            if (pixel != np.asarray(prop.background)).all():

                counted_pixels += 1

                b_dev += abs(pixel[0] - averages[0])
                g_dev += abs(pixel[1] - averages[1])
                r_dev += abs(pixel[2] - averages[2])

    b_dev = b_dev/counted_pixels
    g_dev = g_dev/counted_pixels
    r_dev = r_dev/counted_pixels

    return [b_dev, g_dev, r_dev]


# subset the vectors from all samples into the amount we wish to use for classification training and testing
# size of subset is determined by property file variable 'train_size' and 'test_size'
def subset_data_train_test(vectors_races: {}) -> dict:

    train_vectors_races = dict()
    test_vectors_races = dict()

    for race in vectors_races:
        full_set = vectors_races[race]

        if (prop.train_size + prop.test_size) > len(full_set):
            print('Error: Test/Train overlap - not enough samples for train and test')

        train_vectors_races[race] = full_set[:prop.train_size]
        test_vectors_races[race] = full_set[-1*prop.test_size:]

    return train_vectors_races, test_vectors_races


# Counts features of given images
# Return a dictionary of races containing array of samples(sample = array of encoded features) to feed to classification
def preprocess_training_data(images_dict, info=False):

    dict_races_samples = {}

    # process all races in dictionary
    for race in images_dict:

        print('Counting %s race samples(features)' % race)
        current_race_samples = []

        current_race_images = images_dict[race]
        for img in current_race_images:
            current_race_samples.append(count_features(img=img, info=info))

        dict_races_samples[race] = current_race_samples

    print()
    return dict_races_samples


# Pre-process histogram values
# Count the average of array for every dictionary color
# Return array of encoded values ([3.578, 1.246, 4.489]) for blue, green and red
def encode_histogram(histogram_dict, feature_bins_amount=prop.hist_bin_per_color):

    avg_colors = []

    for color in ['b', 'g', 'r']:

        actual_bins = histogram_dict[color]
        bin_amount = len(actual_bins)
        window_size = bin_amount/feature_bins_amount

        for window in range(0, feature_bins_amount):
            # use float conversion, 'sum' references numpy sum, not built-in, therefore returns numpy float_32
            avg_colors.append(
                float(
                    sum(
                        actual_bins[window*window_size:window*window_size+window_size])/int(bin_amount/feature_bins_amount)
                )
            )

    return avg_colors


# count input features for a single image
# current features:
#   encoded histogram           (BGR - 3* bin size values)
#   deviation from mean pixel   (BGR - 3 values)
#   eye distance                (1 value)
#   eye size                    (1 value)
def count_features(img, info=False) -> []:
    sample = []

    # add colors to feature space(3 new dimensions)
    if prop.features_histogram:
        hist_dict = count_histogram(img)
        encoded_colors = encode_histogram(hist_dict)
        normalized_colors = normalize_colors(encoded_colors)
        sample += normalized_colors

    # add standard color deviations to feature space(another 3 dimensions)
    if prop.features_stdev:
        normalized_stdev = normalize_stdevs(color_deviations(img))
        sample += normalized_stdev

    # add eye distance and size to feature space(another 2 dimensions)
    if prop.features_eyes_distance or prop.features_eyes_size:
        eye_dist, eye_size = get_eyes(img)

        if prop.features_eyes_distance:
            normalized_eye_dist = normalize_eye_distance(eye_dist)
            sample.append(normalized_eye_dist)

        if prop.features_eyes_size:
            normalized_eye_size = normalize_eye_size(eye_size)
            sample.append(normalized_eye_size)

    if info:
        for feature in sample:
            print('%.3f  ' % feature, end='')

        print('', end='\n')

    return sample


# Count histogram values of RGB from image
# Default sampling set to 64, override for different amount
# Returns dictionary of color values, example = {'b': [12, 16, 18, ...], 'r': [5, 78, 12, ...]}
def count_histogram(image, bin_amount=prop.hist_bin_amount, show_hair_masking=False) -> {}:

    channels = opencv.split(image)
    colors = ("b", "g", "r")
    mask = np.zeros(image.shape[:2], np.uint8)
    mask.fill(255)

    # property file variable filter_background determines whether background color is ignored
    if prop.filter_background:

        # basic mask for centering
        # mask[75:175, 100:150] = 255

        # create a mask to ignore white background in the histogram
        for row in range(0, len(image)):
            for col in range(0, len(image[0])):

                if (image[col][row] == np.asarray(prop.background)).all():
                    try:
                        mask[col][row] = 0
                    except IndexError:
                        print('Error, stepping out of picture')

    # property file variable filter_hair determines whether hair color is ignored
    if prop.filter_hair:

        hair_pixel = get_hair_pixel(img=image)

        # define color limits for what we consider 'hair color' based on hair pixel
        low_limit = np.asarray([
                    hair_pixel[0] - prop.hair_tolerance,
                    hair_pixel[1] - prop.hair_tolerance,
                    hair_pixel[2] - prop.hair_tolerance])

        high_limit = np.asarray([
                    hair_pixel[0] + prop.hair_tolerance,
                    hair_pixel[1] + prop.hair_tolerance,
                    hair_pixel[2] + prop.hair_tolerance])

        for row in range(0, len(image)):
            for col in range(0, len(image[0])):

                # check if pixel is within 'hair threshold'
                if in_color_range(source=image[col][row], low=low_limit, high=high_limit):
                    try:
                        mask[col][row] = 0
                    except IndexError:
                        print('Error, stepping out of picture')

    # set to see hair mask over original image for every picture
    if show_hair_masking:
        opencv.imshow('Original image', image)
        opencv.waitKey(0)
        opencv.destroyAllWindows()

        opencv.imshow('Applied mask', mask)
        opencv.waitKey(0)
        opencv.destroyAllWindows()

        plot_image_hist_rgb(image)

    dict_color_hist = {}

    # loop over the image channels
    for (chan, color) in zip(channels, colors):
        # create a histogram for the current channel and concatenate the resulting histograms for each channel
        hist = opencv.calcHist([chan], [0], mask, [bin_amount], [0, 250])
        dict_color_hist[color] = hist

    return dict_color_hist


# as displayed in a histogram tutorial:
# http://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
# plots a histogram of RGB picture with each channel separately
def plot_image_hist_rgb(image, bin_amount=64):

    channels = opencv.split(image)
    colors = ("b", "g", "r")

    plt.figure()
    plt.title("RGB channels histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    features = []

    # loop over the image channels
    for (chan, color) in zip(channels, colors):

        # create a histogram for the current channel and
        # concatenate the resulting histograms for each channel
        hist = opencv.calcHist([chan], [0], None, [bin_amount], [0, 250])
        features.extend(hist)

        # plot the histogram
        plt.plot(hist, color=color)
        plt.xlim([0, bin_amount])

    plt.show()


# Print color composition of a single pixel in BGR
def channels_print(pixel):

    print('B: %d' % pixel[0])
    print('G: %d' % pixel[1])
    print('R: %d' % pixel[2])


# Return hair pixel
# Search by rows from up down and find first non-background pixel
# Return pixel color two lines lower
# Pray for bald people
def get_hair_pixel(img):

    height, width, _ = img.shape

    for row in range(0, height):
        for col in range(0, width):

            # search pixels by rows until you hit head (most probably hair)
            if (img[row][col] != np.asarray(prop.background)).all():

                # return pixel 3 lines lower (should be totally hair, like, I am pretty sure)
                try:
                    return img[row+10][col+5]
                except IndexError:
                    print('Error, out of picture proportions')

    print('Error: Finished searching for hair, no non-background pixel found')


# Check if array values are between respective indices of two other arrays, low and high
def in_color_range(source, low, high) -> bool:

    for index in range(0, len(source)):

        if low[index] <= source[index] <= high[index]:
            pass
        else:
            return False

    return True


# Normalize list of histogram colors
def normalize_colors(color_list) -> []:

    normalized = []

    for index in range(0, len(color_list)):
        normalized.append(normalize_value(color_list[index], prop.max_color, prop.min_color))

    return normalized


# Normalize list of color deviations
def normalize_stdevs(stdev_list) -> []:
    normalized = []

    for index in range(0, len(stdev_list)):
        normalized.append(normalize_value(stdev_list[index], prop.max_dev, prop.min_dev))

    return normalized


# Normalize eye distance
def normalize_eye_distance(eye_dist) -> float:

    return normalize_value(eye_dist, prop.eye_dist_max, prop.eye_dist_min)


# Normalize eye size
def normalize_eye_size(eye_size) -> float:

    return normalize_value(eye_size, prop.eye_size_max, prop.eye_size_min)


# Normalize values on input for classification
def normalize_value(value, max_value, min_value) -> float:

    return (value-min_value)/(max_value-min_value)


# Detect eyes in image and return its parameters(distance, average width/height)
# Return -1 for all parameters if eyes could not be located
# contains source code lines from OpenCV tutorial for face and eyes detection
# http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
def get_eyes(img, info=False):

    face_cascade = opencv.CascadeClassifier(os.path.join(prop.dir_resource_base, 'haarcascade_frontalface_default.xml'))
    eye_cascade = opencv.CascadeClassifier(os.path.join(prop.dir_resource_base, 'haarcascade_eye.xml'))

    eye_distance = prop.eye_dist_default
    eye_avg_size = prop.eye_size_default

    # convert to grayscale
    img_grayscale = opencv.cvtColor(img, opencv.COLOR_BGR2GRAY)

    # detect face
    faces = face_cascade.detectMultiScale(img_grayscale, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_gray = img_grayscale[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if info:
            print(eyes)

        # if you detected at least two eyes, which are horizontally aligned, get their distance and average shape
        if len(eyes) > 1:
            if abs(eyes[0][1]-eyes[1][1]) < prop.eye_horiz_diff_tolerance:
                eye_distance = abs(eyes[0][0]-eyes[1][0])
                eye_avg_size = (eyes[0][3] + eyes[1][3]) / 2

        if info:
            for (ex, ey, ew, eh) in eyes:
                opencv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # display detected face and eyes
    if info:
        print('Eye distance: %.1f  size: %.1f' % (eye_distance, eye_avg_size))
        opencv.imshow('Face/Eyes detection', img)
        opencv.waitKey(0)
        opencv.destroyAllWindows()

    # if eyes are bigger than own distance, probably detection error, return defaults
    if eye_avg_size < eye_distance:
        return eye_distance, eye_avg_size
    else:
        return prop.eye_dist_default, prop.eye_size_default


# print program settings
def print_settings():

    print()
    print('Program settings')
    print('\t training set: %d per race \n\t test set: %d per race\n' % (prop.train_size, prop.test_size))
    print('\t Filter background: %r \n\t Filter hair: %r\n' % (prop.filter_background, prop.filter_hair))
    print('\t Histogram feature: %r' % prop.features_histogram)
    if prop.features_histogram:
        print('\t\t  bins per color: %d' % prop.hist_bin_per_color)
    print('\t Pixel stdev feature: %r' % prop.features_stdev)
    print('\t Eye size feature: %r' % prop.features_eyes_size)
    print('\t Eye distance feature: %r' % prop.features_eyes_distance)
    print()
