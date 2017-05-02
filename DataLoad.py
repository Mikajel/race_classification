import os
import Properties as prop
import cv2 as opencv
import random as rng


def load_image_dataset():
    imgs_races = {}
    dir_races = {}

    # create a dictionary tracking folders of races
    # example: {key: 'asian', value: '/Documents/images_dataset/bmp_asian'}
    for race in prop.race_class_labels:
        if prop.dirty_dataset:
            dir = os.path.join(prop.dir_data_base, prop.dir_data_type + race + prop.dir_data_dirty_mark)
        else:
            dir = os.path.join(prop.dir_data_base, prop.dir_data_type + race)

        dir_races[race] = dir

    # use dir_races dictionary to create a dictionary of images per race
    # value represents race label and value represents array of images
    # example: {key: 'asian', value: [opencv_img_object_1, opencv_img_object_2, ... ]}
    for race in prop.race_class_labels:

        dir_actual = dir_races[race]
        images_actual = []

        # create opencv image object for every file in race directory
        for file in os.listdir(dir_actual):
            full_path = os.path.join(dir_actual, file)

            if os.path.isfile(full_path):
                images_actual.append(opencv.imread(full_path, 1))

        rng.shuffle(images_actual)
        imgs_races[race] = images_actual

    print('Loaded samples of photos:\n')

    for race in imgs_races.keys():
        print('\tRace: %*s \t Amount: %*s' % (10, race, 5, str(len(imgs_races[race]))))

        if prop.test_load:
            img = imgs_races[race]
            opencv.imshow(race, img[0])
            opencv.waitKey(0)
            opencv.destroyAllWindows()

    return imgs_races
