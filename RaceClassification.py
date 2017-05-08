import warnings
import Properties as prop
from KnnClassify import classify_knn
from KmeansClassify import classify_kmeans
from FeedforwardNeural import classify_fnn
from DecisionTree import classify_tree
import Audio as aud
from Utilities import print_settings, display_sample_set, preprocess_training_data
import DataLoad as data

warnings.filterwarnings("ignore")
images_races = data.load_image_dataset()
vectors_races, display_backup = preprocess_training_data(images_dict=images_races)


print_settings()

if prop.use_knn:
    knn_false = classify_knn(vectors_races, info=False)
    if prop.display_incorrect:
        print(len(knn_false))
        display_sample_set(knn_false, display_backup)

if prop.use_kmeans:
    kmeans_false = classify_kmeans(vectors_races, info=True)

if prop.use_tree:
    tree_false = classify_tree(vectors_races, info=True)
    if prop.display_incorrect:
        print(len(tree_false))
        display_sample_set(tree_false, display_backup)

if prop.use_fnn:
    fnn_false = classify_fnn(vectors_races, info=True)
    if prop.display_incorrect:
        print(len(fnn_false))
        display_sample_set(fnn_false, display_backup)

aud.play_sound_wav(prop.audio_finished)

