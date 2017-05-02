import warnings
import Properties as prop
import KnnClassify as knn
from KmeansClassify import classify_kmeans
from KnnClassify import classify_knn
from FeedforwardNeural import classify_fnn
import Audio as aud
import Utilities as util
import DataLoad as data

warnings.filterwarnings("ignore")
images_races = data.load_image_dataset()
vectors_races = util.preprocess_training_data(images_dict=images_races)

util.print_settings()

if prop.use_knn:
    classify_knn(vectors_races, info=False)

if prop.use_kmeans:
    classify_kmeans(vectors_races, info=True)

if prop.use_fnn:
    classify_fnn(vectors_races, info=True)

# aud.play_sound_wav(prop.audio_finished)






