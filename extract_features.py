#from numpy.random import seed
#seed(1)
#from tensorflow import random
#random.set_seed(1)
import shutil
import tqdm
import numpy as np
import cv2
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model
import config
import csv
import keras


def video_to_frames(video):
    path = os.path.join(config.train_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    count = 0
    image_list = []
    # Path to video file
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        cv2.imwrite(os.path.join(config.train_path, 'temporary_images', 'frame%d.jpg' % count), frame)
        image_list.append(os.path.join(config.train_path, 'temporary_images', 'frame%d.jpg' % count))
        count += 1

    cap.release()
    cv2.destroyAllWindows()
    return image_list


def model_cnn_load():
    #model = VGG16(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    #out = model.layers[-2].output
    #model_final = Model(inputs=model.input, outputs=out)
    return keras.models.load_model(f"model_final/vgg16.keras")


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    return img


def extract_features(video, model):
    """

    :param video: The video whose frames are to be extracted to convert into a numpy array
    :param model: the pretrained vgg16 model
    :return: numpy array of size 4096x80
    """
    print(f'Processing video {video}')
    narrator_id, video_id, _ = video[:-4].split("_")
    video_location = f"../epic_kitchens/videos/training_trimmed/{narrator_id}/{narrator_id}_{video_id}/{video}"
    image_list = video_to_frames(video_location)
    samples = np.round(np.linspace(
        0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.zeros((len(image_list), 224, 224, 3))
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        images[i] = img
    images = np.array(images)
    fc_feats = model.predict(images, batch_size=128)
    img_feats = np.array(fc_feats)
    # cleanup
    shutil.rmtree(os.path.join(config.train_path, 'temporary_images'))
    return img_feats

def extract_features2(model, training_video_name=None, testing_video_name=None):
    """

    :param video: The video whose frames are to be extracted to convert into a numpy array
    :param model: the pretrained vgg16 model
    :return: numpy array of size 4096x80
    """
    if training_video_name != None:
        narrator_id, video_id, _ = training_video_name[:-4].split("_")
        video_location = f"../epic_kitchens/videos/training_trimmed/{narrator_id}/{narrator_id}_{video_id}/{training_video_name}"
        video_name = training_video_name
    elif testing_video_name != None:
        video_location = f"data/testing_data/video/{testing_video_name}"
        video_name = testing_video_name
    print(f'Processing video {video_name}')
    image_list = video_to_frames(video_location)
    samples = np.round(np.linspace(
        0, len(image_list) - 1, 80))
    image_list = [image_list[int(sample)] for sample in samples]
    images = np.zeros((len(image_list), 224, 224, 3))
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        images[i] = img
    images = np.array(images)
    fc_feats = model.predict(images, batch_size=128)
    img_feats = np.array(fc_feats)
    # cleanup
    shutil.rmtree(os.path.join(config.train_path, 'temporary_images'))
    return img_feats



def return_list_of_videos():
    video_list = []
    base_dir = "../epic_kitchens/videos/training_trimmed"
    annotator_list = os.listdir(base_dir)
    for annotator_folder in annotator_list:
        if annotator_list != '.ipynb_checkpoints' and annotator_list != '.gitignore' and  annotator_list != '.DS_Store':
            video_folders = os.listdir(f"{base_dir}/{annotator_folder}")
            for video_folder in video_folders:
                if video_folder != '.ipynb_checkpoints' and video_folder != '.gitignore' and  video_folder != '.DS_Store':
                    segments_list = os.listdir(f"{base_dir}/{annotator_folder}/{video_folder}")
                    for segment in segments_list:
                        if segment != '.ipynb_checkpoints' and segment != '.gitignore' and  segment != '.DS_Store' and segment[0] == "P":
                            video_list.append(f"{base_dir}/{annotator_folder}/{video_folder}/{segment}")

    # with open("../training_data_location.csv", 'w') as file:
    #     csvwriter = csv.writer(file)

    #     csvwriter.writerows(video_list)

    return video_list


def extract_feats_pretrained_cnn():
    """
    saves the numpy features from all the videos
    """
    model = model_cnn_load()
    print('Model loaded')

    if not os.path.isdir(os.path.join(config.train_path, 'feat')):
        os.mkdir(os.path.join(config.train_path, 'feat'))

    #video_list = os.listdir(os.path.join(config.train_path, 'video'))
    #video_list = return_list_of_videos()
    video_list = []
    with(open("../epic_kitchens/annotations/training_data.csv", 'r')) as file:
        csvreader = csv.reader(file)
        for line in csvreader:
            file_name = f"{line[2]}_{line[0]}.MP4"
            video_list.append(file_name)
    #First 2000 videos extracted on the 5th Dec
    #2001:7968 - 6th Dec
    video_list = video_list[23990:]
    
    #ًWhen running the script on Colab an item called '.ipynb_checkpoints' 
    #is added to the beginning of the list causing errors later on, so the next line removes it.
    # Substitute test_path with train_path if you're extracting features for training data
    if '.ipynb_checkpoints' in video_list:
      video_list.remove('.ipynb_checkpoints')
    if '.gitignore' in video_list:
      video_list.remove('.gitignore')
    if '.DS_Store' in video_list:
      video_list.remove('.DS_Store')
    print("Length of list: ", len(video_list))
    counter = 1
    
    for video in video_list:

        outfile = os.path.join(config.train_path, 'feat', video + '.npy')
        img_feats = extract_features(video, model)
        np.save(outfile, img_feats)
        print("Saved video: " + str(counter) + "/" + str(len(video_list)))
        counter += 1


if __name__ == "__main__":
    extract_feats_pretrained_cnn()
