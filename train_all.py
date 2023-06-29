'''
Sidharrth Nagappan
sidharrth2002@gmail.com
'''

# import libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import ResNet50
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow.python.client import device_lib
import pandas as pd
import numpy as np
import glob
import random
import locale
from PIL import Image
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
#import tensorflow_addons as tfa
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.layers import concatenate, Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History, CSVLogger
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from tensorflow.keras.utils import plot_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from tensorflow.keras.optimizers import Adam
import string
from keras.preprocessing import image as krs_image
import gc
import argparse
import logging
import json
from keras.utils.layer_utils import count_params
from vit_keras import vit, utils

def plot_train_history(history):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].plot(history.history['dominance_output_mae'],
                 label='Dominance Train MAE')
    axes[0].plot(history.history['val_dominance_output_mae'],
                 label='Dominance Val MAE')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(history.history['arousal_output_mae'],
                 label='Arousal Train MAE')
    axes[1].plot(history.history['val_arousal_output_mae'],
                 label='Arousal Val MAE')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

    axes[2].plot(history.history['valence_output_mae'],
                 label='Valence Train MAE')
    axes[2].plot(history.history['val_valence_output_mae'],
                 label='Valence Val MAE')
    axes[2].set_xlabel('Epochs')
    axes[2].legend()

    axes[3].plot(history.history['loss'], label='Training loss')
    axes[3].plot(history.history['val_loss'], label='Validation loss')
    axes[3].set_xlabel('Epochs')
    axes[3].legend()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='ResNet', help="model name")
parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
parser.add_argument("--finetune_only", action= "store_true", default=False, help="finetune")
parser.add_argument("--object_features", action= "store_true", default=False, help="should you include object features")
parser.add_argument("--scene_features", action= "store_true", default=False, help="should you include scene features")
parser.add_argument("--no_round", action= "store_true", default=False, help="should you round numerical features")
parser.add_argument("--resnet", default=False, required=False, action="store_true", help="should you include resnet image model")
parser.add_argument("--mobilenet", default=False, required=False, action="store_true", help="should you include mobilenet image model")
parser.add_argument("--use_attention", default=False, required=False, action="store_true", help="should you use attention")
parser.add_argument("--efficientnet", default=False, required=False, action="store_true", help="should you include efficientnet image model")
parser.add_argument("--continue_training", default=False, required=False, action="store_true", help="should you continue training")
parser.add_argument("--vit", default=False, required=False, action="store_true", help="should you include vit image model")

args = parser.parse_args()

print(f"Starting training for model {args.model_name}")

# feature selection libraries
color_gt = ['RGB-R', 'RGB-G', 'RGB-B', 'HSL-H', 'HSL-S', 'HSL-L']
object_gt = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle',
             'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair',
             'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe',
             'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorbike', 'mouse', 'orange',
             'oven', 'parking meter', 'person', 'pizza', 'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep',
             'sink', 'skateboard', 'skis', 'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard',
             'teddy bear', 'tennis racket', 'tie', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tvmonitor', 'umbrella',
             'vase', 'wine glass', 'zebra']
scene_gt = ['Indoor/Outdoor', 'shopping and dining', 'workplace (office building, factory, lab, etc.)',
            'home or hotel', 'transportation (vehicle interiors, stations, etc.)', 'sports and leisure', 'cultural (art, education, religion, millitary, law, politics, etc.)',
            'water, ice, snow', 'mountains, hills, desert, sky', 'forest, field, jungle', 'man-made elements', 'transportation (roads, parking, bridges, boats, airports, etc.)',
            'cultural or historical building/place (millitary, religious)', 'sports fields, parks, leisure spaces', 'industrial and construction', 'houses, cabins, gardens, and farms',
            'commercial buildings, shops, markets, cities, and towns', 'airfield', 'airplane_cabin', 'airport_terminal', 'alcove', 'alley', 'amphitheater', 'amusement_arcade',
            'amusement_park', 'apartment_building/outdoor', 'aquarium', 'aqueduct', 'arcade', 'arch', 'archaelogical_excavation', 'archive', 'arena/hockey', 'arena/performance',
            'arena/rodeo', 'army_base', 'art_gallery', 'art_school', 'art_studio', 'artists_loft', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium',
            'auto_factory', 'auto_showroom', 'badlands', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'bank_vault', 'banquet_hall',
            'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basketball_court/indoor', 'bathroom', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beach_house', 'beauty_salon',
            'bedchamber', 'bedroom', 'beer_garden', 'beer_hall', 'berth', 'biology_laboratory', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden',
            'bow_window/indoor', 'bowling_alley', 'boxing_ring', 'bridge', 'bullring', 'burial_chamber', 'bus_interior', 'bus_station/indoor', 'butchers_shop', 'butte', 'cabin/outdoor',
            'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior', 'carrousel', 'castle', 'catacomb', 'cemetery', 'chalet', 'chemistry_lab',
            'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room',
            'conference_center', 'conference_room', 'construction_site', 'corn_field', 'corral', 'corridor', 'cottage', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk',
            'dam', 'delicatessen', 'department_store', 'desert/sand', 'desert/vegetation', 'desert_road', 'diner/outdoor', 'dining_hall', 'dining_room', 'discotheque', 'doorway/outdoor',
            'dorm_room', 'downtown', 'dressing_room', 'driveway', 'drugstore', 'elevator/door', 'elevator_lobby', 'elevator_shaft', 'embassy', 'engine_room', 'entrance_hall',
            'escalator/indoor', 'excavation', 'fabric_store', 'farm', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'field_road', 'fire_escape', 'fire_station', 'fishpond',
            'flea_market/indoor', 'florist_shop/indoor', 'food_court', 'football_field', 'forest/broadleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley',
            'garage/indoor', 'garage/outdoor', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'glacier', 'golf_course', 'greenhouse/indoor',
            'greenhouse/outdoor', 'grotto', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hardware_store', 'hayfield', 'heliport', 'highway', 'home_office',
            'home_theater', 'hospital', 'hospital_room', 'hot_spring', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf',
            'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail_cell', 'japanese_garden',
            'jewelry_shop', 'junkyard', 'kasbah', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'lagoon', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lawn',
            'lecture_room', 'legislative_chamber', 'library/indoor', 'library/outdoor', 'lighthouse', 'living_room', 'loading_dock', 'lobby', 'lock_chamber', 'locker_room', 'mansion',
            'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'mezzanine', 'moat/water', 'mosque/outdoor', 'motel', 'mountain',
            'mountain_path', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'museum/outdoor', 'music_studio', 'natural_history_museum', 'nursery', 'nursing_home', 'oast_house',
            'ocean', 'office', 'office_building', 'office_cubicles', 'oilrig', 'operating_room', 'orchard', 'orchestra_pit', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor',
            'parking_garage/outdoor', 'parking_lot', 'pasture', 'patio', 'pavilion', 'pet_shop', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pier', 'pizzeria', 'playground',
            'playroom', 'plaza', 'pond', 'porch', 'promenade', 'pub/indoor', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'repair_shop',
            'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'river', 'rock_arch', 'roof_garden', 'rope_bridge', 'ruin', 'runway', 'sandbox',
            'sauna', 'schoolhouse', 'science_museum', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum',
            'snowfield', 'soccer_field', 'stable', 'stadium/baseball', 'stadium/football', 'stadium/soccer', 'stage/indoor', 'stage/outdoor', 'staircase', 'storage_room', 'street',
            'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_hole', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/outdoor', 'television_room',
            'television_studio', 'temple/asia', 'throne_room', 'ticket_booth', 'topiary_garden', 'tower', 'toyshop', 'train_interior', 'train_station/platform', 'tree_farm', 'tree_house',
            'trench', 'tundra', 'underwater/ocean_deep', 'utility_room', 'valley', 'vegetable_garden', 'veterinarians_office', 'viaduct', 'village', 'vineyard', 'volcano',
            'volleyball_court/outdoor', 'waiting_room', 'water_park', 'water_tower', 'waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'yard',
            'youth_hostel', 'zen_garden']


# image features data in csv
train_x_data = pd.read_csv('../../data/train_x.csv', index_col='iid')
test_x_data = pd.read_csv('../../data/test_x.csv', index_col="iid")
valid_x_data = pd.read_csv('../../data/valid_x.csv', index_col="iid")

if not args.no_round:
    train_x_data[object_gt] = train_x_data[object_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
    train_x_data[scene_gt] = train_x_data[scene_gt].applymap(lambda x: 1 if x >= 0.5 else 0)

    valid_x_data[object_gt] = valid_x_data[object_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
    valid_x_data[scene_gt] = valid_x_data[scene_gt].applymap(lambda x: 1 if x >= 0.5 else 0)

    test_x_data[object_gt] = test_x_data[object_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
    test_x_data[scene_gt] = test_x_data[scene_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
else:
    print("No rounding will be applied")

rejected_features = color_gt
if not args.object_features:
    rejected_features += object_gt

if not args.scene_features:
    rejected_features += scene_gt

print(train_x_data[scene_gt].isna().sum())

train_x_data.drop(columns=rejected_features, inplace=True)
test_x_data.drop(columns=rejected_features, inplace=True)
valid_x_data.drop(columns=rejected_features, inplace=True)

trainAttrX = train_x_data

train_y_data = pd.read_csv('../../data/train_y.csv', index_col='iid')
test_y_data = pd.read_csv('../../data/test_y.csv', index_col="iid")
valid_y_data = pd.read_csv('../../data/valid_y.csv', index_col="iid")

# image data
train_file_list = glob.glob('../../train/*.jpg', recursive=True)
random.shuffle(train_file_list)

test_file_list = glob.glob('../../test/*.jpg', recursive=True)
random.shuffle(test_file_list)

valid_file_list = glob.glob('../../valid/*.jpg', recursive=True)
random.shuffle(valid_file_list)

train_file_list = [x for x in train_file_list if "(1)" not in x]
test_file_list = [x for x in test_file_list if "(1)" not in x]
valid_file_list = [x for x in valid_file_list if "(1)" not in x]


# Create the arguments for image preprocessing
data_gen_args = dict(
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    shear_range=10,
    channel_shift_range=50,
    rescale=1. / 255,
)

# Create an empty data generator
datagen = ImageDataGenerator()


def my_gen(images_list, dataframe, df_y, batch_size):
    i = 0
    while True:
        batch = {'img_input': [], 'csv_input': [],
                 'valence': [], 'arousal': [], 'dominance': []}
        for b in range(batch_size):
            if i == len(images_list):
                i = 0
                random.shuffle(images_list)
            # Read image from list and convert to array
            image_path = images_list[i]
            image_name = os.path.basename(image_path).replace('.JPG', '')
            #image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
            image = krs_image.load_img(image_path, target_size=(
                224, 224))  # image height & width
            image = datagen.apply_transform(image, data_gen_args)
            image = krs_image.img_to_array(image)
            #image = tf.keras.utils.img_to_array(image)

            # Read data from csv using the name of current image
            csv_features = dataframe.loc[image_name, :]
            y = df_y.loc[image_name, :]
            val_train = y[0]
            aro_train = y[1]
            dom_train = y[2]

            # print(image_name)
            batch['img_input'].append(image)
            batch['csv_input'].append(csv_features)
            batch['valence'].append(val_train)
            batch['arousal'].append(aro_train)
            batch['dominance'].append(dom_train)

            i += 1

        batch['img_input'] = np.array(batch['img_input'])
        batch['csv_input'] = np.array(batch['csv_input'])
        batch['valence'] = np.array(batch['valence'])
        batch['arousal'] = np.array(batch['arousal'])
        batch['dominance'] = np.array(batch['dominance'])

        yield [batch['csv_input'], batch['img_input']], [batch['valence'], batch['arousal'], batch['dominance']]

gen_train = my_gen(train_file_list, train_x_data, train_y_data, 32)
gen_test = my_gen(test_file_list, test_x_data, test_y_data, 32)
gen_valid = my_gen(valid_file_list, valid_x_data, valid_y_data, 32)

def self_attention(inputs_i, inputs_n, name):
        """
        :param inputs_a: image input (B, T, dim)
        :param inputs_v: numerical input (B, T, dim)
        :param name: scope name
        :return:
        """

        inputs_i = tf.expand_dims(inputs_i, axis=1)
        inputs_n = tf.expand_dims(inputs_n, axis=1)
        # inputs = (B, 2, T, dim)
        inputs = tf.concat([inputs_i, inputs_n], axis=1)
        t = inputs.get_shape()[2]
        share_param = True
        hidden_size = inputs.shape[-1]  # D value - hidden size of the RNN layer
        kernel_init = tf.compat.v1.glorot_uniform_initializer(seed=0, dtype=tf.float32)
        # bias_init = tf.zeros_initializer()
        dense = Dense(hidden_size, kernel_initializer=kernel_init)
        if share_param:
            scope_name = 'self_attn'
        else:
            scope_name = 'self_attn' + name
        # print(scope_name)
        inputs = tf.transpose(inputs, [2, 0, 1])
        with tf.compat.v1.variable_scope(scope_name):
            outputs = []
            for x in range(t):
                t_x = inputs[x, :, :]
                # t_x => B, 2, dim
                den = True
                if den:
                    x_proj = dense(t_x)
                    x_proj = tf.nn.tanh(x_proj)
                else:
                    x_proj = t_x
                u_w = tf.Variable(tf.compat.v1.random_normal([hidden_size, 1], stddev=0.01, seed=1234))
                x = tf.tensordot(x_proj, u_w, axes=1)
                alphas = tf.nn.softmax(x, axis=-1)
                print(f"Shape of alphas: {alphas.shape}")
                output = tf.matmul(tf.transpose(t_x, [1, 0]), alphas)
                output = tf.squeeze(output, -1)
                outputs.append(output)

            final_output = tf.stack(outputs, axis=1)
            # print('final_output', final_output.get_shape())
            return final_output

def create_resnet(width, height, depth):
    # inputs= Input(shape=(width, height, depth), name='img_input')

    res = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(width, height, depth),
    )

    for layer in res.layers:
        layer.trainable = False  # false: freeze, true:train by own

    x = layers.BatchNormalization()(res.output)
    top_dropout_rate = args.dropout
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Flatten()(x)
    x = Dense(1024, activation="relu", name='csv_img')(x)
    x = Dense(224, activation="relu")(x)

    model = Model(res.input, x, name="ResNet")

    return model, [i.name for i in res.layers]

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def create_mobilenet(width, height, depth):
    # inputs= Input(shape=(width, height, depth), name='img_input')

    res = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape = (width, height, depth),
    )

    for layer in res.layers:
        layer.trainable = False #false: freeze, true:train by own

    x = layers.BatchNormalization()(res.output)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Flatten()(x)

    x = Dense(1024, activation="relu",name='csv_img')(x)
    x = Dense(512, activation="relu")(x)

    model = Model(res.input, x, name="ResNet")
    return model, [i.name for i in res.layers]

def create_efficientNet(width, height, depth):

    # inputs= Input(shape=(width, height, depth), name='img_input')

    eff = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape = (width, height, depth),
        pooling='max'
    )

    eff.trainable = False

    # for layer in eff.layers:
    #     layer.trainable = False #false: freeze, true:train by own

    # x = layers.GlobalAveragePooling2D(name="avg_pool")(eff.output)
    x = layers.BatchNormalization()(eff.output)
    top_dropout_rate = args.dropout
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Flatten()(x)

    x = Dense(1024, activation="relu",name='csv_img')(x)
    x = Dense(512, activation="relu")(x)

    model = Model(eff.input, x, name="EfficientNet")
    return model, [i.name for i in eff.layers]

def create_vit(width, height, depth):
    # inputs= Input(shape=(width, height, depth), name='img_input')

    res = vit.vit_b32(
        image_size=width,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=1000,
    )

    for layer in res.layers:
        layer.trainable = False  # false: freeze, true:train by own

    x = layers.BatchNormalization()(res.output)
    top_dropout_rate = args.dropout
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Flatten()(x)
    x = Dense(1024, activation="relu", name='csv_img')(x)
    x = Dense(224, activation="relu")(x)

    model = Model(res.input, x, name="ResNet")

    return model, [i.name for i in res.layers]

def create_mlp(dim):
    # define our MLP network
    model = Sequential()
    model.add(Dense(416, input_dim=dim, activation="relu", name='csv'))
    model.add(Dense(224, activation="relu"))

    return model

mlp = create_mlp(trainAttrX.shape[1])
to_concatenate = [mlp.output]
image_model = None
if args.resnet:
    resnet = create_resnet(224, 224, 3)
    # store resnet layers
    image_model, image_model_layers = resnet
    to_concatenate.append(image_model.output)

elif args.mobilenet:
    mobilenet = create_mobilenet(224, 224, 3)
    image_model, image_model_layers = mobilenet
    # store mobilenet layers
    to_concatenate.append(image_model.output)

elif args.efficientnet:
    effnet = create_efficientNet(224, 224, 3)
    image_model, image_model_layers = effnet
    to_concatenate.append(image_model.output)

elif args.vit:
    vit = create_vit(224, 224, 3)
    image_model, image_model_layers = vit
    to_concatenate.append(image_model.output)

if not args.finetune_only:

    if args.use_attention:
        combinedInput = self_attention(image_model.output, mlp.output, 'combined')
    else:
        combinedInput = concatenate(to_concatenate)

    print(combinedInput)

    x = tf.keras.layers.Flatten()(combinedInput)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)

    valence_output = Dense(units=1, activation='linear', name='valence_output')(x)
    arousal_output = Dense(units=1, activation='linear', name='arousal_output')(x)
    dominance_output = Dense(units=1, activation='linear', name='dominance_output')(x)

    model = Model(inputs=[mlp.input, image_model.input], outputs=[
                valence_output, arousal_output, dominance_output])

    opt = Adam(learning_rate=0.001)

    model.compile(optimizer=opt, run_eagerly=True,
                loss={'valence_output': 'mse', 'arousal_output': 'mse',
                        'dominance_output': 'mse'},
                metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6,
                    mode='min', min_delta=0.0001),
        ModelCheckpoint(f'./trained_models/{args.model_name}',
                        monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        CSVLogger(
            f'{args.model_name}_results.csv', separator=',', append=True
        )
    ]

    batch_size = 32
    trainidx = len(train_file_list)
    testidx = len(test_file_list)
    valididx = len(valid_file_list)

    if args.continue_training:
        print("Continuing training")
        model = None
        gc.collect()
        model = load_model(f'./trained_models/{args.model_name}')

        for layer in model.layers:
            if layer.name in image_model_layers:
                layer.trainable = False

        model.compile(optimizer=opt, run_eagerly=True,
            loss={'valence_output': 'mse', 'arousal_output': 'mse',
                    'dominance_output': 'mse'},
            metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})

        model.summary()

    history = model.fit(gen_train, epochs=1000, steps_per_epoch=trainidx/batch_size,
                        validation_data=gen_valid, validation_steps=valididx/batch_size, callbacks=callbacks, batch_size=32)

    plot_train_history(history)

# assume saved in memory, delete
model = None
del model
gc.collect()

print("Loading model to finetune...")
# load best model to finetune
model = load_model(f"./trained_models/{args.model_name}")

# only applies for Resnet
fine_tune_at = 150

print(f"Before unfreezing layers params: ")
import numpy as np

trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
total_params = trainable_params + non_trainable_params

print(trainable_params)
print(non_trainable_params)
print(total_params)
print(f"Number of resnet layers: {len(image_model_layers)}")
# only finetune Resnet layers
counter = 0
# layers_to_finetune = []

if args.vit:
    # finetune all layers
    for layer in model.layers:
        layer.trainable = True
else:
    if args.efficientnet:
        finetune_from = 20
    else:
        finetune_from = 150

    for layer in model.layers:
        # check if the layer is from Resnet
        if layer.name in image_model_layers and counter >= (len(image_model_layers) - finetune_from):
            print(f"Found trainable layer: {layer.name}")
            layer.trainable = True
            counter += 1
        elif layer.name in image_model_layers:
            layer.trainable = False
            counter += 1

# for layer in layers_to_finetune[fine_tune_at:]:
#     layer.trainable = True

# for layer in layers_to_finetune[:fine_tune_at]:
#     layer.trainable = False

opt = Adam(learning_rate=0.0001)

model.compile(optimizer=opt, run_eagerly=True,
              loss={'valence_output': 'mse', 'arousal_output': 'mse',
                    'dominance_output': 'mse'},
              metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})

print(f"After unfreezing layers params: ")
trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
total_params = trainable_params + non_trainable_params

print(trainable_params)
print(non_trainable_params)
print(total_params)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5,
                  mode='min', min_delta=0.0001),
    ModelCheckpoint(f'./trained_models/{args.model_name}_finetuned',
                    monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    CSVLogger(
        f'{args.model_name}_finetuned_results.csv', separator=',', append=True
    )
]

batch_size = 32
trainidx = len(train_file_list)
testidx = len(test_file_list)
valididx = len(valid_file_list)

print("Finetuning model...")

history = model.fit(gen_train, epochs=20, steps_per_epoch=trainidx/batch_size, validation_data=gen_valid,
                    validation_steps=valididx/batch_size, callbacks=callbacks, batch_size=32)

plot_train_history(history)

print("Evaluating model on test data")

test_results = model.evaluate(gen_test, steps=testidx/batch_size)

print(test_results)

print("Writing test results to file")
with open(f'{args.model_name}_test_results.json', 'w') as f:
    json.dump(test_results, f)

print("Done")
print("Clearing memory")

model = None
del model
gc.collect()

print("END")

# '''
# Sidharrth Nagappan
# sidharrth2002@gmail.com
# '''

# # import libraries
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet import ResNet50
# from sklearn.feature_selection import RFE
# import tensorflow as tf
# from tensorflow.python.client import device_lib
# import pandas as pd
# import numpy as np
# import glob
# import random
# import locale
# from PIL import Image
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from tensorflow import keras
# #import tensorflow_addons as tfa
# from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
# from tensorflow.keras import layers, models, Model, Input
# from tensorflow.keras.layers import concatenate, Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History, CSVLogger
# from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
# from tensorflow.keras.utils import plot_model
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers.core import Activation, Dropout, Dense
# from tensorflow.keras.optimizers import Adam
# import string
# from keras.preprocessing import image as krs_image
# import gc
# import argparse
# import logging
# import json
# from keras.utils.layer_utils import count_params

# def plot_train_history(history):
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#     axes[0].plot(history.history['dominance_output_mae'],
#                  label='Dominance Train MAE')
#     axes[0].plot(history.history['val_dominance_output_mae'],
#                  label='Dominance Val MAE')
#     axes[0].set_xlabel('Epochs')
#     axes[0].legend()

#     axes[1].plot(history.history['arousal_output_mae'],
#                  label='Arousal Train MAE')
#     axes[1].plot(history.history['val_arousal_output_mae'],
#                  label='Arousal Val MAE')
#     axes[1].set_xlabel('Epochs')
#     axes[1].legend()

#     axes[2].plot(history.history['valence_output_mae'],
#                  label='Valence Train MAE')
#     axes[2].plot(history.history['val_valence_output_mae'],
#                  label='Valence Val MAE')
#     axes[2].set_xlabel('Epochs')
#     axes[2].legend()

#     axes[3].plot(history.history['loss'], label='Training loss')
#     axes[3].plot(history.history['val_loss'], label='Validation loss')
#     axes[3].set_xlabel('Epochs')
#     axes[3].legend()

# # parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_name", type=str, default='ResNet', help="model name")
# parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
# parser.add_argument("--finetune_only", action= "store_true", default=False, help="finetune")
# parser.add_argument("--object_features", action= "store_true", default=False, help="should you include object features")
# parser.add_argument("--scene_features", action= "store_true", default=False, help="should you include scene features")
# parser.add_argument("--no_round", action= "store_true", default=False, help="should you round numerical features")
# parser.add_argument("--resnet", default=True, required=False, action="store_true", help="should you include resnet image model")
# parser.add_argument("--mobilenet", default=False, required=False, action="store_true", help="should you include mobilenet image model")
# parser.add_argument("--use_attention", default=False, required=False, action="store_true", help="should you use attention")
# parser.add_argument("--efficientnet", default=False, required=False, action="store_true", help="should you include efficientnet image model")
# parser.add_argument("--continue_training", default=False, required=False, action="store_true", help="should you continue training")

# args = parser.parse_args()

# print(f"Starting training for model {args.model_name}")

# # feature selection libraries
# color_gt = ['RGB-R', 'RGB-G', 'RGB-B', 'HSL-H', 'HSL-S', 'HSL-L']
# object_gt = ['aeroplane', 'apple', 'backpack', 'banana', 'baseball bat', 'baseball glove', 'bear', 'bed', 'bench', 'bicycle',
#              'bird', 'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell phone', 'chair',
#              'clock', 'cow', 'cup', 'diningtable', 'dog', 'donut', 'elephant', 'fire hydrant', 'fork', 'frisbee', 'giraffe',
#              'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorbike', 'mouse', 'orange',
#              'oven', 'parking meter', 'person', 'pizza', 'pottedplant', 'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep',
#              'sink', 'skateboard', 'skis', 'snowboard', 'sofa', 'spoon', 'sports ball', 'stop sign', 'suitcase', 'surfboard',
#              'teddy bear', 'tennis racket', 'tie', 'toilet', 'toothbrush', 'traffic light', 'train', 'truck', 'tvmonitor', 'umbrella',
#              'vase', 'wine glass', 'zebra']
# scene_gt = ['Indoor/Outdoor', 'shopping and dining', 'workplace (office building, factory, lab, etc.)',
#             'home or hotel', 'transportation (vehicle interiors, stations, etc.)', 'sports and leisure', 'cultural (art, education, religion, millitary, law, politics, etc.)',
#             'water, ice, snow', 'mountains, hills, desert, sky', 'forest, field, jungle', 'man-made elements', 'transportation (roads, parking, bridges, boats, airports, etc.)',
#             'cultural or historical building/place (millitary, religious)', 'sports fields, parks, leisure spaces', 'industrial and construction', 'houses, cabins, gardens, and farms',
#             'commercial buildings, shops, markets, cities, and towns', 'airfield', 'airplane_cabin', 'airport_terminal', 'alcove', 'alley', 'amphitheater', 'amusement_arcade',
#             'amusement_park', 'apartment_building/outdoor', 'aquarium', 'aqueduct', 'arcade', 'arch', 'archaelogical_excavation', 'archive', 'arena/hockey', 'arena/performance',
#             'arena/rodeo', 'army_base', 'art_gallery', 'art_school', 'art_studio', 'artists_loft', 'assembly_line', 'athletic_field/outdoor', 'atrium/public', 'attic', 'auditorium',
#             'auto_factory', 'auto_showroom', 'badlands', 'bakery/shop', 'balcony/exterior', 'balcony/interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'bank_vault', 'banquet_hall',
#             'bar', 'barn', 'barndoor', 'baseball_field', 'basement', 'basketball_court/indoor', 'bathroom', 'bazaar/indoor', 'bazaar/outdoor', 'beach', 'beach_house', 'beauty_salon',
#             'bedchamber', 'bedroom', 'beer_garden', 'beer_hall', 'berth', 'biology_laboratory', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth/indoor', 'botanical_garden',
#             'bow_window/indoor', 'bowling_alley', 'boxing_ring', 'bridge', 'bullring', 'burial_chamber', 'bus_interior', 'bus_station/indoor', 'butchers_shop', 'butte', 'cabin/outdoor',
#             'campsite', 'campus', 'canal/natural', 'canal/urban', 'candy_store', 'canyon', 'car_interior', 'carrousel', 'castle', 'catacomb', 'cemetery', 'chalet', 'chemistry_lab',
#             'childs_room', 'church/indoor', 'church/outdoor', 'classroom', 'clean_room', 'cliff', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room',
#             'conference_center', 'conference_room', 'construction_site', 'corn_field', 'corral', 'corridor', 'cottage', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk',
#             'dam', 'delicatessen', 'department_store', 'desert/sand', 'desert/vegetation', 'desert_road', 'diner/outdoor', 'dining_hall', 'dining_room', 'discotheque', 'doorway/outdoor',
#             'dorm_room', 'downtown', 'dressing_room', 'driveway', 'drugstore', 'elevator/door', 'elevator_lobby', 'elevator_shaft', 'embassy', 'engine_room', 'entrance_hall',
#             'escalator/indoor', 'excavation', 'fabric_store', 'farm', 'fastfood_restaurant', 'field/cultivated', 'field/wild', 'field_road', 'fire_escape', 'fire_station', 'fishpond',
#             'flea_market/indoor', 'florist_shop/indoor', 'food_court', 'football_field', 'forest/broadleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley',
#             'garage/indoor', 'garage/outdoor', 'gas_station', 'gazebo/exterior', 'general_store/indoor', 'general_store/outdoor', 'gift_shop', 'glacier', 'golf_course', 'greenhouse/indoor',
#             'greenhouse/outdoor', 'grotto', 'gymnasium/indoor', 'hangar/indoor', 'hangar/outdoor', 'harbor', 'hardware_store', 'hayfield', 'heliport', 'highway', 'home_office',
#             'home_theater', 'hospital', 'hospital_room', 'hot_spring', 'hotel/outdoor', 'hotel_room', 'house', 'hunting_lodge/outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf',
#             'ice_skating_rink/indoor', 'ice_skating_rink/outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn/outdoor', 'islet', 'jacuzzi/indoor', 'jail_cell', 'japanese_garden',
#             'jewelry_shop', 'junkyard', 'kasbah', 'kennel/outdoor', 'kindergarden_classroom', 'kitchen', 'lagoon', 'lake/natural', 'landfill', 'landing_deck', 'laundromat', 'lawn',
#             'lecture_room', 'legislative_chamber', 'library/indoor', 'library/outdoor', 'lighthouse', 'living_room', 'loading_dock', 'lobby', 'lock_chamber', 'locker_room', 'mansion',
#             'manufactured_home', 'market/indoor', 'market/outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'mezzanine', 'moat/water', 'mosque/outdoor', 'motel', 'mountain',
#             'mountain_path', 'mountain_snowy', 'movie_theater/indoor', 'museum/indoor', 'museum/outdoor', 'music_studio', 'natural_history_museum', 'nursery', 'nursing_home', 'oast_house',
#             'ocean', 'office', 'office_building', 'office_cubicles', 'oilrig', 'operating_room', 'orchard', 'orchestra_pit', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage/indoor',
#             'parking_garage/outdoor', 'parking_lot', 'pasture', 'patio', 'pavilion', 'pet_shop', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pier', 'pizzeria', 'playground',
#             'playroom', 'plaza', 'pond', 'porch', 'promenade', 'pub/indoor', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'repair_shop',
#             'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'river', 'rock_arch', 'roof_garden', 'rope_bridge', 'ruin', 'runway', 'sandbox',
#             'sauna', 'schoolhouse', 'science_museum', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall/indoor', 'shower', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum',
#             'snowfield', 'soccer_field', 'stable', 'stadium/baseball', 'stadium/football', 'stadium/soccer', 'stage/indoor', 'stage/outdoor', 'staircase', 'storage_room', 'street',
#             'subway_station/platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_hole', 'swimming_pool/indoor', 'swimming_pool/outdoor', 'synagogue/outdoor', 'television_room',
#             'television_studio', 'temple/asia', 'throne_room', 'ticket_booth', 'topiary_garden', 'tower', 'toyshop', 'train_interior', 'train_station/platform', 'tree_farm', 'tree_house',
#             'trench', 'tundra', 'underwater/ocean_deep', 'utility_room', 'valley', 'vegetable_garden', 'veterinarians_office', 'viaduct', 'village', 'vineyard', 'volcano',
#             'volleyball_court/outdoor', 'waiting_room', 'water_park', 'water_tower', 'waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'yard',
#             'youth_hostel', 'zen_garden']


# # image features data in csv
# train_x_data = pd.read_csv('../../data/train_x.csv', index_col='iid')
# test_x_data = pd.read_csv('../../data/test_x.csv', index_col="iid")
# valid_x_data = pd.read_csv('../../data/valid_x.csv', index_col="iid")

# if not args.no_round:
#     train_x_data[object_gt] = train_x_data[object_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
#     train_x_data[scene_gt] = train_x_data[scene_gt].applymap(lambda x: 1 if x >= 0.5 else 0)

#     valid_x_data[object_gt] = valid_x_data[object_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
#     valid_x_data[scene_gt] = valid_x_data[scene_gt].applymap(lambda x: 1 if x >= 0.5 else 0)

#     test_x_data[object_gt] = test_x_data[object_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
#     test_x_data[scene_gt] = test_x_data[scene_gt].applymap(lambda x: 1 if x >= 0.5 else 0)
# else:
#     print("No rounding will be applied")

# rejected_features = color_gt
# if not args.object_features:
#     rejected_features += object_gt

# if not args.scene_features:
#     rejected_features += scene_gt

# print(train_x_data[scene_gt].isna().sum())

# train_x_data.drop(columns=rejected_features, inplace=True)
# test_x_data.drop(columns=rejected_features, inplace=True)
# valid_x_data.drop(columns=rejected_features, inplace=True)

# trainAttrX = train_x_data

# train_y_data = pd.read_csv('../../data/train_y.csv', index_col='iid')
# test_y_data = pd.read_csv('../../data/test_y.csv', index_col="iid")
# valid_y_data = pd.read_csv('../../data/valid_y.csv', index_col="iid")

# # image data
# train_file_list = glob.glob('../../train/*.jpg', recursive=True)
# random.shuffle(train_file_list)

# test_file_list = glob.glob('../../test/*.jpg', recursive=True)
# random.shuffle(test_file_list)

# valid_file_list = glob.glob('../../valid/*.jpg', recursive=True)
# random.shuffle(valid_file_list)

# train_file_list = [x for x in train_file_list if "(1)" not in x]
# test_file_list = [x for x in test_file_list if "(1)" not in x]
# valid_file_list = [x for x in valid_file_list if "(1)" not in x]


# # Create the arguments for image preprocessing
# data_gen_args = dict(
#     horizontal_flip=True,
#     brightness_range=[0.5, 1.5],
#     shear_range=10,
#     channel_shift_range=50,
#     rescale=1. / 255,
# )

# # Create an empty data generator
# datagen = ImageDataGenerator()


# def my_gen(images_list, dataframe, df_y, batch_size):
#     i = 0
#     while True:
#         batch = {'img_input': [], 'csv_input': [],
#                  'valence': [], 'arousal': [], 'dominance': []}
#         for b in range(batch_size):
#             if i == len(images_list):
#                 i = 0
#                 random.shuffle(images_list)
#             # Read image from list and convert to array
#             image_path = images_list[i]
#             image_name = os.path.basename(image_path).replace('.JPG', '')
#             #image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
#             image = krs_image.load_img(image_path, target_size=(
#                 224, 224))  # image height & width
#             image = datagen.apply_transform(image, data_gen_args)
#             image = krs_image.img_to_array(image)
#             #image = tf.keras.utils.img_to_array(image)

#             # Read data from csv using the name of current image
#             csv_features = dataframe.loc[image_name, :]
#             y = df_y.loc[image_name, :]
#             val_train = y[0]
#             aro_train = y[1]
#             dom_train = y[2]

#             # print(image_name)
#             batch['img_input'].append(image)
#             batch['csv_input'].append(csv_features)
#             batch['valence'].append(val_train)
#             batch['arousal'].append(aro_train)
#             batch['dominance'].append(dom_train)

#             i += 1

#         batch['img_input'] = np.array(batch['img_input'])
#         batch['csv_input'] = np.array(batch['csv_input'])
#         batch['valence'] = np.array(batch['valence'])
#         batch['arousal'] = np.array(batch['arousal'])
#         batch['dominance'] = np.array(batch['dominance'])

#         yield [batch['csv_input'], batch['img_input']], [batch['valence'], batch['arousal'], batch['dominance']]

# gen_train = my_gen(train_file_list, train_x_data, train_y_data, 32)
# gen_test = my_gen(test_file_list, test_x_data, test_y_data, 32)
# gen_valid = my_gen(valid_file_list, valid_x_data, valid_y_data, 32)

# def self_attention(inputs_i, inputs_n, name):
#         """
#         :param inputs_a: image input (B, T, dim)
#         :param inputs_v: numerical input (B, T, dim)
#         :param name: scope name
#         :return:
#         """

#         inputs_i = tf.expand_dims(inputs_i, axis=1)
#         inputs_n = tf.expand_dims(inputs_n, axis=1)
#         # inputs = (B, 2, T, dim)
#         inputs = tf.concat([inputs_i, inputs_n], axis=1)
#         t = inputs.get_shape()[2]
#         share_param = True
#         hidden_size = inputs.shape[-1]  # D value - hidden size of the RNN layer
#         kernel_init = tf.compat.v1.glorot_uniform_initializer(seed=0, dtype=tf.float32)
#         # bias_init = tf.zeros_initializer()
#         dense = Dense(hidden_size, kernel_initializer=kernel_init)
#         if share_param:
#             scope_name = 'self_attn'
#         else:
#             scope_name = 'self_attn' + name
#         # print(scope_name)
#         inputs = tf.transpose(inputs, [2, 0, 1])
#         with tf.compat.v1.variable_scope(scope_name):
#             outputs = []
#             for x in range(t):
#                 t_x = inputs[x, :, :]
#                 # t_x => B, 2, dim
#                 den = True
#                 if den:
#                     x_proj = dense(t_x)
#                     x_proj = tf.nn.tanh(x_proj)
#                 else:
#                     x_proj = t_x
#                 u_w = tf.Variable(tf.compat.v1.random_normal([hidden_size, 1], stddev=0.01, seed=1234))
#                 x = tf.tensordot(x_proj, u_w, axes=1)
#                 alphas = tf.nn.softmax(x, axis=-1)
#                 print(f"Shape of alphas: {alphas.shape}")
#                 output = tf.matmul(tf.transpose(t_x, [1, 0]), alphas)
#                 output = tf.squeeze(output, -1)
#                 outputs.append(output)

#             final_output = tf.stack(outputs, axis=1)
#             # print('final_output', final_output.get_shape())
#             return final_output

# def create_resnet(width, height, depth):
#     # inputs= Input(shape=(width, height, depth), name='img_input')

#     res = ResNet50(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(width, height, depth),
#     )

#     for layer in res.layers:
#         layer.trainable = False  # false: freeze, true:train by own

#     x = layers.BatchNormalization()(res.output)
#     top_dropout_rate = args.dropout
#     x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
#     x = layers.Flatten()(x)
#     x = Dense(1024, activation="relu", name='csv_img')(x)
#     x = Dense(224, activation="relu")(x)

#     model = Model(res.input, x, name="ResNet")

#     return model, [i.name for i in res.layers]

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# def create_mobilenet(width, height, depth):
#     # inputs= Input(shape=(width, height, depth), name='img_input')

#     res = MobileNetV2(
#         weights='imagenet',
#         include_top=False,
#         input_shape = (width, height, depth),
#     )

#     for layer in res.layers:
#         layer.trainable = False #false: freeze, true:train by own

#     x = layers.BatchNormalization()(res.output)
#     top_dropout_rate = 0.2
#     x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
#     x = layers.Flatten()(x)

#     x = Dense(1024, activation="relu",name='csv_img')(x)
#     x = Dense(512, activation="relu")(x)

#     model = Model(res.input, x, name="ResNet")
#     return model, [i.name for i in res.layers]

# def create_efficientNet(width, height, depth):

#     # inputs= Input(shape=(width, height, depth), name='img_input')

#     eff = EfficientNetB3(
#         weights='imagenet',
#         include_top=False,
#         input_shape = (width, height, depth),
#         pooling='max'
#     )

#     eff.trainable = False

#     # for layer in eff.layers:
#     #     layer.trainable = False #false: freeze, true:train by own

#     # x = layers.GlobalAveragePooling2D(name="avg_pool")(eff.output)
#     x = layers.BatchNormalization()(eff.output)
#     top_dropout_rate = args.dropout
#     x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
#     x = layers.Flatten()(x)

#     x = Dense(1024, activation="relu",name='csv_img')(x)
#     x = Dense(512, activation="relu")(x)

#     model = Model(eff.input, x, name="EfficientNet")
#     return model, [i.name for i in eff.layers]



# def create_mlp(dim):
#     # define our MLP network
#     model = Sequential()
#     model.add(Dense(416, input_dim=dim, activation="relu", name='csv'))
#     model.add(Dense(224, activation="relu"))

#     return model

# mlp = create_mlp(trainAttrX.shape[1])
# to_concatenate = [mlp.output]
# image_model = None
# if args.resnet:
#     resnet = create_resnet(224, 224, 3)
#     # store resnet layers
#     image_model, image_model_layers = resnet
#     to_concatenate.append(image_model.output)

# elif args.mobilenet:
#     mobilenet = create_mobilenet(224, 224, 3)
#     image_model, image_model_layers = mobilenet
#     # store mobilenet layers
#     to_concatenate.append(image_model.output)

# elif args.efficientnet:
#     effnet = create_efficientNet(224, 224, 3)
#     image_model, image_model_layers = effnet
#     to_concatenate.append(image_model.output)

# if not args.finetune_only:

#     if args.use_attention:
#         combinedInput = self_attention(image_model.output, mlp.output, 'combined')
#     else:
#         combinedInput = concatenate(to_concatenate)

#     print(combinedInput)

#     x = tf.keras.layers.Flatten()(combinedInput)
#     x = Dense(1024, activation="relu")(x)
#     x = Dense(512, activation="relu")(x)

#     valence_output = Dense(units=1, activation='linear', name='valence_output')(x)
#     arousal_output = Dense(units=1, activation='linear', name='arousal_output')(x)
#     dominance_output = Dense(units=1, activation='linear', name='dominance_output')(x)

#     model = Model(inputs=[mlp.input, image_model.input], outputs=[
#                 valence_output, arousal_output, dominance_output])

#     opt = Adam(learning_rate=0.001)

#     model.compile(optimizer=opt, run_eagerly=True,
#                 loss={'valence_output': 'mse', 'arousal_output': 'mse',
#                         'dominance_output': 'mse'},
#                 metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})

#     model.summary()

#     callbacks = [
#         EarlyStopping(monitor='val_loss', patience=15,
#                     mode='min', min_delta=0.0001),
#         ModelCheckpoint(f'./trained_models/{args.model_name}',
#                         monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
#         CSVLogger(
#             f'{args.model_name}_results.csv', separator=',', append=True
#         )
#     ]

#     batch_size = 32
#     trainidx = len(train_file_list)
#     testidx = len(test_file_list)
#     valididx = len(valid_file_list)

#     if args.continue_training:
#         print("Continuing training")
#         model = None
#         gc.collect()
#         model = load_model(f'./trained_models/{args.model_name}')

#         for layer in model.layers:
#             if layer.name in image_model_layers:
#                 layer.trainable = False

#         model.compile(optimizer=opt, run_eagerly=True,
#             loss={'valence_output': 'mse', 'arousal_output': 'mse',
#                     'dominance_output': 'mse'},
#             metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})

#         model.summary()

#     history = model.fit(gen_train, epochs=1000, steps_per_epoch=trainidx/batch_size,
#                         validation_data=gen_valid, validation_steps=valididx/batch_size, callbacks=callbacks, batch_size=32)

#     plot_train_history(history)

# # assume saved in memory, delete
# model = None
# del model
# gc.collect()

# print("Loading model to finetune...")
# # load best model to finetune
# model = load_model(f"./trained_models/{args.model_name}")

# # only applies for Resnet
# fine_tune_at = 150

# print(f"Before unfreezing layers params: ")
# import numpy as np

# trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
# non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
# total_params = trainable_params + non_trainable_params

# print(trainable_params)
# print(non_trainable_params)
# print(total_params)
# print(f"Number of resnet layers: {len(image_model_layers)}")
# # only finetune Resnet layers
# counter = 0
# # layers_to_finetune = []

# if args.efficientnet ==  "efficientnet":
#     finetune_from = 20
# else:
#     finetune_from = 150

# for layer in model.layers:
#     # check if the layer is from Resnet
#     if layer.name in image_model_layers and counter >= (len(image_model_layers) - finetune_from):
#         print(f"Found trainable layer: {layer.name}")
#         layer.trainable = True
#         counter += 1
#     elif layer.name in image_model_layers:
#         layer.trainable = False
#         counter += 1

# # for layer in layers_to_finetune[fine_tune_at:]:
# #     layer.trainable = True

# # for layer in layers_to_finetune[:fine_tune_at]:
# #     layer.trainable = False

# opt = Adam(learning_rate=0.0001)

# model.compile(optimizer=opt, run_eagerly=True,
#               loss={'valence_output': 'mse', 'arousal_output': 'mse',
#                     'dominance_output': 'mse'},
#               metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})

# print(f"After unfreezing layers params: ")
# trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
# non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
# total_params = trainable_params + non_trainable_params

# print(trainable_params)
# print(non_trainable_params)
# print(total_params)

# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=10,
#                   mode='min', min_delta=0.0001),
#     ModelCheckpoint(f'./trained_models/{args.model_name}_finetuned',
#                     monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
#     CSVLogger(
#         f'{args.model_name}_finetuned_results.csv', separator=',', append=True
#     )
# ]

# batch_size = 32
# trainidx = len(train_file_list)
# testidx = len(test_file_list)
# valididx = len(valid_file_list)

# print("Finetuning model...")

# history = model.fit(gen_train, epochs=20, steps_per_epoch=trainidx/batch_size, validation_data=gen_valid,
#                     validation_steps=valididx/batch_size, callbacks=callbacks, batch_size=32)

# plot_train_history(history)

# print("Evaluating model on test data")

# test_results = model.evaluate(gen_test, steps=testidx/batch_size)

# print(test_results)

# print("Writing test results to file")
# with open(f'{args.model_name}_test_results.json', 'w') as f:
#     json.dump(test_results, f)

# print("Done")
# print("Clearing memory")

# model = None
# del model
# gc.collect()

# print("END")