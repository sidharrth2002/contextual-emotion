#!/usr/bin/env python
# coding: utf-8

from tensorboard import summary
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
from tensorflow.keras import layers, models, Model, Input
from tensorflow.keras.layers import concatenate, Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History, CSVLogger
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Dropout,Dense
from tensorflow.keras.optimizers import Adam
import string
from keras.preprocessing import image as krs_image
# feature selection libraries
from sklearn.feature_selection import RFE
import argparse
import gc
from vit_keras import vit

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--image_model', type=str, default='efficientnet')
parser.add_argument('--num_layers', help="Number of fully connected layers", type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument("--finetune_only", action= "store_true", default=False, help="finetune")
parser.add_argument("--continue_training", action="store_true", default=False)

args = parser.parse_args()

color_gt = ['RGB-R', 'RGB-G','RGB-B','HSL-H','HSL-S','HSL-L']
object_gt = ['aeroplane','apple','backpack','banana','baseball bat','baseball glove','bear','bed','bench','bicycle',
'bird','boat','book','bottle','bowl','broccoli','bus','cake','car','carrot','cat','cell phone','chair',
'clock','cow','cup','diningtable','dog','donut','elephant','fire hydrant','fork','frisbee','giraffe',
'handbag','horse','hot dog','keyboard','kite','knife','laptop','microwave','motorbike','mouse','orange',
'oven','parking meter','person','pizza','pottedplant','refrigerator','remote','sandwich','scissors','sheep',
'sink','skateboard','skis','snowboard','sofa','spoon','sports ball','stop sign','suitcase','surfboard',
'teddy bear','tennis racket','tie','toilet','toothbrush','traffic light','train','truck','tvmonitor','umbrella',
'vase','wine glass','zebra']
scene_gt = ['Indoor/Outdoor','shopping and dining','workplace (office building, factory, lab, etc.)',
'home or hotel','transportation (vehicle interiors, stations, etc.)','sports and leisure','cultural (art, education, religion, millitary, law, politics, etc.)',
'water, ice, snow','mountains, hills, desert, sky','forest, field, jungle','man-made elements','transportation (roads, parking, bridges, boats, airports, etc.)',
'cultural or historical building/place (millitary, religious)','sports fields, parks, leisure spaces','industrial and construction','houses, cabins, gardens, and farms',
'commercial buildings, shops, markets, cities, and towns','airfield','airplane_cabin','airport_terminal','alcove','alley','amphitheater','amusement_arcade',
'amusement_park','apartment_building/outdoor','aquarium','aqueduct','arcade','arch','archaelogical_excavation','archive','arena/hockey','arena/performance',
'arena/rodeo','army_base','art_gallery','art_school','art_studio','artists_loft','assembly_line','athletic_field/outdoor','atrium/public','attic','auditorium',
'auto_factory','auto_showroom','badlands','bakery/shop','balcony/exterior','balcony/interior','ball_pit','ballroom','bamboo_forest','bank_vault','banquet_hall',
'bar','barn','barndoor','baseball_field','basement','basketball_court/indoor','bathroom','bazaar/indoor','bazaar/outdoor','beach','beach_house','beauty_salon',
'bedchamber','bedroom','beer_garden','beer_hall','berth','biology_laboratory','boardwalk','boat_deck','boathouse','bookstore','booth/indoor','botanical_garden',
'bow_window/indoor','bowling_alley','boxing_ring','bridge','bullring','burial_chamber','bus_interior','bus_station/indoor','butchers_shop','butte','cabin/outdoor',
'campsite','campus','canal/natural','canal/urban','candy_store','canyon','car_interior','carrousel','castle','catacomb','cemetery','chalet','chemistry_lab',
'childs_room','church/indoor','church/outdoor','classroom','clean_room','cliff','closet','clothing_store','coast','cockpit','coffee_shop','computer_room',
'conference_center','conference_room','construction_site','corn_field','corral','corridor','cottage','courthouse','courtyard','creek','crevasse','crosswalk',
'dam','delicatessen','department_store','desert/sand','desert/vegetation','desert_road','diner/outdoor','dining_hall','dining_room','discotheque','doorway/outdoor',
'dorm_room','downtown','dressing_room','driveway','drugstore','elevator/door','elevator_lobby','elevator_shaft','embassy','engine_room','entrance_hall',
'escalator/indoor','excavation','fabric_store','farm','fastfood_restaurant','field/cultivated','field/wild','field_road','fire_escape','fire_station','fishpond',
'flea_market/indoor','florist_shop/indoor','food_court','football_field','forest/broadleaf','forest_path','forest_road','formal_garden','fountain','galley',
'garage/indoor','garage/outdoor','gas_station','gazebo/exterior','general_store/indoor','general_store/outdoor','gift_shop','glacier','golf_course','greenhouse/indoor',
'greenhouse/outdoor','grotto','gymnasium/indoor','hangar/indoor','hangar/outdoor','harbor','hardware_store','hayfield','heliport','highway','home_office',
'home_theater','hospital','hospital_room','hot_spring','hotel/outdoor','hotel_room','house','hunting_lodge/outdoor','ice_cream_parlor','ice_floe','ice_shelf',
'ice_skating_rink/indoor','ice_skating_rink/outdoor','iceberg','igloo','industrial_area','inn/outdoor','islet','jacuzzi/indoor','jail_cell','japanese_garden',
'jewelry_shop','junkyard','kasbah','kennel/outdoor','kindergarden_classroom','kitchen','lagoon','lake/natural','landfill','landing_deck','laundromat','lawn',
'lecture_room','legislative_chamber','library/indoor','library/outdoor','lighthouse','living_room','loading_dock','lobby','lock_chamber','locker_room','mansion',
'manufactured_home','market/indoor','market/outdoor','marsh','martial_arts_gym','mausoleum','medina','mezzanine','moat/water','mosque/outdoor','motel','mountain',
'mountain_path','mountain_snowy','movie_theater/indoor','museum/indoor','museum/outdoor','music_studio','natural_history_museum','nursery','nursing_home','oast_house',
'ocean','office','office_building','office_cubicles','oilrig','operating_room','orchard','orchestra_pit','pagoda','palace','pantry','park','parking_garage/indoor',
'parking_garage/outdoor','parking_lot','pasture','patio','pavilion','pet_shop','pharmacy','phone_booth','physics_laboratory','picnic_area','pier','pizzeria','playground',
'playroom','plaza','pond','porch','promenade','pub/indoor','racecourse','raceway','raft','railroad_track','rainforest','reception','recreation_room','repair_shop',
'residential_neighborhood','restaurant','restaurant_kitchen','restaurant_patio','rice_paddy','river','rock_arch','roof_garden','rope_bridge','ruin','runway','sandbox',
'sauna','schoolhouse','science_museum','server_room','shed','shoe_shop','shopfront','shopping_mall/indoor','shower','ski_resort','ski_slope','sky','skyscraper','slum',
'snowfield','soccer_field','stable','stadium/baseball','stadium/football','stadium/soccer','stage/indoor','stage/outdoor','staircase','storage_room','street',
'subway_station/platform','supermarket','sushi_bar','swamp','swimming_hole','swimming_pool/indoor','swimming_pool/outdoor','synagogue/outdoor','television_room',
'television_studio','temple/asia','throne_room','ticket_booth','topiary_garden','tower','toyshop','train_interior','train_station/platform','tree_farm','tree_house',
'trench','tundra','underwater/ocean_deep','utility_room','valley','vegetable_garden','veterinarians_office','viaduct','village','vineyard','volcano',
'volleyball_court/outdoor','waiting_room','water_park','water_tower','waterfall','watering_hole','wave','wet_bar','wheat_field','wind_farm','windmill','yard',
'youth_hostel','zen_garden']


# In[3]:


# image features data in csv
train_x_data = pd.read_csv('../../data/train_x_rounded.csv', index_col ='iid')
test_x_data = pd.read_csv('../../data/test_x_rounded.csv',index_col ="iid")
valid_x_data = pd.read_csv('../../data/valid_x_rounded.csv',index_col ="iid")

# rejected_features = color_gt + scene_gt + ['boat', 'car_interior', 'forest, field, jungle', 'forest/broadleaf']
rejected_features = color_gt

train_x_data.drop(columns=rejected_features,inplace=True)
test_x_data.drop(columns=rejected_features,inplace=True)
valid_x_data.drop(columns=rejected_features,inplace=True)

trainAttrX = train_x_data

train_y_data = pd.read_csv('../../data/train_y.csv', index_col ='iid')
test_y_data = pd.read_csv('../../data/test_y.csv',index_col ="iid")
valid_y_data = pd.read_csv('../../data/valid_y.csv',index_col ="iid")

# image data 
train_file_list = glob.glob('../../train/*.jpg', recursive = True)
random.shuffle(train_file_list)

test_file_list = glob.glob('../../test/*.jpg', recursive = True)
random.shuffle(test_file_list)

valid_file_list = glob.glob('../../valid/*.jpg', recursive = True)
random.shuffle(valid_file_list)

train_file_list = [x for x in train_file_list if "(1)" not in x]
test_file_list = [x for x in test_file_list if "(1)" not in x]
valid_file_list = [x for x in valid_file_list if "(1)" not in x]

len(train_file_list)



len(test_file_list)


# In[11]:


len(valid_file_list)


# In[12]:


# data generator

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
        batch = {'img_input': [], 'csv_input': [], 'valence': [], 'arousal': [], 'dominance': []}
        for b in range(batch_size):
            if i == len(images_list):
                i = 0
                random.shuffle(images_list)
            # Read image from list and convert to array
            image_path = images_list[i]
            image_name = os.path.basename(image_path).replace('.JPG', '')
            #image = tf.keras.utils.load_img(image_path, target_size=(224, 224)) 
            image = krs_image.load_img(image_path, target_size=(224, 224)) #image height & width
            image = datagen.apply_transform(image, data_gen_args)
            image = krs_image.img_to_array(image)
            #image = tf.keras.utils.img_to_array(image)

            # Read data from csv using the name of current image
            csv_features = dataframe.loc[image_name, :]
            y = df_y.loc[image_name, :]
            val_train = y[0]
            aro_train = y[1]
            dom_train = y[2]

            #print(image_name)
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
        

        yield [batch['img_input']], [batch['valence'], batch['arousal'], batch['dominance']]

gen_train = my_gen(train_file_list, train_x_data, train_y_data, 32)
gen_test = my_gen(test_file_list, test_x_data, test_y_data, 32)
gen_valid = my_gen(valid_file_list, valid_x_data, valid_y_data, 32)


# In[13]:


def create_mlp(dim):
	# define our MLP network
	model = Sequential()
	model.add(Dense(512, input_dim=dim, activation="relu", name='csv'))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(128, activation="relu"))
	model.add(Dense(64, activation="relu"))
    
	return model


# # In[14]:


from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def create_resnet(width, height, depth):
    # inputs= Input(shape=(width, height, depth), name='img_input')

    res = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape = (width, height, depth),
    )

    for layer in res.layers:
        layer.trainable = False #false: freeze, true:train by own

    x = layers.BatchNormalization()(res.output)
    top_dropout_rate = args.dropout
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Flatten()(x)

    model = Model(res.input, x, name="ResNet")
    return model


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
    top_dropout_rate = args.dropout
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    x = layers.Flatten()(x)

    model = Model(res.input, x, name="ResNet")
    return model


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

    # x = layers.Flatten()(x)
    #activation

    model = Model(eff.input, x, name="EfficientNet")
    return model

def create_vit(width, height, depth):

    # inputs= Input(shape=(width, height, depth), name='img_input')

    trans = vit.vit_b32(
        image_size=224,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )

    trans.trainable = False

    for layer in trans.layers:
        layer.trainable = False

    # x = layers.GlobalAveragePooling2D(name="avg_pool")(eff.output)
    x = layers.BatchNormalization()(trans.output)
    top_dropout_rate = args.dropout
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    # x = layers.Flatten()(x)
    #activation

    model = Model(trans.input, x, name="ViT")
    return model

image_model = None
image_model_layers = []
if args.image_model == 'resnet':
    image_model = create_resnet(224, 224, 3)
    image_model_layers = image_model.layers
elif args.image_model == 'mobilenet':
    image_model = create_mobilenet(224, 224, 3)
    image_model_layers = image_model.layers
elif args.image_model == 'vit':
    image_model = create_vit(224, 224, 3)
    image_model_layers = image_model.layers
else:
    print("creating efficientnet")
    image_model = create_efficientNet(224, 224, 3)
    image_model_layers = image_model.layers

from tensorflow.keras.optimizers import Adam

x = None

if args.num_layers == 1:
    x = Dense(1024, activation="relu",name='csv_img')(image_model.output)

elif args.num_layers == 2:
    x = Dense(1024, activation="relu",name='csv_img')(image_model.output)
    x = Dense(512, activation="relu")(x)

valence_output = Dense(units=1, activation='linear', name='valence_output')(x)
arousal_output = Dense(units=1, activation='linear', name='arousal_output')(x)
dominance_output = Dense(units=1, activation='linear', name='dominance_output')(x)

from tensorflow.keras.models import load_model

if args.continue_training:
    model = load_model(f"./trained_models/{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-dropout")
else:
    model = Model(inputs=[image_model.input], outputs=[valence_output, arousal_output, dominance_output])

opt = Adam(learning_rate=0.001)

model.compile(optimizer=opt, run_eagerly=True,
              loss={'valence_output': 'mse', 'arousal_output': 'mse', 'dominance_output': 'mse'},
              metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})


# # In[17]:


model.summary()


# # In[19]:


callbacks = [
    CSVLogger(
        f"{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-dropout_results.csv", separator=',', append=True
    ),
    EarlyStopping(monitor='val_loss', patience=7, mode='min', min_delta=0.0001),
    ModelCheckpoint(f"./trained_models/{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-dropout", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
]

batch_size = 32
trainidx = len(train_file_list)
testidx = len(test_file_list)
valididx = len(valid_file_list)


# # In[20]:

if not args.finetune_only:
    history = model.fit(gen_train, epochs=1000, steps_per_epoch=trainidx/batch_size, validation_data=gen_valid, validation_steps=valididx/batch_size, callbacks=callbacks, batch_size=32)

# model.save('b4_mixedinput_fc_1024_512.h5')


# # ### Fine-tuning

# # In[18]:

model = None
gc.collect()
model = load_model(f"./trained_models/{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-dropout")


# In[19]:


# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(model.layers))


# Freeze all the layers before the `fine_tune_at` layer

if args.image_model == 'efficientnet':
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

elif args.image_model == 'vit':
    for layer in model.layers:
        layer.trainable = True

else:
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in model.layers[fine_tune_at:]:
        layer.trainable = True


# In[20]:

model.summary()


# In[21]:

opt = Adam(learning_rate=0.0001)

model.compile(optimizer=opt, run_eagerly=True,
              loss={'valence_output': 'mse', 'arousal_output': 'mse', 'dominance_output': 'mse'},
              metrics={'valence_output': 'mae', 'arousal_output': 'mae', 'dominance_output': 'mae'})


callbacks = [
    CSVLogger(
        f"{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-finetuned_results.csv", separator=',', append=True
    ),
    EarlyStopping(monitor='val_loss', patience=5, mode='min', min_delta=0.0001),
    ModelCheckpoint(f"./trained_models/{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-dropout-finetuned", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
]

batch_size = 32
trainidx = len(train_file_list)
testidx = len(test_file_list)
valididx = len(valid_file_list)

history = model.fit(gen_train, epochs=100, steps_per_epoch=trainidx/batch_size, validation_data=gen_valid, validation_steps=valididx/batch_size, callbacks=callbacks, batch_size=32)

def plot_train_history(history):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].plot(history.history['dominance_output_mae'], label='Dominance Train MAE')
    axes[0].plot(history.history['val_dominance_output_mae'], label='Dominance Val MAE')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(history.history['arousal_output_mae'], label='Arousal Train MAE')
    axes[1].plot(history.history['val_arousal_output_mae'], label='Arousal Val MAE')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

    axes[2].plot(history.history['valence_output_mae'], label='Valence Train MAE')
    axes[2].plot(history.history['val_valence_output_mae'], label='Valence Val MAE')
    axes[2].set_xlabel('Epochs')
    axes[2].legend()

    axes[3].plot(history.history['loss'], label='Training loss')
    axes[3].plot(history.history['val_loss'], label='Validation loss')
    axes[3].set_xlabel('Epochs')
    axes[3].legend()

plot_train_history(history)

from tensorflow.keras.models import load_model

model = None
gc.collect()
model = load_model(f"./trained_models/{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-dropout-finetuned")

val_results = model.evaluate(gen_valid, steps=valididx/batch_size, return_dict=True)

import json
with open(f"{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-val_results.json", 'w') as f:
    json.dump(val_results, f)


test_results = model.evaluate(gen_test, steps=testidx/batch_size, return_dict=True)

import json
with open(f"{args.image_model}-{args.num_layers}-layer-{str(args.dropout).replace('.', '')}-test_results.json", 'w') as f:
    json.dump(test_results, f)

