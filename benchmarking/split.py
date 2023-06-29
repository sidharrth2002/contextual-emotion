import os
import random
import shutil

parent_folder = '/media/workstation/2TB/Sidharrth/Benchmarking/folds/'
folder = '/media/workstation/2TB/Sidharrth/Benchmarking/all_images/'
# split a folder of images into 5 folds randomly
images = os.listdir(folder)

for i in range(5):
    os.mkdir(parent_folder + 'fold_' + str(i))

random.shuffle(images)
for i in range(5):
    for j in range(len(images)//5):
        shutil.copy(folder+images[i*len(images)//5+j], parent_folder+'fold_'+str(i)+'/')