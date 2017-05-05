
############################################################################################# Part I: Data Preprocessing #############################
# 
# This part of the program extracts all necessary information fromt the gold166 database of neurons. 
# Returns an organized directory of tiff files, their numpy representations,
# and their respective numpy ground truth (trace) files.


from __future__ import print_function

from PIL import Image
import numpy as np 
import PIL
from skimage import io
from numpy import genfromtxt

import glob


#################################### Noisy neuron numpys ################################### 

# loads multiple tiff image stacks into an array of numpy arrays 

pathname = '/Users/Hasan/Desktop/Workspace/CV/Final/tiffDirectory/*.tif'
  
allImagePaths = glob.glob(pathname)

allImages = []
imageShapes = np.ndarray((20,3))

for i in range (0, len(allImagePaths)):
    allImages.append(np.asarray(io.imread(allImagePaths[i])))
    imageShapes[i] = np.asarray(io.imread(allImagePaths[i])).shape



#################################### Getting traces, ground truth files ####################



# extracts and returns the x,y,z coordinates from swc files for labeled neuron voxels

def getCoordinates (inputPath):
    
    print ('            Getting some coordinates...')
    
    input_file = open(inputPath, 'r')
   
    discountLength = 0
    m = 0
    
    num_lines = sum(1 for line in open(inputPath)) - discountLength
    coord = np.ndarray(shape=(num_lines,3))
    
    for line in input_file:
        
        #print ('Line:', line)
        
        while (line[0] == '#'): 
            #print ('                   cutting fluff')
            line = next(input_file)
            discountLength+=1


        coordinates = line.strip().split(' ')
        coord[m][0] = float(coordinates[4])
        coord[m][1] = float(coordinates[3])
        coord[m][2] = float(coordinates[2])
        m+=1

        if (line[0] == ' '):
            return np.around(coord)

    return np.around(coord), discountLength
    input_file.close()




# gets all the x,y,z coordinate sets from all the swc files

def getAllCoordinateSets (txtPaths):
    
    print ('    Getting all coordinates...')

    #txtPaths = '/Users/Hasan/Desktop/Workspace/CV/Final/tracesDirectory2/*.txt'

    allTxtPaths = glob.glob(txtPaths)

    coordinateSets = []

    for i in range (0, len(allTxtPaths)):
        inputPath = allTxtPaths[i]
        c, fluffCount = getCoordinates(inputPath)
        coordinateSets.append(c)
        #print (i)
    
    
    print (' ')
    print ('\nAll coordinates have been grabbed!')
    
    return coordinateSets, fluffCount


# creates ground truth trace files for each image using the extracted coordinate sets

def get_groundTruths():
    
    print ('Constructing ground truths...')
    
    groundTruths = []
    
    y, fluff = getAllCoordinateSets('/Users/Hasan/Desktop/Workspace/CV/Final/tracesDirectory2/*.txt')
    coordSets = y
  
    #print (' \nConstructing ground truths for each coordinate set... \n')

    counter = 1
    
    for i in range (0, len(coordSets)):     
        #print ('Current image shape:', imageShapes[i])
        count = 0
        currGroundTruth = np.zeros((imageShapes[i][0], imageShapes[i][1], imageShapes[i][2]))
        
        for k in range (0, (len(coordSets[i]) - fluff)):

            z = coordSets[i][k][0]
            y = coordSets[i][k][1]
            x = coordSets[i][k][2]
            
            #print ('z, y, x. ', z, y, x, '. Image dimension: ', imageShapes[i])
            count+=1
            
            currGroundTruth[z][y][x] = 1
       
    
        #print ('     Adding ground truth w/ ', count, ' neuron pixels!')
        counter+=1
        groundTruths.append(currGroundTruth)
    

    return groundTruths


# Creates ground truth files and saves them as numpy array file

gt = get_groundTruths()

print (" \nExtraction Done!\n")


############################################################################################## Part II: Architecture #############################

import keras
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import backend as K


from skimage.transform import resize

z_ax = 20
y_ax = 512
x_ax = 512

print('-'*70)
print (" \nWelcome to Part II: Architecture! \n ")


def dice_coef(y_true, y_pred):
    y_true_flat = K.flatten(y_true) #turn into 1d vector
    y_pred_flat = K.flatten(y_pred)
    
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + 1.) / (K.sum(y_true_flat)+K.sum(y_pred_flat) + 1.)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def make_model():

    print (" Making model\n")
    
    inputs = Input((y_ax, x_ax, 1))


    print (" Conv1...\n")

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    print (" Conv2...\n")

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    print (" Conv3...\n")

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    print (" Conv4...\n")

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)


    print (" Conv5...\n")

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    print (" Conv10...\n")

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def resize_imgs(imgs):
    
    print('\nresizing...')
    print("shape of imgs before resize is ", imgs.shape)

    imgs_p = np.ndarray((imgs.shape[0], y_ax, x_ax), dtype=np.uint8)
    
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (y_ax, x_ax), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]

    print("shape after adding axis: ", imgs_p.shape)

    return imgs_p


################################### Create and run model #############################


def train_and_predict():

    train = allImages[0] 
    trainTruth = gt[0]     # both truth and train are the same size; checked

    test = allImages[1]   
    testTruth = gt[1]

    #get only 2 layers

    print("Extracting one depth layer from each image...")

    print ( "The depth of the train, test image", len(train[0]), len(test[0]))

    trainTruth = trainTruth[50:51]
    train = train[50:51]
    test = test[50:51]
    testTruth = testTruth[50:51]

    print ("Shapes for train, trainTruth: ", train.shape, trainTruth.shape)

    
    train = resize_imgs(train)
    trainTruth = resize_imgs(trainTruth)
    test = resize_imgs(test)
    testTruth = resize_imgs(testTruth)

    train = train.astype('float32')
    trainTruth = trainTruth.astype('float32')
    test = test.astype('float32')
    testTruth = testTruth.astype('float32')
    
    ############## Extra stuff!

    
    mean = np.mean(train)  # mean for data centering
    std = np.std(train)  # std for data normalization

    train -= mean
    train /= std

    trainTruth /= 255.0  # scale masks to [0, 1]

    ##############

    model = make_model()

    print ("\nFitting...\n")

    model.fit(train, trainTruth, batch_size=1, epochs=1, validation_data=(test, testTruth))
    
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print ("\nDone! Score:", score)


import time
start_time = time.time()
train_and_predict()
print("--- %s seconds ---" % (time.time() - start_time))
    



