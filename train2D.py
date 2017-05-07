
############################################################################################## Part II: Architecture #############################

import keras
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras import backend as K


from skimage.transform import resize

y_ax = 64
x_ax = 64

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

    train = allImages[0:5] #a list of 5, each element is a npy array
    trainTruth = gt[0:5]   # both truth and train are the same size; checked

    test = allImages[6:11]   
    testTruth = gt[6:11]

    trainS = []
    trainTruthS = []
    testS = []
    testTruthS = []


    
    print("\nExtracting one depth layer from each image...")

    for i in range (0, len(train)):
        
        depth = len(train[i])   

        train[i] = train[i][(depth/2)][:][:]
        trainTruth[i] = trainTruth[i][(depth/2)]
        test[i] = test[i][(depth/2)]
        testTruth[i] = testTruth[i][(depth/2)]

        # train, trainTruth, test, testTruth should now contain 5 2D slices each

    print (train[0].shape)
    print ('')
    print (train[0][0].shape)
    print (train[0][0][0].shape)

    '''

    train = resize_imgs(train[0])


    print ("Hello")
    
    for i in range (0, len(train)):

        train = resize_imgs(train[i])
        #trainTruth = resize_imgs(trainTruth[i])
        #test = resize_imgs(test[i])
        #testTruth = resize_imgs(testTruth[i])
    

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
    '''

import time
start_time = time.time()
train_and_predict()
print("--- %s seconds ---" % (time.time() - start_time))
    



