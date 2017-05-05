from skimage.transform import resize
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.optimizers import Adam
from keras import backend as K

K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')

x_ax = 32
y_ax = 512
z_ax = 512

def dice_coef(y_true, y_pred):
    y_true_flat = K.flatten(y_true) #turn into 1d vector
    y_pred_flat = K.flatten(y_pred)
    
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + 1.) / (K.sum(y_true_flat)+K.sum(y_pred_flat) + 1.)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def make_model():
    inputs = Input((x_ax, y_ax, z_ax, 1))
    conv1 = Conv3D(32, (3, 3, 1), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    
    conv2 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(conv3)
    #comment
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    
    conv4 = Conv3D(256, (3, 3, 1), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 1), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4)

    conv5 = Conv3D(512, (3, 3, 1), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 1), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 1))(conv5), conv4], axis=4)
    conv6 = Conv3D(256, (3, 3, 1), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 1), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 1))(conv6), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 1), activation='relu', padding='same')(conv7)
    #comment
    #up8 = concatenate([UpSampling3D(size=(2, 2, 1))(conv3), conv2], axis=4)
    up8 = concatenate([UpSampling3D(size=(2, 2, 1))(conv7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 1), activation='relu', padding='same')(conv8)
    
    up9 = concatenate([UpSampling3D(size=(2, 2, 1))(conv8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 1), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 1), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def resize_imgs(imgs):
    print('resizing...')
    print("shape of imgs before resize is ", imgs.shape)
    imgs = np.ndarray((imgs.shape[0], x_ax, y_ax, z_ax), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs[i] = resize(imgs[i], (x_ax, y_ax, z_ax))#, preserve_range=True)

    imgs = imgs[..., np.newaxis]
    print("shape after adding axis: ", imgs.shape)
    return imgs

def train():
    gt = np.load('testCat.npy')
    train = np.load('rawTiff1.npy')
    test = np.load('rawTiff2.npy')
    #get only 2 layers
    gt = gt[50:82]
    print("HEY I'M GONNA PRINT SOME SHAPES ", gt.shape)
    train = train[50:82]
    test = test[50:82]
    
    gt = gt[None, ...]
    train = train[None, ...]
    test = test[None, ...]

    gt = resize_imgs(gt)
    train = resize_imgs(train)
    test = resize_imgs(test)
    

    model = make_model()
    model.fit(train, gt, batch_size=1, epochs=2)

if __name__ == "__main__":
    import time
    start_time = time.time()
    train()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
