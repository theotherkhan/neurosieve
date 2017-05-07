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


#################################### Getting traces, ground truth files ####################



# extracts and returns the x,y,z coordinates from swc files for labeled neuron voxels

def getCoordinates (inputPath):
    
    print ('            Getting coordinates...')
    
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
    
    print ('Constructing ground truths for each coordinate set...')
    
    groundTruths = []

    ## PUT THE FILEPATH FOR THE TRACE DIRECTORY W/ .txt FILES HERE (note: the order of the .txt files in 
    ## this directory must correspond exactly to the order of the images in the .tiff directory for this to work 
    ## ... make sure all the files match up!)
    
    y, fluff = getAllCoordinateSets('/Users/Hasan/Desktop/Workspace/CV/Final/tracings/*.txt')
    coordSets = y
  
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


#################################### MAIN / DRIVER ################################### 

# loads multiple tiff image stacks into an array of numpy arrays 

# PUT TIF DIRECTORY PATHWAY HERE: 

pathname = '/Users/Hasan/Desktop/Workspace/CV/Final/tracings/*.tif'
  
allImagePaths = glob.glob(pathname)

allImages = []
imageShapes = np.ndarray((2,3))

for i in range (0, len(allImagePaths)):
    allImages.append(np.asarray(io.imread(allImagePaths[i])))
    imageShapes[i] = np.asarray(io.imread(allImagePaths[i])).shape
    imageShapes[i+1] = np.asarray(io.imread(allImagePaths[i])).shape


# Creates ground truth files:
gt = get_groundTruths()

print (" \nExtraction Done! A list of images (stored as numpy arrays) is available as allImages[]. \nA list of all ground truths (also stored as numpy arrays) is available as gt[]. \n")

print ("SWC1:", gt[0] , "done")
print ("SWC2:", gt[1] , "done")

'''
length = len(gt[0])  #161
gt0 = gt[0]
gt1 = gt[1]
length2 = len(gt0[0][4][:])   #664



######## count number of one's #########
ones = 0
for i in range(length):
    for j in range(length2):

        if gt1[0][i][j] != 0:
            ones +=1
print("number of 1s is: ", ones)

###
'''
gtc1 = 0
gtc2 = 0

matched = 0
total = 0
sub = 0


for i in range (0, len(gt[0])):
    for k in range (0, len(gt[0][i])):
        for j in range (0, len(gt[0][i][k])):
            #print(gt[0][i][k][j])
            if (gt[0][i][k][j] != 0 and gt[1][i][k][j] != 0):
                matched+=1
            if (gt[0][i][k][j] == 0 and gt[1][i][k][j] == 0):
                sub+=1        
            total+=1

'''
for i in range (0, len(gt[1])):
    for k in range (0, len(gt[1][i])):
        for j in range (0, len(gt[1][i][k])):
            #print(gt[0][i][k][j])
            if (gt[1][i][k][j] != 0):
                gtc2+=1




print ("Counts:", gtc1, gtc2)





for i in range (0, len(gt[0])):
    for k in range (0, len(gt[0][i])):
        for j in range (0, len(gt[0][i][k])):

            for x in range (0, len(gt[1])):
                for y in range (0, len(gt[1][x])):
                    for z in range (0, len(gt[1][x][y])):

                        if gt[0][i][k][j] == gt[1][x][y][z]:
                            matched += 1
                        total += 1

'''

acc = float(( float(matched) / float(total-sub)) * 100.0)
print("matched, total, accuracy:", matched, total, acc)



