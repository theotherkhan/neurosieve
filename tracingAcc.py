################ code to determine accuracy of APP2 tracing when compared to gold standard tracing ##############################
################ what is needed: the directory pathway of the tif file that was traced, and the corresponding .txt files for the gold standard trace and the app2 trace of that tif image #########

from __future__ import print_function
import numpy as np 
from skimage import io
import glob


################ Getting traces, ground truth files ####################


# extracts and returns the x,y,z coordinates from swc files for labeled neuron voxels

def getCoordinates (inputPath):
    
    print ('            Getting coordinates...')
    
    input_file = open(inputPath, 'r')
   
    discountLength = 0
    m = 0
    
    num_lines = sum(1 for line in open(inputPath)) - discountLength
    coord = np.ndarray(shape=(num_lines,3))
    
    for line in input_file:
                
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
        count = 0
        currGroundTruth = np.zeros((imageShapes[i][0], imageShapes[i][1], imageShapes[i][2]))
        
        for k in range (0, (len(coordSets[i]) - fluff)):

            z = coordSets[i][k][0]
            y = coordSets[i][k][1]
            x = coordSets[i][k][2]
            
            count+=1
            
            currGroundTruth[z][y][x] = 1
       
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

#print ("SWC1:", gt[0] , "done")
#print ("SWC2:", gt[1] , "done")


###################### DETERMINE ACCURACY OF APP2 TRACE BASED ON THE CREATED NUMPY ARRAYS #######################################
matched = 0
total = 0
sub = 0

########## algorithm to compare the gold standard numpy array to the app2 traced numpy array ###########
for i in range (0, len(gt[0])):
    for k in range (0, len(gt[0][i])):
        for j in range (0, len(gt[0][i][k])):
            if (gt[0][i][k][j] != 0 and gt[1][i][k][j] != 0):
                matched+=1
            if (gt[0][i][k][j] == 0 and gt[1][i][k][j] == 0):
                sub+=1        
            total+=1


#subtrct sub from total (we don't want to consider the empty space in the images in our accuracy calculation)
acc = float(( float(matched) / float(total-sub)) * 100.0)
print("accuracy:", acc)



