import numpy as np
import math 


gt = np.load('C:/Users/sonip/Documents/Computer Vision/salad/groundTruth1.npy')
rawtiff = np.load('C:/Users/sonip/Documents/Computer Vision/salad/rawTiff2.npy')


length = len(gt)     #155 
accuracy = np.zeros(length)  # will hold the accuracy values for the images at each index
length2 = len(gt[0][3][:])  #616

#s = 0
#for x in range(616):
#    if rawtiff[0][0][x] == 1:
#        s += 1        
#print("number of 1 = ", s)
      
temp = 0
count = 0

#calculate accuracy 
for i in range(length):
    for j in range(length2):    
        if gt[0][i][j] == rawtiff[0][i][j]:
            temp +=1
        count += 1

        a = (temp/count) * 100
        a2 = round(a,2) 
        accuracy[i] = a2
                
#print(accuracy[3])

#we now have an array that holds the accuracy values of the ground truth images compared to the raw tiff images.         
    
   

  

    

