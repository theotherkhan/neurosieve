import numpy as np
import math 


gt = np.load('C:/Users/sonip/Documents/Computer Vision/salad/groundTruth1.npy')
rawtiff = np.load('C:/Users/sonip/Documents/Computer Vision/salad/rawTiff2.npy')

#155 = length of the gt and rawtiff arrays. 
#the following array holds the accuracy values for the images at each index
accuracy = np.zeros(155)

#print(len(gt[0][3][:]))    equals 616

#s = 0
#for x in range(616):
#    if rawtiff[0][0][x] == 1:
#        s += 1        
#print("number of 1 = ", s)
      
temp = 0
count = 0

#gt[0][0][:] = 1

#calculate accuracy based on root mean square formula
for i in range(155):
    for j in range(616):    
        if gt[0][i][j] == rawtiff[0][i][j]:
            temp +=1
        count += 1

        divide = temp/count
        rms = math.sqrt(divide) * 100
        a = round(rms,2) 
        accuracy[i] = a
                
#print(accuracy[3])

#we now have an array that holds the accuracy values of the ground truth images compared to the raw tiff images.         
    
   

  

    

