import numpy as np
import math 


gt = np.load('C:/Users/sonip/Documents/Computer Vision/salad/groundTruth1.npy')
rawtiff = np.load('C:/Users/sonip/Documents/Computer Vision/salad/rawTiff2.npy')

#155 is the length of the gt and rawtiff arrays. 
#array holds the accuracy values for the images at each index
accuracy = np.zeros(155)

#print(len(gt[0][3][:]))    equals 616

temp = 0
count = 0


#gt[0][0][:] = 1
  
for i in range(155):
    for j in range(616):    
        if gt[0][i][j] == rawtiff[0][i][j]:
            temp +=1
        count += 1

    #print(count)
        divide = temp/count
        rms = math.sqrt(divide) * 100
        a = round(rms,2) 
        accuracy[i] = a
             
print(accuracy[3])
    
   

  

    

