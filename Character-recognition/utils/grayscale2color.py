import numpy as np
import glob
from PIL import Image 

imageList = glob.glob('../letters/color/*.png')

cnt = 0
for image in imageList:
	image_file = Image.open(image) # open colour image
	targetName = image.replace(' ','-')
	#targetName = targetName.replace("normal","color")
	#image_file = image_file.convert('RGB') 
	#print targetName
	image_file.save('../letters/new/'+targetName.split('/')[-1])
	cnt+=1
	if cnt%1000==0:
		print cnt