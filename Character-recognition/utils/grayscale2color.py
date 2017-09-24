import numpy as np
import glob
from PIL import Image 

imageList = glob.glob('../letters/normal/*.gif')
READ_PATH = '../letters/normal/'
WRITE_PATH = '../letters/color/'
cnt = 0
for image in imageList:
	image_file = Image.open(image) # open colour image
	targetName = image.replace(' ','-')
	targetName = image.replace('.gif','.png')
	targetName = targetName.replace("normal","color")
	image_file = image_file.convert('RGB') 
	image_file.save(targetName)
	cnt+=1
	if cnt%1000==0:
		print cnt