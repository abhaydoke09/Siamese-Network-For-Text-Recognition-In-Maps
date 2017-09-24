import glob
files = glob.glob('../letters/new_color/*.png')

f = open('../image-labels.txt','wb')

cnt = 0
labelDictionary = {}

for i in xrange(26):
	labelDictionary[chr(65+i)] = cnt
	labelDictionary[chr(97+i)] = cnt
	cnt+=1
for i in xrange(10):
	labelDictionary[chr(48+i)] = cnt
	cnt+=1


for filename in files:
	imagename = filename.split('/')[-1]
	c = imagename.split('--')[1].split('.')[0]
	f.write(imagename+' '+str(labelDictionary[c])+'\n')

f.close()
