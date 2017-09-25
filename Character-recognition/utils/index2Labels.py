def indexToLabel():
    labelDictionary = {}
    cnt = 0
    for i in xrange(26):
      labelDictionary[cnt] = chr(65+i)
      cnt+=1
    for i in xrange(10):
      labelDictionary[cnt] = chr(48+i)
      cnt+=1
    return labelDictionary

f = open('predictions.txt','rb')
lines = f.readlines()
f.close()

labels = indexToLabel()
f = open('new_predictions.txt','wb')
for line in lines:
	line = line.replace('\n','')
	label, index = line.split(',')
	f.write(str(label) + ',' + str(labels[int(index)]) + '\n')
f.close()