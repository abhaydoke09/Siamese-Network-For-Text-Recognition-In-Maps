f = open('../font_recognition/font_labels.txt','rb')
lines = f.readlines()
f.close()

class_to_label = {}
for line in lines:
	class_name,label = line.split(' ')
	class_to_label[class_name] = label

f = open('../font_recognition/train.txt','rb')
train_images = f.readlines()
f.close()
f = open('../font_recognition/font_train.txt','wb')
for image in train_images:
	class_name = image.split('/')[-1].split('--')[0]
	image_name = image.split(' ')[0]
	f.write(image_name+' '+class_to_label[class_name])
f.close()

f = open('../font_recognition/test.txt','rb')
train_images = f.readlines()
f.close()
f = open('../font_recognition/font_test.txt','wb')
for image in train_images:
	class_name = image.split('/')[-1].split('--')[0]
	image_name = image.split(' ')[0]
	f.write(image_name+' '+class_to_label[class_name])
f.close()


f = open('../font_recognition/validation.txt','rb')
train_images = f.readlines()
f.close()
f = open('../font_recognition/font_validation.txt','wb')
for image in train_images:
	class_name = image.split('/')[-1].split('--')[0]
	image_name = image.split(' ')[0]
	f.write(image_name+' '+class_to_label[class_name])
f.close()



