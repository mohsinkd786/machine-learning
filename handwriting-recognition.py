import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

#  analyze sample image
import pylab as pl

digits = load_digits()

# print('##########')
# print(digits.images[2])

#print('##############')
pl.gray()
#pl.matshow(digits.images[2])
# 0 is black
# 255 is white
# matrix - every number is a gray scale pixel (0-255)

# print(digits.images[1])
#pl.show()

images_labels = list(zip(digits.images,digits.target))
# print('$$$$')
# print(len(images_labels))

# plt.figure(figsize=(5,5))
#plt.show()
# render the digits & their corresponding labels first 10
for index,(image,label) in enumerate(images_labels[:10]):
    plt.subplot(2,5,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('%i' % label)

# plt.show()

# define variables
n_samples = len(digits.images)
print(n_samples)

# print('$$$ Samples')
# gray scale number matrix
A = digits.images.reshape(n_samples,-1)

# numbers / digits 
B = digits.target
print(A.shape)
print(B.shape)
# digit
# print(B)

# test split 20 % for test split
A_training,A_test,B_training,B_test = train_test_split(A,B,test_size = 0.20)

# using Random for Classifier
classifier = RandomForestClassifier()

# train model
model = classifier.fit(A_training,B_training)

# print(model)

# validation score
score = classifier.score(A_test,B_test)
# % age of accuracy
print(score)

# run sample 
# apply prediction

i = 7
pl.gray()
pl.matshow(digits.images[i])
pl.show()

# apply prediction on a specific number
prediction = classifier.predict(A_test[i].reshape(1,-1))
#print(prediction)

