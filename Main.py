import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

TRAIN_DIR="train/"
TEST_DIR="test/"
VALIDATION_DIR="validation/"
size=200

print("Creating training dataset...")
train=ImageDataGenerator(rescale=1/255)
train_dataset=train.flow_from_directory(TRAIN_DIR,target_size=(size,size),batch_size=3,class_mode='binary')
validation_dataset=train.flow_from_directory(VALIDATION_DIR,target_size=(size,size),batch_size=3,class_mode='binary')

# print("Saving training dataset into training_dataset.npy\n")
# np.save("training_dataset.npy",train_dataset,allow_pickle=True)

# print("Saving validation dataset into validation_dataset.npy\n")
# np.save("validation_dataset.npy",validation_dataset,allow_pickle=True)

print("Dataset class indices: \n")
print(train_dataset.class_indices)
# print(train_dataset.classes)


print("\n\nCreating training model...\n ")
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(size,size,3)),
									tf.keras.layers.MaxPool2D(2,2),
									tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
									tf.keras.layers.MaxPool2D(2,2),
									tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
									tf.keras.layers.MaxPool2D(2,2),
									tf.keras.layers.Flatten(),
									tf.keras.layers.Dense(512,activation='relu'),
									tf.keras.layers.Dense(1,activation='sigmoid')
								])


print("\n\nCompiling training model...\n")
model.compile(loss='binary_crossentropy',
			optimizer=RMSprop(lr=0.001),
			metrics=['accuracy'])


print("\n\nFit to model.. \n")
model_fit=model.fit(train_dataset,steps_per_epoch=5,
					epochs=30,
					validation_data=validation_dataset)


print("\n\nGoing through the test data to test whether medical plant : \n")
for i in os.listdir(TEST_DIR):
	img=image.load_img(TEST_DIR+'//'+ i,target_size=(size,size))

	X=image.img_to_array(img)
	X=np.expand_dims(X,axis=0)
	images=np.vstack([X])
	val=model.predict(images)
	if val==0:
		print("\n\n"+i+" is a Medical Plant.\n")
		plt.imshow(img)
		plt.show()
	else:
		print("\n\n"+i+" is not a Medical Plant.\n")

plt.plot(model_fit.history['loss'], label='LOSS (training data)')
plt.plot(model_fit.history['accuracy'], label='ACCURACY (validation data)')
plt.title('Automatic Medical Plant Identification')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()