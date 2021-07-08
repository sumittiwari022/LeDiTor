#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


tf.__version__


# In[3]:


_URL = "http://127.0.0.1:81/pv/PlantVillage.zip"


# In[5]:


zip_file = tf.keras.utils.get_file(origin=_URL, 
                                   fname="PlantVillage.zip",
                                   extract=True)


# In[6]:


base_dir = os.path.join(os.path.dirname(zip_file), 'PlantVillage\\train')
base_dir2 = os.path.join(os.path.dirname(zip_file), 'PlantVillage\\train')


# In[7]:


IMAGE_SIZE = 256
BATCH_SIZE = 32


# In[8]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2)


# In[9]:


train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')


# In[10]:


val_generator = datagen.flow_from_directory(
    base_dir2,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')


# In[11]:


for image_batch, label_batch in train_generator:
  break
image_batch.shape, label_batch.shape


# In[12]:


print (train_generator.class_indices)


# In[13]:


labels = '\n'.join(sorted(train_generator.class_indices.keys()))


# In[14]:


with open('new_labels.txt', 'w') as f:
  f.write(labels)


# In[15]:


IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


# In[16]:


# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')


# In[17]:


base_model.trainable = False


# In[18]:


model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(47, activation='softmax')
])


# In[19]:


model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[20]:


model.summary()


# In[ ]:





# In[21]:


print('Number of trainable variables = {}'.format(len(model.trainable_variables)))


# In[22]:


epochs = 101


# In[23]:


history = model.fit_generator(train_generator, 
                    epochs=epochs, 
                    validation_data=val_generator)


# In[24]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


# In[25]:


loss = history.history['loss']
val_loss = history.history['val_loss']


# In[26]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')


# In[27]:


plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[28]:


base_model.trainable = True


# In[29]:


print("Number of layers in the base model: ", len(base_model.layers))


# In[30]:


fine_tune_at = 101


# In[31]:


for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# In[32]:


model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(1e-5),
              metrics=['accuracy'])


# In[33]:


model.summary()


# In[34]:


print('Number of trainable variables = {}'.format(len(model.trainable_variables)))


# In[35]:


history_fine = model.fit_generator(train_generator, 
                         epochs=11,
                         validation_data=val_generator)


# In[36]:


saved_model_dir = '.'
tf.saved_model.save(model, saved_model_dir)


# In[37]:


converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()


# In[38]:


with open('new_model.tflite', 'wb') as f:
  f.write(tflite_model)


# In[39]:


acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']


# In[40]:


loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']


# In[41]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')


# In[42]:


plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




