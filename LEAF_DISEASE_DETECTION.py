#!/usr/bin/env python
# coding: utf-8

# In[5]:


## importing libraries


# In[6]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[7]:


#Data Preprocessing


# In[8]:


#training image preprocessing


# In[10]:


training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128,128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[11]:


#validation image preprocessing


# In[12]:


validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128,128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[13]:


training_set


# In[14]:


for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


# In[15]:


#Bulding  modeling


# In[16]:


from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential


# In[17]:


model = Sequential()


# In[18]:


## Bulding convolution layer


# In[19]:


model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[20]:


model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[21]:


model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[22]:


model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[23]:


model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[24]:


model.add(Dropout(0.25))


# In[25]:


model.add(Flatten())


# In[26]:


model.add(Dense(units=1500,activation='relu'))


# In[27]:


model.add(Dropout(0.4))


# In[34]:


#output layer
model.add(Dense(units=38,activation='softmax'))


# In[35]:


## Compiling model


# In[36]:


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[31]:


model.summary()


# In[ ]:




