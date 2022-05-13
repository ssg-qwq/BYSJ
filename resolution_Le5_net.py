
import cv_toolkit as toolkit
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.utils import all_utils
from keras import models
import keras.layers as layers
import keras.regularizers as rglers
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import gc

gc.collect()
charNum=200
dataNum=80
print('run script')
(xtr,ytr,xte,yte),(dimH,dimW)= toolkit.pngs2npdataset('D:\\MachineLearning\\BYSJ\\Gnt1.1TrainPart2',charNum,dataNum)
xtr=xtr/255.0
xte=xte/255.0
gc.collect()

print(dimH,dimW)
plt.imshow(xtr[56],cmap='gray')
print(xtr[56][20])
plt.show()
print(xtr.shape,xte.shape)


print('处理Y数据')
print(ytr)
ytr,yte,dic,maxy=toolkit.rand_to_unique(ytr,yte)

ytr=all_utils.to_categorical(ytr,maxy)
yte=all_utils.to_categorical(yte,maxy)
print(ytr.shape,yte.shape)

print(maxy,'dims in dataset')

gc.collect()
print('正在搭建网络')

#数据增强
data_aug=ImageDataGenerator(
)
buf=ImageDataGenerator()
#第一层
model=Sequential()
model.add(Conv2D(filters=6,kernel_size=(5,5),strides=(1,1),input_shape=(dimH,dimW,1),padding='valid',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=120,activation='relu'))
model.add(Dense(units=84,activation='relu'))
model.add(Dense(units=maxy,activation='softmax'))

print(model.summary())
all_utils.plot_model(model,to_file='model.png')

# earlyStop=EarlyStopping(monitor='val_loss',patience=3)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


print('开始训练')
his=model.fit_generator(data_aug.flow(xtr,ytr,batch_size=36),steps_per_epoch=xtr.shape[0]//36,epochs=100,validation_data=data_aug.flow(xte,yte,batch_size=36))
loss1,acc1= model.evaluate(xte,yte)
model.save('.')


Loss=his.history['loss']
valLoss=his.history['val_loss']
Acc=his.history['accuracy']
valAcc=his.history['val_accuracy']
epos=range(1,len(Acc)+1)
print(dic)

plt.plot(epos,Acc,'red',label='accuracy')
plt.plot(epos,valAcc,'brown',label='val_accuracy')
plt.plot(epos,Loss,'green',label='loss')
plt.plot(epos,valLoss,'blue',label='val_loss')
plt.legend()
plt.show()
gc.collect()

print(loss1,acc1)
