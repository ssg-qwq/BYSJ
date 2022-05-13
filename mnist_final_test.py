from email.mime import image
import imp
from itertools import accumulate
from keras import models
import keras
from keras.datasets import mnist
from cProfile import label
from doctest import OutputChecker
from heapq import merge
from operator import mod
from statistics import mode
from tokenize import Name
from unicodedata import name
import dataset_reader as rd
import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.utils import all_utils
from keras import models
import keras.layers as layers
import keras.regularizers as rglers
from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import gc
"""
-———————— ABCDE  NET ————————-
本来是做Chinese HandWriting DB
Recognizing用的自制网络模型
我将其简化后使用mnist数据集测试
效果不错
-————————————————————————————-
"""



# #数据增强
# data_aug=ImageDataGenerator(
#     #随机旋转
#     rotation_range=5,
#     #随机放缩
#     zoom_range=0.06,
#     #横向平移
#     width_shift_range=0.1,
#     #纵向平移
#     height_shift_range=0.1,
#     #斜切
#     shear_range=0.05
# )

(XTrain,YTrain),(XTest,YTest)=mnist.load_data()


print(XTrain.shape,YTrain.shape,XTest.shape,YTest.shape)
print(XTrain[0][0])

plt.imshow(XTrain[0],cmap='gray')
plt.show()

XTrain=XTrain.reshape(60000,28,28,1)/255.0
XTest=XTest.reshape(10000,28,28,1)/255.0
# for batch in data_aug.flow(XTrain,YTrain,batch_size=1):
#     plt.imshow(batch[0][0],cmap='gray')
#     plt.show()
#     break
#数转为onehot
YTrain=all_utils.to_categorical(YTrain,10)
YTest=all_utils.to_categorical(YTest,10)

print(YTrain)


gc.collect()
print('正在搭建网络')


# #model=Functional()
# HWDB_input=layers.Input(shape=(28,28,1),name='input')
# ConA= layers.Conv2D(filters=40,kernel_size=(2,2),padding='valid',activation='relu',name='ConA')(HWDB_input)#H,W#padding=same的时候填充的是0（黑色） 之后解决这个问题就可以用same
# ConC= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConC')(HWDB_input)#H,W
# ConE= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConE')(HWDB_input)#H,W
# #第二层
# DropA=layers.Dropout(0.1)(ConA)
# DropC=layers.Dropout(0.1)(ConC)
# DropE=layers.Dropout(0.1)(ConE)
# #第三层
# PoolA=layers.MaxPooling2D(pool_size=(2,2),name='PoolA')(DropA) #H/3,W/3
# PoolC=layers.MaxPooling2D(pool_size=(2,2),name='PoolC')(DropC) #H/3,W/3
# PoolE=layers.MaxPooling2D(pool_size=(2,2),name='PoolE')(DropE) #H/3,W/3
# #第4到7层
# ConA2= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConA2')(PoolA)
# PoolA2=layers.MaxPooling2D(pool_size=(2,2),name='PoolA2')(ConA2)
# ConA3= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConA3')(PoolA2)
# PoolA3=layers.MaxPooling2D(pool_size=(2,2),name='PoolA3')(ConA3)
# ConA4= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConA4')(PoolA3)
# PoolA4=layers.MaxPooling2D(pool_size=(2,2),name='PoolA4')(ConA4)


# ConC2= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConC2')(PoolC)
# PoolC2=layers.MaxPooling2D(pool_size=(2,2),name='PoolC2')(ConC2)
# ConC3= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConC3')(PoolC2)
# PoolC3=layers.MaxPooling2D(pool_size=(2,2),name='PoolC3')(ConC3)
# ConC4= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConC4')(PoolC3)
# PoolC4=layers.MaxPooling2D(pool_size=(2,2),name='PoolC4')(ConC4)

# ConE2= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConE2')(PoolE)
# PoolE2=layers.MaxPooling2D(pool_size=(2,2),name='PoolE2')(ConE2)
# ConE3= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConE3')(PoolE2)
# PoolE3=layers.MaxPooling2D(pool_size=(2,2),name='PoolE3')(ConE3)
# ConE4= layers.Conv2D(filters=20,kernel_size=(2,2),padding='same',activation='relu',name='ConE4')(PoolE3)
# PoolE4=layers.MaxPooling2D(pool_size=(2,2),name='PoolE4')(ConE4)

# #GAP层
# GAPA=layers.GlobalAveragePooling2D()(PoolA)
# GAPA2=layers.GlobalAveragePooling2D()(PoolA2)
# GAPA3=layers.GlobalAveragePooling2D()(PoolA3)
# GAPA4=layers.GlobalAveragePooling2D()(PoolA4)

# GAPC=layers.GlobalAveragePooling2D()(PoolC)
# GAPC2=layers.GlobalAveragePooling2D()(PoolC2)
# GAPC3=layers.GlobalAveragePooling2D()(PoolC3)
# GAPC4=layers.GlobalAveragePooling2D()(PoolC4)

# GAPE=layers.GlobalAveragePooling2D()(PoolE)
# GAPE2=layers.GlobalAveragePooling2D()(PoolE2)
# GAPE3=layers.GlobalAveragePooling2D()(PoolE3)
# GAPE4=layers.GlobalAveragePooling2D()(PoolE4)

# #合并层
# Merge=layers.concatenate([GAPA,GAPA2,GAPA3,GAPA4,GAPC,GAPC2,GAPC3,GAPC4,GAPE,GAPE2,GAPE3,GAPE4])
# #隐藏层
# Hid=layers.Dense(units=256,activation='relu',name='Hid2')(Merge)
# #out
# HWDB_output=layers.Dense(units=10,activation='softmax',name='output')(Hid)
# #model
# model=models.Model(inputs=HWDB_input,outputs=HWDB_output)

# print(model.summary())
# all_utils.plot_model(model,to_file='model.png')

# # earlyStop=EarlyStopping(monitor='val_loss',patience=3)
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# checkPoint1=ModelCheckpoint('D:\\MachineLearning\\BYSJ\\checkpoint1',monitor='val_accuracy',mode='max',save_best_only=True)
# print('开始训练')
# his=model.fit(XTrain,YTrain,validation_split=0.1, epochs=100,callbacks=[checkPoint1], batch_size=1024)
# # his=model.fit_generator(data_aug.flow(XTrain,YTrain,batch_size=1024),steps_per_epoch=XTrain.shape[0]//1024,epochs=15,validation_data=data_aug.flow(XTest,YTest,batch_size=1024))
# model.fit(XTrain,YTrain, epochs=1, batch_size=1024)
# loss,acc= model.evaluate(XTest,YTest)
# model.save('.')

# loadModel=models.load_model('D:\\MachineLearning\\BYSJ\\checkpoint1')
# loss2,acc2= loadModel.evaluate(XTest,YTest)


#数据增强
data_aug=ImageDataGenerator(
)
buf=ImageDataGenerator()
#第一层
model=Sequential()
model.add(layers.Conv2D(filters=6,kernel_size=(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=120,activation='relu'))
model.add(layers.Dense(units=84,activation='relu'))
model.add(layers.Dense(units=10,activation='softmax'))

print(model.summary())
all_utils.plot_model(model,to_file='model.png')

# earlyStop=EarlyStopping(monitor='val_loss',patience=3)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


print('开始训练')
his=model.fit_generator(data_aug.flow(XTrain,YTrain,batch_size=1024),steps_per_epoch=XTrain.shape[0]//1024,epochs=100,validation_data=data_aug.flow(XTest,YTest,batch_size=1024))
loss1,acc1= model.evaluate(XTest,YTest)
model.save('.')


Loss=his.history['loss']
valLoss=his.history['val_loss']
Acc=his.history['accuracy']
valAcc=his.history['val_accuracy']
epos=range(1,len(Acc)+1)
plt.plot(epos,Acc,'red',label='accuracy')
plt.plot(epos,valAcc,'brown',label='val_accuracy')
plt.plot(epos,Loss,'green',label='loss')
plt.plot(epos,valLoss,'blue',label='val_loss')
plt.legend()
plt.show()
gc.collect()

print(loss1,acc1)