
import cv_toolkit as toolkit
import numpy as np

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
    #随机旋转
    rotation_range=5,
    #随机放缩
    zoom_range=0.06,
    #横向平移
    width_shift_range=0.1,
    #纵向平移
    height_shift_range=0.1,
    #斜切
    shear_range=0.05
)
buf=ImageDataGenerator()
#model=Functional()
HWDB_input=layers.Input(shape=(dimH,dimW,1),name='input')
#第一层
ConA= layers.Conv2D(filters=40,kernel_size=(4,3),padding='valid',activation='relu',name='ConA')(HWDB_input)#H,W#padding=same的时候填充的是0（黑色） 之后解决这个问题就可以用same
ConC= layers.Conv2D(filters=40,kernel_size=(12,9),padding='valid',activation='relu',name='ConC')(HWDB_input)#H,W
ConE= layers.Conv2D(filters=40,kernel_size=(24,18),padding='valid',activation='relu',name='ConE')(HWDB_input)#H,W
#第二层
DropA=layers.Dropout(0.1)(ConA)
DropC=layers.Dropout(0.1)(ConC)
DropE=layers.Dropout(0.1)(ConE)
#第三层
PoolA=layers.MaxPooling2D(pool_size=(3,3),name='PoolA')(DropA) #H/3,W/3
PoolC=layers.MaxPooling2D(pool_size=(3,3),name='PoolC')(DropC) #H/3,W/3
PoolE=layers.MaxPooling2D(pool_size=(3,3),name='PoolE')(DropE) #H/3,W/3
#第4到7层
ConA2= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConA2')(PoolA)
PoolA2=layers.MaxPooling2D(pool_size=(2,2),name='PoolA2')(ConA2)
ConA3= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConA3')(PoolA2)
PoolA3=layers.MaxPooling2D(pool_size=(2,2),name='PoolA3')(ConA3)
ConA4= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConA4')(PoolA3)
PoolA4=layers.MaxPooling2D(pool_size=(2,2),name='PoolA4')(ConA4)


ConC2= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConC2')(PoolC)
PoolC2=layers.MaxPooling2D(pool_size=(2,2),name='PoolC2')(ConC2)
ConC3= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConC3')(PoolC2)
PoolC3=layers.MaxPooling2D(pool_size=(2,2),name='PoolC3')(ConC3)
ConC4= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConC4')(PoolC3)
PoolC4=layers.MaxPooling2D(pool_size=(2,2),name='PoolC4')(ConC4)

ConE2= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConE2')(PoolE)
PoolE2=layers.MaxPooling2D(pool_size=(2,2),name='PoolE2')(ConE2)
ConE3= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConE3')(PoolE2)
PoolE3=layers.MaxPooling2D(pool_size=(2,2),name='PoolE3')(ConE3)
ConE4= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConE4')(PoolE3)
PoolE4=layers.MaxPooling2D(pool_size=(2,2),name='PoolE4')(ConE4)

#GAP层
GAPA=layers.GlobalAveragePooling2D()(PoolA)
GAPA2=layers.GlobalAveragePooling2D()(PoolA2)
GAPA3=layers.GlobalAveragePooling2D()(PoolA3)
GAPA4=layers.GlobalAveragePooling2D()(PoolA4)

GAPC=layers.GlobalAveragePooling2D()(PoolC)
GAPC2=layers.GlobalAveragePooling2D()(PoolC2)
GAPC3=layers.GlobalAveragePooling2D()(PoolC3)
GAPC4=layers.GlobalAveragePooling2D()(PoolC4)

GAPE=layers.GlobalAveragePooling2D()(PoolE)
GAPE2=layers.GlobalAveragePooling2D()(PoolE2)
GAPE3=layers.GlobalAveragePooling2D()(PoolE3)
GAPE4=layers.GlobalAveragePooling2D()(PoolE4)

#合并层
Merge=layers.concatenate([GAPA,GAPA2,GAPA3,GAPA4,GAPC,GAPC2,GAPC3,GAPC4,GAPE,GAPE2,GAPE3,GAPE4])
#隐藏层
Hid=layers.Dense(units=256,activation='relu',name='Hid2')(Merge)
#out
HWDB_output=layers.Dense(units=maxy,activation='softmax',name='output')(Hid)
#model
model=models.Model(inputs=HWDB_input,outputs=HWDB_output)

print(model.summary())
all_utils.plot_model(model,to_file='model.png')

# earlyStop=EarlyStopping(monitor='val_loss',patience=3)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkPoint1=ModelCheckpoint('D:\\MachineLearning\\BYSJ\\checkpoint1',monitor='val_accuracy',mode='max',save_best_only=True)


print('开始训练')
# his=model.fit(XTrain,YTrain,validation_split=0.1, epochs=60, batch_size=1024)
his=model.fit_generator(data_aug.flow(xtr,ytr,batch_size=36),steps_per_epoch=xtr.shape[0]//36,epochs=120,callbacks=[checkPoint1],validation_data=data_aug.flow(xte,yte,batch_size=36))
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

#读checkpoint
loadModel=models.load_model('D:\\MachineLearning\\BYSJ\\checkpoint1')
loss2,acc2= loadModel.evaluate(xte,yte)


checkPoint2=ModelCheckpoint('D:\\MachineLearning\\BYSJ\\checkpoint2',monitor='val_accuracy',mode='max',save_best_only=True)
loadModel.fit_generator(buf.flow(xtr,ytr,batch_size=36),steps_per_epoch=xtr.shape[0]//36,epochs=50,callbacks=[checkPoint2],validation_data=buf.flow(xte,yte,batch_size=36))

loadModel2=models.load_model('D:\\MachineLearning\\BYSJ\\checkpoint2')
loss3,acc3= loadModel2.evaluate(xte,yte)

print(loss1,acc1)
print(loss2,acc2)
print(loss3,acc3)


# import cv_toolkit as toolkit
# import numpy as np

# from keras.utils import all_utils
# from keras import models
# import keras.layers as layers
# import keras.regularizers as rglers
# from keras.callbacks import ModelCheckpoint

# from keras.preprocessing.image import ImageDataGenerator

# import matplotlib.pyplot as plt
# import gc

# gc.collect()
# charNum=40
# dataNum=80
# print('run script')
# (xtr,ytr,xte,yte),(dimH,dimW)= toolkit.pngs2npdataset('D:\\MachineLearning\\BYSJ\\Gnt1.1TrainPart2',charNum,dataNum)
# xtr=xtr/255.0
# xte=xte/255.0
# gc.collect()

# print(dimH,dimW)
# plt.imshow(xtr[56],cmap='gray')
# print(xtr[56][20])
# plt.show()
# print(xtr.shape,xte.shape)


# print('处理Y数据')
# print(ytr)
# ytr,yte,dic,maxy=toolkit.rand_to_unique(ytr,yte)

# ytr=all_utils.to_categorical(ytr,maxy)
# yte=all_utils.to_categorical(yte,maxy)
# print(ytr.shape,yte.shape)

# print(maxy,'dims in dataset')

# gc.collect()
# print('正在搭建网络')

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
# buf=ImageDataGenerator()
# #model=Functional()
# HWDB_input=layers.Input(shape=(dimH,dimW,1),name='input')
# #第一层
# ConA= layers.Conv2D(filters=16,kernel_size=(4,3),padding='valid',activation='relu',name='ConA')(HWDB_input)#H,W#padding=same的时候填充的是0（黑色） 之后解决这个问题就可以用same
# ConC= layers.Conv2D(filters=24,kernel_size=(12,9),padding='valid',activation='relu',name='ConC')(HWDB_input)#H,W
# ConE= layers.Conv2D(filters=36,kernel_size=(24,18),padding='valid',activation='relu',name='ConE')(HWDB_input)#H,W
# #第二层
# BNA=layers.BatchNormalization()(ConA)
# BNC=layers.BatchNormalization()(ConC)
# BNE=layers.BatchNormalization()(ConE)
# #第三层
# PoolA=layers.MaxPooling2D(pool_size=(3,3),name='PoolA')(BNA) #H/3,W/3
# PoolC=layers.MaxPooling2D(pool_size=(3,3),name='PoolC')(BNC) #H/3,W/3
# PoolE=layers.MaxPooling2D(pool_size=(3,3),name='PoolE')(BNE) #H/3,W/3
# #第4到7层
# ConA2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConA2')(PoolA)
# BNA2=layers.BatchNormalization()(ConA2)
# PoolA2=layers.MaxPooling2D(pool_size=(2,2),name='PoolA2')(BNA2)
# ConA3= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConA3')(PoolA2)
# BNA3=layers.BatchNormalization()(ConA3)
# PoolA3=layers.MaxPooling2D(pool_size=(2,2),name='PoolA3')(BNA3)
# ConA4= layers.Conv2D(filters=30,kernel_size=(3,3),padding='valid',activation='relu',name='ConA4')(PoolA3)
# BNA4=layers.BatchNormalization()(ConA4)
# PoolA4=layers.MaxPooling2D(pool_size=(2,2),name='PoolA4')(BNA4)


# ConC2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConC2')(PoolC)
# BNC2=layers.BatchNormalization()(ConC2)
# PoolC2=layers.MaxPooling2D(pool_size=(2,2),name='PoolC2')(BNC2)
# ConC3= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConC3')(PoolC2)
# BNC3=layers.BatchNormalization()(ConC3)
# PoolC3=layers.MaxPooling2D(pool_size=(2,2),name='PoolC3')(BNC3)
# ConC4= layers.Conv2D(filters=30,kernel_size=(3,3),padding='valid',activation='relu',name='ConC4')(PoolC3)
# BNC4=layers.BatchNormalization()(ConC4)
# PoolC4=layers.MaxPooling2D(pool_size=(2,2),name='PoolC4')(BNC4)

# ConE2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConE2')(PoolE)
# BNE2=layers.BatchNormalization()(ConE2)
# PoolE2=layers.MaxPooling2D(pool_size=(2,2),name='PoolE2')(BNE2)
# ConE3= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConE3')(PoolE2)
# BNE3=layers.BatchNormalization()(ConE3)
# PoolE3=layers.MaxPooling2D(pool_size=(2,2),name='PoolE3')(BNE3)
# ConE4= layers.Conv2D(filters=30,kernel_size=(3,3),padding='valid',activation='relu',name='ConE4')(PoolE3)
# BNE4=layers.BatchNormalization()(ConE4)
# PoolE4=layers.MaxPooling2D(pool_size=(2,2),name='PoolE4')(BNE4)

# #平铺层
# FlatA=layers.Flatten()(PoolA)
# FlatA2=layers.Flatten()(BNA2)
# FlatA3=layers.Flatten()(BNA3)
# FlatA4=layers.Flatten()(BNA4)

# FlatC=layers.Flatten()(PoolC)
# FlatC2=layers.Flatten()(BNC2)
# FlatC3=layers.Flatten()(BNC3)
# FlatC4=layers.Flatten()(BNC4)

# FlatE=layers.Flatten()(PoolE)
# FlatE2=layers.Flatten()(BNE2)
# FlatE3=layers.Flatten()(BNE3)
# FlatE4=layers.Flatten()(BNE4)


# #权值管理层
# HidA=layers.Dense(units=128,activation='relu',name='HidA')(FlatA)
# HidA2=layers.Dense(units=128,activation='relu',name='HidA2')(FlatA2)
# HidA3=layers.Dense(units=128,activation='relu',name='HidA3')(FlatA3)
# HidA4=layers.Dense(units=256,activation='relu',name='HidA4')(FlatA4)

# HidC=layers.Dense(units=128,activation='relu',name='HidC')(FlatC)
# HidC2=layers.Dense(units=128,activation='relu',name='HidC2')(FlatC2)
# HidC3=layers.Dense(units=128,activation='relu',name='HidC3')(FlatC3)
# HidC4=layers.Dense(units=256,activation='relu',name='HidC4')(FlatC4)

# HidE=layers.Dense(units=128,activation='relu',name='HidE')(FlatE)
# HidE2=layers.Dense(units=128,activation='relu',name='HidE2')(FlatE2)
# HidE3=layers.Dense(units=128,activation='relu',name='HidE3')(FlatE3)
# HidE4=layers.Dense(units=256,activation='relu',name='HidE4')(FlatE4)

# #合并层
# Merge=layers.concatenate([HidA,HidA2,HidA3,HidA4,HidC,HidC2,HidC3,HidC4,HidE,HidE2,HidE3,HidE4])
# #隐藏层
# Hid1=layers.Dense(units=512,activation='relu',name='Hid1')(Merge)
# Hid2=layers.Dense(units=256,activation='relu',name='Hid2')(Hid1)
# #out
# HWDB_output=layers.Dense(units=maxy,activation='softmax',name='output')(Hid2)
# #model
# model=models.Model(inputs=HWDB_input,outputs=HWDB_output)

# print(model.summary())
# all_utils.plot_model(model,to_file='model.png')

# # earlyStop=EarlyStopping(monitor='val_loss',patience=3)
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# checkPoint1=ModelCheckpoint('D:\\MachineLearning\\BYSJ\\checkpoint1',monitor='val_accuracy',mode='max',save_best_only=True)


# print('开始训练')
# # his=model.fit(XTrain,YTrain,validation_split=0.1, epochs=60, batch_size=1024)
# his=model.fit_generator(data_aug.flow(xtr,ytr,batch_size=80),steps_per_epoch=xtr.shape[0]//80,epochs=100,callbacks=[checkPoint1],validation_data=data_aug.flow(xte,yte,batch_size=80))
# loss1,acc1= model.evaluate(xte,yte)
# model.save('.')


# Loss=his.history['loss']
# valLoss=his.history['val_loss']
# Acc=his.history['accuracy']
# valAcc=his.history['val_accuracy']
# epos=range(1,len(Acc)+1)
# print(dic)

# plt.plot(epos,Acc,'red',label='accuracy')
# plt.plot(epos,valAcc,'brown',label='val_accuracy')
# plt.plot(epos,Loss,'green',label='loss')
# plt.plot(epos,valLoss,'blue',label='val_loss')
# plt.legend()
# plt.show()
# gc.collect()

# #读checkpoint
# loadModel=models.load_model('D:\\MachineLearning\\BYSJ\\checkpoint1')
# loss2,acc2= loadModel.evaluate(xte,yte)


# checkPoint2=ModelCheckpoint('D:\\MachineLearning\\BYSJ\\checkpoint2',monitor='val_accuracy',mode='max',save_best_only=True)
# loadModel.fit_generator(buf.flow(xtr,ytr,batch_size=40),steps_per_epoch=xtr.shape[0]//40,epochs=20,callbacks=[checkPoint2],validation_data=buf.flow(xte,yte,batch_size=40))

# loadModel2=models.load_model('D:\\MachineLearning\\BYSJ\\checkpoint2')
# loss3,acc3= loadModel2.evaluate(xte,yte)

# print(loss1,acc1)
# print(loss2,acc2)
# print(loss3,acc3)

