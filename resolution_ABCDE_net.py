
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
ConA= layers.Conv2D(filters=24,kernel_size=(4,3),padding='valid',activation='relu',name='ConA')(HWDB_input)#H,W#padding=same的时候填充的是0（黑色） 之后解决这个问题就可以用same
ConB= layers.Conv2D(filters=24,kernel_size=(7,5),padding='valid',activation='relu',name='ConB')(HWDB_input)#H,W
ConC= layers.Conv2D(filters=24,kernel_size=(9,7),padding='valid',activation='relu',name='ConC')(HWDB_input)#H,W
ConD= layers.Conv2D(filters=24,kernel_size=(11,9),padding='valid',activation='relu',name='ConD')(HWDB_input)#H,W
ConE= layers.Conv2D(filters=24,kernel_size=(19,15),padding='valid',activation='relu',name='ConE')(HWDB_input)#H,W
#第二层
PoolA=layers.MaxPooling2D(pool_size=(3,3),name='PoolA')(ConA) #H/3,W/3
PoolB=layers.MaxPooling2D(pool_size=(3,3),name='PoolB')(ConB) #H/3,W/3
PoolC=layers.MaxPooling2D(pool_size=(3,3),name='PoolC')(ConC) #H/3,W/3
PoolD=layers.MaxPooling2D(pool_size=(3,3),name='PoolD')(ConD) #H/3,W/3
PoolE=layers.MaxPooling2D(pool_size=(3,3),name='PoolE')(ConE) #H/3,W/3
#第三层
DropOutA=layers.Dropout(0.1)(PoolA)
DropOutB=layers.Dropout(0.1)(PoolB)
DropOutC=layers.Dropout(0.1)(PoolC)
DropOutD=layers.Dropout(0.1)(PoolD)
DropOutE=layers.Dropout(0.1)(PoolE)
#第4到7层
ConA2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConA2')(DropOutA)
PoolA2=layers.MaxPooling2D(pool_size=(2,2),name='PoolA2')(ConA2)
DropOutA2=layers.Dropout(0.1)(PoolA2)
ConA3= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConA3')(PoolA2)
PoolA3=layers.MaxPooling2D(pool_size=(2,2),name='PoolA3')(ConA3)
DropOutA3=layers.Dropout(0.1)(PoolA3)
ConA4= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConA4')(PoolA3)
PoolA4=layers.MaxPooling2D(pool_size=(2,2),name='PoolA4')(ConA4)
DropOutA4=layers.Dropout(0.1)(PoolA4)
ConB2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConB2')(DropOutB)
PoolB2=layers.MaxPooling2D(pool_size=(2,2),name='PoolB2')(ConB2)
DropOutB2=layers.Dropout(0.1)(PoolB2)
ConB3= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConB3')(PoolB2)
PoolB3=layers.MaxPooling2D(pool_size=(2,2),name='PoolB3')(ConB3)
DropOutB3=layers.Dropout(0.1)(PoolB3)
ConC2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConC2')(DropOutC)
PoolC2=layers.MaxPooling2D(pool_size=(2,2),name='PoolC2')(ConC2)
DropOutC2=layers.Dropout(0.1)(PoolC2)
# ConC3= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConC3')(PoolC2)
# PoolC3=layers.MaxPooling2D(pool_size=(2,2),name='PoolC3')(ConC3)
# DropOutC3=layers.Dropout(0.1)(PoolC3)
# ConD2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConD2')(DropOutD)
# PoolD2=layers.MaxPooling2D(pool_size=(2,2),name='PoolD2')(ConD2)
# DropOutD2=layers.Dropout(0.1)(PoolD2)
# ConD3= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConD3')(PoolD2)
# PoolD3=layers.MaxPooling2D(pool_size=(2,2),name='PoolD3')(ConD3)
# DropOutD3=layers.Dropout(0.1)(PoolD3)
# ConE2= layers.Conv2D(filters=20,kernel_size=(3,3),padding='valid',activation='relu',name='ConE2')(DropOutE)
# PoolE2=layers.MaxPooling2D(pool_size=(2,2),name='PoolE2')(ConE2)
# DropOuE2=layers.Dropout(0.1)(PoolE2)
# ConE3= layers.Conv2D(filters=40,kernel_size=(3,3),padding='valid',activation='relu',name='ConE3')(PoolE2)
# PoolE3=layers.MaxPooling2D(pool_size=(2,2),name='PoolE3')(ConE3)
# DropOutE3=layers.Dropout(0.1)(PoolE3)
#平铺层
FlatA=layers.Flatten()(PoolA)
FlatA2=layers.Flatten()(DropOutA2)
FlatA3=layers.Flatten()(DropOutA3)
FlatA4=layers.Flatten()(DropOutA4)
FlatB=layers.Flatten()(PoolB)
FlatB2=layers.Flatten()(DropOutB2)
FlatB3=layers.Flatten()(DropOutB3)
FlatC=layers.Flatten()(DropOutC)
FlatC2=layers.Flatten()(DropOutC2)
FlatD1=layers.Flatten()(DropOutD)
# FlatD2=layers.Flatten()(DropOutD2)
FlatE1=layers.Flatten()(DropOutE)
# FlatE2=layers.Flatten()(DropOutE2)
#权值管理层
HidA=layers.Dense(units=256,activation='relu',name='HidA')(FlatA)
HidA2=layers.Dense(units=256,activation='relu',name='HidA2')(FlatA2)
HidA3=layers.Dense(units=256,activation='relu',name='HidA3')(FlatA3)
HidA4=layers.Dense(units=256,activation='relu',name='HidA4')(FlatA4)
HidB=layers.Dense(units=256,activation='relu',name='HidB')(FlatB)
HidB2=layers.Dense(units=256,activation='relu',name='HidB2')(FlatB2)
HidB3=layers.Dense(units=256,activation='relu',name='HidB3')(FlatB3)
HidC=layers.Dense(units=256,activation='relu',name='HidC')(FlatC)
HidC2=layers.Dense(units=256,activation='relu',name='HidC2')(FlatC2)
HidD1=layers.Dense(units=256,activation='relu',name='HidD1')(FlatD1)
HidE1=layers.Dense(units=256,activation='relu',name='HidE1')(FlatE1)
#合并层
Merge=layers.concatenate([HidA,HidA2,HidA3,HidA4,HidB,HidB2,HidB3,HidC,HidC2,HidD1,HidE1])
#隐藏层
Hid1=layers.Dense(units=512,activation='relu',name='Hid1')(Merge)
Hid2=layers.Dense(units=256,activation='tanh',name='Hid2')(Hid1)
#out
HWDB_output=layers.Dense(units=maxy,activation='softmax',name='output')(Hid2)
#model
model=models.Model(inputs=HWDB_input,outputs=HWDB_output)

print(model.summary())
all_utils.plot_model(model,to_file='model.png')

# earlyStop=EarlyStopping(monitor='val_loss',patience=3)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

checkPoint1=ModelCheckpoint('D:\\MachineLearning\\BYSJ\\checkpoint1',monitor='val_accuracy',mode='max',save_best_only=True)


print('开始训练')
# his=model.fit(XTrain,YTrain,validation_split=0.1, epochs=60, batch_size=1024)
his=model.fit_generator(data_aug.flow(xtr,ytr,batch_size=64),steps_per_epoch=xtr.shape[0]//64,epochs=150,callbacks=[checkPoint1],validation_data=data_aug.flow(xte,yte,batch_size=64))
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
loadModel.fit_generator(buf.flow(xtr,ytr,batch_size=64),steps_per_epoch=xtr.shape[0]//64,epochs=20,callbacks=[checkPoint2],validation_data=buf.flow(xte,yte,batch_size=64))

loadModel2=models.load_model('D:\\MachineLearning\\BYSJ\\checkpoint1')
loss3,acc3= loadModel.evaluate(xte,yte)

print(loss1,acc1)
print(loss2,acc2)
print(loss3,acc3)

