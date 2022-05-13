
from asyncore import read
from pickletools import uint8
import struct
import os
from sys import path
from webbrowser import get
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gc
import cv2 as cv

def gnt2png(path):
    """
        returns(xtr,ytr),(xte,yte),(dimX,dimY),dataCount
        maxCharNum.default:reads all chars
    """
    fileList=os.listdir(path) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    #归一化数据
    maxWid=0
    maxHei=0
    #[[img],tag]
    dataSetX=[]
    # dataSetXTe=[]
    dataSetY=[]
    for file in fileList:
        if file.endswith('.gnt'):
            with open(path+"\\"+file,"rb") as f:
                print('正在读取',file)
                readCharNum=0
                for i in range(200):
                    f.read(6)
                    width=struct.unpack('<h', bytes(f.read(2)))
                    height=struct.unpack('<h',bytes(f.read(2)))
                    f.read(width[0]*height[0])
                    
                while f.read(4):
                    imgBytes = []
                    tag=bytes(f.read(2)).decode('gbk','ignore')
                    # tagstr=bytes(f.read(2))
                    # tag= struct.unpack('H'*len(tagstr)//2,bytearray)[0]<<8+struct.unpack('H'*len(tagstr)//2,bytearray) [1]
                    width=struct.unpack('<h', bytes(f.read(2)))
                    height=struct.unpack('<h',bytes(f.read(2)))
                    if width[0]>maxWid:maxWid=width[0]
                    if height[0]>maxHei:maxHei=height[0]
                    for i in range(width[0]*height[0]):
                        #ord函数将bytestring转换为byte
                        imgBytes.append(ord(f.read(1)))
                    dataSetX.append(255-np.array(imgBytes).reshape(height[0],width[0]))
                    dataSetY.append(tag)


                    
    print(maxHei,maxWid)
    print(dataSetY)
    shapeFile=os.path.join(path,'dataset.shape')
    with open(shapeFile,'w') as f:
        f.writelines(str(maxHei)+'\n')
        f.writelines(str(maxWid))
        

    print('shape normalizing...')
    #归一化
    dLen=len(dataSetX)
    for i in range(dLen):
        print('processing',i,'in',dLen)
        hei,wid= dataSetX[i].shape
        hu=(maxHei-hei)//2
        hd=(maxHei-hei)//2+(maxHei-hei)%2
        wl=(maxWid-wid)//2
        wr=(maxWid-wid)//2+(maxWid-wid)%2



        thisPath=path+'\\'+dataSetY[i]
        if not(os.path.exists(thisPath)):os.makedirs(thisPath)
        thisFile=os.path.join(thisPath,dataSetY[i]+'_'+str(i)+'.png')
        cv.imencode('.png',np.pad(dataSetX[i],((hu,hd),(wl,wr)),'constant',constant_values=(0,0)))[1].tofile(thisFile)


# path='D:\\MachineLearning\\BYSJ\\Gnt1.1TrainPart2'
# get_datas(path)

# gc.collect()



def pngs2npdataset(path,charNum,dataNum):
    """
    -——————pngs_to_np.array(number,hei,wid,1)——————-
    path:filePath where gnt2png
    charNum:select first (charnum) chars
    dataNum:num of datas per char
    returns:(TrainsX,TrainsY,TestX,TestY),(hei,wid)
    -——————————————————————————————————————————————-
    """
    maxHei=0
    maxWid=0
    fList=os.listdir(path)
    for member in fList:
        if member.endswith('.shape'):
            with open(path+'\\'+member,'r') as f:
                maxHei=int(f.readline())
                maxWid=int(f.readline())
    print('环境下数据形状',maxHei,maxWid)
    if maxHei==0:raise Exception(print('根目录没有.shape文件存在'))
    TrainX=np.zeros(((dataNum*4//5)*charNum,maxHei,maxWid,1))
    TestX=np.zeros(((dataNum-dataNum*4//5)*charNum,maxHei,maxWid,1))
    TrainY=[]
    TestY=[]
    iTr=0
    iTe=0
    isRead=0
    for member in fList:
        thisPath=path+'\\'+member
        if os.path.isdir(thisPath):
            print('正在读取',member)
            dList=os.listdir(thisPath)
            if len(dList)<dataNum:continue
            else:isRead+=1
            if(isRead>charNum):break
            j=0
            for png in dList:
                if png.endswith('.png'):
                    imgPath=thisPath+'\\'+png
                    bmp=cv.imdecode(np.fromfile(imgPath,dtype=np.uint8),-1)
                    if(j<dataNum*4//5):
                        TrainX[iTr,:,:,0]=bmp
                        TrainY.append(member)
                        iTr+=1
                    else:
                        TestX[iTe,:,:,0]=bmp
                        TestY.append(member)
                        iTe+=1
                    j+=1
                    print('第',str(j),'个数据')
                    if j>=dataNum:break
    return (TrainX,TrainY,TestX,TestY),(maxHei,maxWid)
                    
# gc.collect()
# (a,b,c,d),(e,f)=imgs_to_npdataset('D:\\MachineLearning\\BYSJ\\Gnt1.1TrainPart2',20,80)
# print(a.shape)
# print(len(b))
# print(c.shape)
# print(len(d))
# print(e)
# print(f)

# print(b[222])
# plt.imshow(a[222,:,:,0])
# plt.show()



def rand_to_unique(ytr,yte):
    """
    dic:dict(keys:chinese words,values:unique numbers like 0,1,2)
    outy:y eles write in unique numbers
    """
    dic=dict()
    trLen=len(ytr)
    teLen=len(yte)
    outytr=np.ones(trLen)
    outyte=np.ones(teLen)
    maxy=-1
    for i in range(trLen):
        if ytr[i] in dic.keys():
            outytr[i]=dic[ytr[i]]
        else:
            maxy+=1
            outytr[i]=maxy
            dic.update({ytr[i]:maxy})
    for i in range(teLen):
        if yte[i] in dic.keys():
            outyte[i]=dic[yte[i]]
        else:
            maxy+=1
            outyte[i]=maxy
            dic.update({yte[i]:maxy})
    maxy+=1
    return outytr,outyte,dic,maxy