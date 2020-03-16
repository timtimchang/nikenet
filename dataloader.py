import cv2
import os
import random
import numpy as np
import math

def loaddata(img_name):
    #print(img_name)
    pic = cv2.imread(img_name)
    pic = cv2.resize(pic, (224,224), interpolation=cv2.INTER_CUBIC)
    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)

    #pic = cv2.normalize(pic,None,0,255,cv2.NORM_MINMAX)
    #pic = cv2.normalize(pic, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    pic =  pic[:,:,np.newaxis]
    return pic/255.0
def GenBatch(TrainData,bsize=4):
    class_num = len(TrainData)
    if(len(TrainData)<bsize):
        return None,None,1
    input=[] # abs path file name
    label=[] # INT label
    class_list=random.sample(range(0, class_num), bsize)
    delete_class=[]
    for i in class_list:
        #print(i)
        
        iter_class=TrainData[i]
        delete_class.append(TrainData[i])
        #print("i:",i,len(iter_class))
        file_list=random.sample(range(0, len(iter_class)), bsize)
        delete_list=[]
        for j in file_list:
            try:
                data = loaddata(TrainData[i][j][0])
            except BaseException:
                print("index:",i,TrainData[i][j])
            input.append(data)
            label.append([TrainData[i][j][1]])
            delete_list.append(TrainData[i][j])
        for j in delete_list:
            TrainData[i].remove(j)
            
    for i in delete_class:
        if(len(i)<bsize):
            TrainData.remove(i)
    #print(len(TrainData))
    #print(label)
    
    return np.array(input), np.array(label),0

def GenRandomBatch(TrainData,bsize=32):
    input=[] # abs path file dname
    label=[] # INT label
    img_index=random.sample(range(0, len(TrainData)), bsize)
    for i in img_index:
        try:
            data = loaddata(TrainData[i][0])
        except BaseException:
            print("index:",i,TrainData[i])
        input.append(data)
        label.append([TrainData[i][1]])
            
    return np.array(input), np.array(label)     
def triplet_to_softmax(data):
    transform_data = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            transform_data.append(data[i][j])
    return transform_data

def dataloader(data_path):
    dataset=data_path
    index=0
    TrainData=[]
    TestData=[]
    ratio=0.95
    for dicname in (os.listdir(dataset)):
        path=os.path.join(dataset,dicname)
        if path == 'dataset/downloads/.DS_Store' : continue
        TrainData.append([])
        TestData.append([])
        class_size=len(os.listdir(path))
        counter=0
        random.seed(777)
        allfile_list=os.listdir(path)
        for filename in  allfile_list :
            counter+=1
            if(counter<=int(class_size*ratio)):
                TrainData[index].append([os.path.join(path,filename),index])
            else:
                TestData[index].append([os.path.join(path,filename),index])
    
        index+=1
    return TrainData, TestData

        
def GenNoBatch(TestData):
    input = []
    label = []
    end = 0
    
    if len(TestData) == 0:
        end = 1

    for i in range(len(TestData)):
        amount = len(TestData[i])
        for j in range(amount):
            #print("index:",i,TrainData[i][j])
            data = loaddata(TestData[i][j][0])
            input.append(data)
            label.append([TestData[i][j][1]])

    return np.array(input), np.array(label),end

if __name__ == '__main__':
    dataset='dataset/downloads'
    index=0
    TrainData=[]
    # file_set = [file for file in os.listdir(dataset)]
    for dicname in (os.listdir(dataset)):
        #TrainData.append([])
        path=os.path.join(dataset,dicname)
        if path == 'dataset/downloads/.DS_Store' : continue
        #print(len(os.listdir(path)))
        #print((path))
        #if(len(os.listdir(path))<10):
        #     print("error",path)
        #     continue
        TrainData.append([])
       # print(os.listdir(path))
        #img_paths = [img_file for img_file in os.listdir(path) if (os.path.isfile(img_file)) and (img_file.endswith('.jpg') or img_file.endswith('.png'))]
        for filename in  (os.listdir(path)):
           #if (os.path.isfile(img_file)) and (img_file.endswith('.jpg') or img_file.endswith('.png')):
           # try:
           #     loaddata(os.path.join(path,filename))
           # except:
           #     print(os.path.join(path,filename))
                #os.system("rm -rf "+(os.path.join(path,filename)))
            TrainData[index].append([os.path.join(path,filename),index])
        index+=1
    
    #print("len",len(TrainData[60]))
    #print(len(TrainData[5]))
    input,label = GenBatch(TrainData,8,len(TrainData))
    loaddata(input[0])

