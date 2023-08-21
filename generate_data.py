from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class GenerateData():
    def __init__(self, datapath):
        self.data_path = datapath

    def splitData(self, val_filename= 'val'):
        train_csv_path = self.data_path +"/"+ 'train.csv'
        train = pd.read_csv(train_csv_path)
        validation_data = pd.DataFrame(train.iloc[:3589,:])
        train_data = pd.DataFrame(train.iloc[3589:,:])
        train_data.to_csv(self.data_path+"/train.csv")
        validation_data.to_csv(self.data_path+"/"+val_filename+".csv")
        print("Done splitting the file")

    def strToImage(self, str_img = ' '):
        imgarray_str = str_img.split(' ')
        imgarray = np.asarray(imgarray_str,dtype=np.uint8).reshape(48,48)
        return Image.fromarray(imgarray)

    def saveImages(self, datatype='train'):
        foldername= self.data_path+"/"+datatype
        csvfile_path= self.data_path+"/"+datatype+'.csv'
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        data = pd.read_csv(csvfile_path)
        images = data['pixels']
        numberofimages = images.shape[0]
        for index in tqdm(range(numberofimages)):
            img = self.strToImage(images[index])
            img.save(os.path.join(foldername,'{}{}.jpg'.format(datatype,index)),'JPEG')
        print('Done saving {} data'.format((foldername)))
