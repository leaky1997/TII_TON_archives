import torch.utils.data as data
import torch
import numpy as np
import scipy.io as sio
import os
from einops import rearrange
#%%
class data_default(data.Dataset):
    def __init__(self,data_dir = None, flag = 'train') -> None: # flag = 'train / test / val'
        super().__init__()
        self.mat = sio.loadmat(data_dir)        
        self.xmode = 'x_'+flag
        self.ymode = 'y_'+flag  
        # if flag == 'val': # 一般用不到，因为考虑泛化性能，测试集往往是另外的一个领域，但是如果不考虑泛化性，则需要单独把test 划分为 val 和 test , 后续可以把data_maker + 验证集
        #     self.lenth = int(len(self.mat['x_test'])//2)
        #     self.x, self.y = self.mat['x_test'][:self.lenth],self.mat['y_test'][:self.lenth]
        # else:      
        #     self.x, self.y = self.mat[self.xmode],self.mat[self.ymode]
        self.x, self.y = self.mat[self.xmode],self.mat[self.ymode]
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        self.length = len(self.x)              
            

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.length
class data_multiple(data_default): 
    def __init__(self,data_list = None, flag = 'train'): # only have train and valid TODO with domain label
        if flag == 'val': flag ='test'
        self.xmode = 'x_'+flag
        self.ymode = 'y_'+flag  
        
        for i, data_dir in enumerate(data_list):
            self.mat = sio.loadmat(data_dir)         
            self.x, self.y = (np.concatenate((self.x, self.mat[self.xmode]),axis=0), np.concatenate((self.y, self.mat[self.ymode]),axis=0)) if i else (self.mat[self.xmode], self.mat[self.ymode])
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        self.length = len(self.x)   

class data_monitoring():
    """_summary_
    Args:
        rootpath (string): Root directory

        size: sample point number

        pred: predict point time i.e. size*pred
        
        data_path: data path
        
        scale: whether to scale the data
    """
    
    
    
    def __init__(self, data_dir,
                 flag='train', size=1024,pred = 1,
                  scale=True,sample_freq = 20480.0,
                 bearing=''):

        self.size = size
        self.pred_size = int(pred * self.size)
        self.bearing = bearing
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': [0.2,1],
                    'val': [0.6,1],
                    'test': [0,1],}
                    # 'pred':[0,1]}
        self.set_type = type_map[flag]
        self.scale = scale

        # self.data_path = data_dir
        self.path = data_dir # os.path.join(self.root_path,self.data_path) # dataset root path 
        
        # self.scaler = StandardScaler() # MinMaxScaler() # 
        self.sample_freq = sample_freq # not used by now
        
        if os.path.exists(self.path + f'/{self.bearing}.npy'):
            print('###load data from existing npy file###')
            self.data_dict = np.load(self.path + f'/{self.bearing}.npy',allow_pickle=True).item() #  不要cycle 1 的数据
        else:
            print('no data file')
            self.__read_data__()
            self.__pre_processing__()
        

        self.length = len(self.data_dict['split_data'])
        
        board_1 = int(self.set_type[0] * self.length)
        
        board_2 = int(self.set_type[1] * self.length)
                
        self.x = self.data_dict['split_data'][board_1:board_2].astype('float32')
        self.x = torch.from_numpy(self.x)
        self.x = rearrange(self.x,'n c l -> n l c')  # dun
        
        self.y = self.data_dict['life'][board_1:board_2].astype('float32')
        
        self.t = self.data_dict['cycle'][board_1:board_2].astype('float32')
        self.t = torch.from_numpy(self.t)
        
        self.y = torch.from_numpy(self.y)
        self.length = len(self.x)
        
    def __len__(self):
        return self.length
                
    def __read_data__(self):
        pass
        # self.data_stamp = data_stamp
        
    def __pre_processing__(self):
        pass # todo # 
    
    def __getitem__(self, index):
        '''
        return a dict with keys；
            'x': split_data[index]
            'anomaly_label': anomaly_label[index]
            'li
        ''' 

        
        return self.x[index], self.y[index] #, self.t[index]  # TODO 


    
## todo RUL data  
  
if __name__ == '__main__':
    # # dataset = data_default(data_dir = "/home/richie_thu/_Richie_project/dataset/CWRU_0hp_10.mat",flag = 'train')
    # # x,y = dataset.__getitem__(100)
    # # print(x,y)
    # data_dir = "/home/richie_thu/_Richie_project/dataset/"
    # data_list = ["CWRU_0hp_10.mat","CWRU_1hp_10.mat","CWRU_2hp_10.mat","CWRU_3hp_10.mat"]
    # data_list = [data_dir + d for d in data_list]
    # dataset = data_multiple(data_list = data_list,flag = 'train')
    # x,y = dataset.__getitem__(100)
    # print(x,y)
    data_dir = "/home/richie_thu/_Richie_project/U-PHM/dataset/XJTU"
    data_path = 'XJTU'
    dataset = data_monitoring(
        data_dir = data_dir,
        flag='train', size=1024,pred = 1,
        bearing = 'Bearing1_1'
    )
    x,y = dataset.__getitem__(100)
    print(x,y)
    