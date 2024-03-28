import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee'])

plt.style.use('science')
sns.set_theme(style="white",font='sans-serif',font_scale=1.4) #serif,simhei
plt.rcParams['font.sans-serif']=['Times New Roman'] #用来正常显示中文标签
font1 = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18,
        }
font2 = {'family': 'simhei',
        'weight': 'normal',
        'size': 18,
        }
fonts = {'en':font1,
        'cn':font2}

def heatmap1(matrix,x_labels='',
            y_labels='',x_title = '',y_title = '',
            title = '',font_type = 'en',plot_dir = './',name = '',cmap =None,
            save_type = 'svg'):
    '''
    
    camp 关注大小默认即可
    把中间值作白，则用 coolworm
    
    weight 2 36*32 so figsize = (9,8)
    weight 3 32*32 so figsize = (8,8)
    
    '''
    f, ax = plt.subplots(figsize=(9, 3))
    font = fonts[font_type]
    
    sns.heatmap(matrix, cmap =cmap, annot=False, fmt="f",
                linewidths=.5, ax=ax,xticklabels = x_labels, yticklabels = y_labels,
                vmax=-0.5, vmin=0.5)
    plt.title(title,font)
    # plt.title('Result Analysis')

    plt.xlabel(x_title,font)
    plt.ylabel(y_title,font)
    plt.savefig(plot_dir + f'{name}heatmap.'+save_type,dpi=512) 
def heatmap_confusion(matrix,x_labels=['1','2','3'],
            y_labels=['1','2','3'],x_title = 'Label',y_title = 'Prediction',
            title = '',font_type = 'en',plot_dir = './',name = '',cmap =None,
            save_type = '.svg'):
    '''
    
    camp 关注大小默认即可
    把中间值作白，则用 coolworm
    
    weight 2 36*32 so figsize = (9,8)
    weight 3 32*32 so figsize = (8,8)
    
    '''
    f, ax = plt.subplots(figsize=(4, 4))
    font = fonts[font_type]
    
    sns.heatmap(matrix, cmap =cmap, annot=True, fmt=".0f",
                linewidths=.5, ax=ax,xticklabels = x_labels, yticklabels = y_labels,
                vmax=None, vmin=None)
    plt.title(title,font)
    # plt.title('Result Analysis')

    plt.xlabel(x_title,font)
    plt.ylabel(y_title,font)
    plt.savefig(plot_dir + f'{name}heatmap'+save_type, transparent=True,dpi=512)       
def scatterplot(x,name = 'Ottawa',label = ['N','I','O'],font_type = 'en',plot_dir = './'):
    
    '''
    feature for ottawa
    '''
    import matplotlib.pyplot as plt
    length = len(x)
    classes = int(length//100)
    clist = ['r','lightsalmon','gold','yellow','palegreen',
         'lightseagreen','lightblue','slateblue','violet','purple']
    markerlist=['.',',','o','v','^','<','>','D','*','h','x','+']
    f, ax = plt.subplots(figsize=(9, 8))
    font = fonts[font_type]  
    for i in range(classes):
        plt.scatter(x[0+i*100:i*100+100,0],x[0+i*100:i*100+100,1],label=label[i])
    
    plt.legend(loc='best',fontsize=14, fancybox=False, shadow=False)  
    
    plt.savefig(plot_dir + f'{name}scatter.svg',dpi=256)
    plt.show()

def scatterplot1(x,name = 'Ottawa',label = ['N','I','O'],font_type = 'en',plot_dir = './'):
    
    '''
    feature for ottawa
    '''
    import matplotlib.pyplot as plt
    length = len(x)
    classes = int(length//100)
    clist = ['cadetblue','royalblue','y','grey','palegreen',
         'lightseagreen','lightblue','slateblue','violet','purple']
    markerlist=['*','v','^','o','v','^','<','>','D','*','h','x','+']
    f, ax = plt.subplots(figsize=(6, 6))
    font = fonts[font_type]  
    t = list(range(0,length))
    for i in range(classes):
        plt.scatter(t[0:100],x[0+i*100:i*100+100],label=label[i],c=clist[i],marker=markerlist[i])
    
    plt.legend(loc='best',fontsize=14, fancybox=False, shadow=False)  
    
    plt.savefig(plot_dir + f'{name}scatter.svg',dpi=256)
    plt.show()
    
def lineplot(dic,fig_name = './'):
    '''

    '''
    for i in dic.keys():
        plt.title('Result Analysis')
        plt.plot(dic[i],label=i)
        plt.legend() # 显示图例
        plt.xticks([0,1,2,3])
        plt.yticks(['0','2','3','0'])        
        plt.xlabel('iteration times')
        plt.ylabel('rate')
    plt.savefig(fig_name+'.svg',dpi=512)   
    plt.show()
    
    
def plot2D(dic,setname='Ottawa',save_dir='plot_dir/',TLname='default',num = 100, class_num = 3,sample = 32):
    #for name,dimen_result in dic.items():
        
    tSNEx=dic[:,0]
    tSNEy=dic[:,1]
    
    
    


#        clist = ['r','y','g','b','c','aqua','lawngreen','lightskyblue','limegreen','lemonchiffon','mediumblue'] 
    # clist = ['r','lightsalmon','gold','yellow','palegreen',
    #          'lightseagreen','lightblue','slateblue','violet','purple']
    markerlist=['.','v','o',',','^','<','>','D','*','h','x','+']
    clist = ['seegreen','firebrick','darkblue']
    if setname=='Ottawa':
        label = ['Nor','IF','OF']

    for i in range(3):
        plt.scatter(tSNEx[0+i*num:sample+i*num],tSNEy[0+i*num:sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[i],label=label[i])
    plt.xticks([])
 
    plt.yticks([])
    plt.savefig(save_dir+TLname+'.svg',format='svg')
    plt.show()
#%% TSNE DG
           
def plot2DDG(dic,setname='ottawa',save_dir='plot_dir/',TLname='default',
             num = 100, class_num = 3,sample = 32):
    #for name,dimen_result in dic.items():
        
    tSNEx=dic[:,0]
    tSNEy=dic[:,1]
    
    # num = 602
    
    
    target_idx = int(num*class_num) 
    target_idx2 = int(num*class_num*2) 

    target_idx3 = int(num*class_num*3)   
    


#        clist = ['r','y','g','b','c','aqua','lawngreen','lightskyblue','limegreen','lemonchiffon','mediumblue'] 
    clist = ['cadetblue','royalblue','y','grey','palegreen',
             'lightseagreen','','slateblue','violet','purple']
    labelCWRU = ['NOR','I07','I14','I21',
             'B07','B14','B21',
             'O07','O14','O21'
             ]
    labelSBDB = ['Nor','I02','I04','I06',
     'B02','B04','B06',
     'O02','O04','O06'
     ]
    labelottawa = ['Nor','IF','OF']
    if setname=='CWRU':
        label=labelCWRU
    elif setname=='SBDB':
        label=labelSBDB
    elif setname =='ottawa':
        label = labelottawa


    
    markerlist=['*','^','o','v',',','<','>','D','*','h','x','+']

    for i in range(class_num):
        
        
        plt.scatter(tSNEx[0+i*num:sample+i*num],tSNEy[0+i*num:sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[0],label='S0'+label[i])

        plt.scatter(tSNEx[target_idx+i*num:target_idx+sample+i*num],tSNEy[target_idx+i*num:target_idx+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[1],label='S1'+label[i])

        plt.scatter(tSNEx[target_idx2+i*num:target_idx2+sample+i*num],tSNEy[target_idx2+i*num:target_idx2+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[2],label='S2'+label[i])

        plt.scatter(tSNEx[target_idx3+i*num:target_idx3+sample+i*num],tSNEy[target_idx3+i*num:target_idx3+sample+i*num],
                    s=100,marker=markerlist[i],c='none',edgecolor=clist[3],label='T'+label[i])

        
    # plt.legend(loc='best',fontsize=14, fancybox=True, shadow=True)  
    plt.xticks([])
 
    plt.yticks([])
    plt.savefig(save_dir+TLname+'.svg',format='svg')
    plt.savefig(save_dir+TLname+'.png',format='png')
    plt.show()
    
    
#%%
if __name__ == '__main__': 
    # test heatmap 
    # x = np.random.random((4,4))
    # heatmap(x,x_labels = ['1','2','3','4'],y_labels=['1','2','3','4'],cmap = 'coolwarm')
    # test line plot
    # x = np.random.random((4,4))
    # y = np.random.random((1,4))
    # x2 = np.random.random((1,4))
    # y2 = np.random.random((1,4)) 
    # dic = {'x1':x,
    #        'x2':x2}
    # lineplot(dic)   
#%% Ottawa weight
    # npylist = ["epoch0.npy",
    #            "epoch277.npy",
    #            "epoch441.npy",
    #            "epoch521.npy",
    #            "epoch631.npy",
    #            "epoch895.npy",
    #            "epoch1018.npy",
    #            "epoch1247.npy",
    #            "epoch1793.npy"]
    # npylist = ["epoch0.npy",  # THU
    #            "epoch302.npy",
    #            "epoch506.npy",
    #            "epoch565.npy",
    #            "epoch652.npy",]
    npylist = ["epoch0.npy", # Ottawa
               "epoch177.npy",
               "epoch336.npy",
               "epoch493.npy",
               "epoch529.npy",]      
    for npy in npylist:
        w = np.load(npy, allow_pickle=True).item()
        for j in range(4,12):
            heatmap1(w['parms_weight'][j].squeeze(-1),
                     plot_dir = './',name=npy+str(j), cmap = 'coolwarm',
                     save_type = 'png')
#%% draw confusion matric
    m = np.load('0.5Confusion matrix.npy', allow_pickle=True) # 0.3164062500Confusion matrix.npy
    # heatmap_confusion(m[:4,:4].T,x_labels = ['N','IF','BF','OF'],y_labels=['N','IF','BF','OF'],cmap = 'coolwarm',
    #                   save_type = '.svg')
    m = np.load('0.5Confusion matrix.npy', allow_pickle=True)
    heatmap_confusion(m.T[:3,:3],x_labels = ['N','IF','OF'],y_labels=['N','IF','OF'],cmap = 'coolwarm')

#%% TSNE
# from sklearn.manifold import TSNE
# num = 100
# sample = 50
# tSNE = TSNE(n_components=2,n_iter=4000)
# #    print(sfis_plus_tfit)
# tSNE_results = tSNE.fit_transform(xf.reshape(300,32))
# scatterplot(tSNE_results,name = 'Ottawa',label = ['N','I','O'],font_type = 'en',plot_dir = './')
#%% 

    m = [[0.00,0.00,0.00,0.00,0.00],
[26.04,26.04,25.26,0.00,0.00], 
[30.73,26.04,63.28,79.17,81.51], 
[38.28,38.02,80.47,76.56,75.78], 
[37.50,37.50,75.00,83.59,83.59] 
]
    heatmap1(matrix=m,x_labels=['1','0.1','0.01','0.001','0.0001'],
             y_labels = ['1','0.1','0.01','0.001','0.0001'])

