import pandas as pd
import os
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

def evaluate(conf,class_num):
    # get acc,recall, precision, F1,UF1,UAR 
    Acc, Precision, Recall = 0, 0, 0
    UF1, UAR = 0, 0
    for i in range(class_num):
        TP = conf[i][i]
        TN = sum([conf[x][y] for x in range(class_num) for y in range(class_num) if x != i and y != i])

        FP = sum([conf[j][i] for j in range(class_num) if j != i])
        FN = sum([conf[i][j] for j in range(class_num) if j != i])

        #acc = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)

        Acc += TP
        Recall += recall
        Precision += precision

        UF1 += (2 * precision * recall) / (precision + recall)
        uar = TP / (TP + FN)
        UAR += uar
    Acc=Acc/sum([conf[x][y] for x in range(class_num) for y in range(class_num)])
    Recall /= class_num
    Precision /= class_num
    macro_F1 = (2 * Precision * Recall) / (Precision + Recall)

    UF1 /= class_num
    UAR /= class_num
    return round(Acc, 4), round(Recall, 4), round(Precision, 4), round(macro_F1, 4), round(UF1, 4), round(UAR, 4)

def get_result(conf,s,class_num):
    conf=list(map(list,zip(*conf)))
    print(s+' result:')           
    print(conf)
    print('acc,recall,precision,F1,UF1,UAR:')
    print(evaluate(conf))
    for i in range(class_num):
        add=sum(conf[i])
        for j in range(class_num):
            conf[i][j]=round(conf[i][j]/add,4)
    print(conf)
    return conf
    
def draw_conf_matirx(conf):
    plt.imshow(conf, cmap=plt.cm.Blues)
    indices = range(len(conf))
    plt.xticks(indices)
    plt.yticks(indices)
    plt.colorbar()
    for first_index in range(len(conf)):
        for second_index in range(len(conf[first_index])):
            plt.text(first_index, second_index, conf[first_index][second_index])
     
    plt.show()

def get_3_class_result(result_path):
    class_num=3

    folders=os.listdir(result_path) 

    conf_sum=[[0]*class_num for _ in range(class_num)]
    smic_sum=[[0]*class_num for _ in range(class_num)]   #s01
    samm_sum=[[0]*class_num for _ in range(class_num)]   #006
    casme_sum=[[0]*class_num for _ in range(class_num)]  #sub01

    for folder in tqdm(folders):
       csv_path=os.path.join(result_path,folder,'conf_matrix.csv')
       df=pd.read_csv(csv_path)
       for i in range(class_num):
           for j in range(class_num):
               conf_sum[i][j]=conf_sum[i][j]+int(df.loc[df['Unnamed: 0']==i,str(j)][i])
               if folder[0]=='s' and len(folder)==3:
                   smic_sum[i][j]=smic_sum[i][j]+int(df.loc[df['Unnamed: 0']==i,str(j)][i])
               elif folder[0]=='0':
                   samm_sum[i][j]=samm_sum[i][j]+int(df.loc[df['Unnamed: 0']==i,str(j)][i])
               else:
                   casme_sum[i][j]=casme_sum[i][j]+int(df.loc[df['Unnamed: 0']==i,str(j)][i])
   conf=get_result(conf_sum,'Full')
   #draw_conf_matirx(conf)
   get_result(smic_sum,'SMIC')
   get_result(casme_sum,'CASME')
   get_result(samm_sum,'SAMM')

def get_5_class_result(result_path):
    class_num=5

    folders=os.listdir(result_path) 

    conf_sum=[[0]*class_num for _ in range(class_num)]
    samm_sum=[[0]*class_num for _ in range(class_num)]   #006
    casme_sum=[[0]*class_num for _ in range(class_num)]  #sub01

    for folder in tqdm(folders):
       csv_path=os.path.join(result_path,folder,'conf_matrix.csv')
       df=pd.read_csv(csv_path)
       for i in range(class_num):
           for j in range(class_num):
               conf_sum[i][j]=conf_sum[i][j]+int(df.loc[df['Unnamed: 0']==i,str(j)][i])
               if folder[0]=='0':
                   samm_sum[i][j]=samm_sum[i][j]+int(df.loc[df['Unnamed: 0']==i,str(j)][i])
               else:
                   casme_sum[i][j]=casme_sum[i][j]+int(df.loc[df['Unnamed: 0']==i,str(j)][i])

   get_result(conf_sum,'Full')
   get_result(casme_sum,'CASME')
   get_result(samm_sum,'SAMM')

parser = argparse.ArgumentParser(description='ME recognition using ISDA')
parser.add_argument('--type', default='3_class', type=str,help='3_class or 5_class')
parser.add_argument('--result_path', '-p', default=2, type=int,help='your result path')
args = parser.parse_args()

if args.type=='3_class':
    get_3_class_result(args.result_path)  #the result path of 3-class
else:
    get_5_class_result(args.result_path)  #the result path of 5-class
  
