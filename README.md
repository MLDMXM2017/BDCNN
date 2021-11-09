# BDCNN
the source code of paper, "Block Division Convolutional Network with Implicit Deep Features Augmentation for Micro-Expression Recognition."

## Requirements

Python==3.7.6    
torch==1.8.1   
torchvision==0.9.1  
pandas   
tqdm    
sklearn    
matplotlib   
opencv_python==4.2.0.34  
pickle  

## Usage
### Dataset Preparation

You may put the 3-class and 5-class data into the folder data.  
The directory of data shall be represented as follows：    
* 006  
  * Train  
    * img001.img  
    * ...  
    * label.txt  
  * test  
    * img001.img  
    * ...  
    * label.txt  
* 007  
...  

### Training the BDCNN

Training the 3-class data by using the following command:
```Bash  
cd code  
python train_split.py --model BDCNN --save_path result/BDCNN
```

Training the 5-class data by using the following command:
```Bash   
python train_5type_split.py  --model BDCNN --save_path result/BDCNN_5type
```
### Get results

Then you can run the result.py to get the result of acc,recall,F1,UF1, and UAR  
```Bash    
python result.py  --type 3_class --result_path result/BDCNN
```

### Draw the tsne
```Bash    
python tsne.py  --path result/BDCNN --foder 006 --name BDCNN
```






