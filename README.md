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







