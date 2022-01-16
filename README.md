# kerasTransferLearningCNNflowerDataset


We will learn to implement transfer learning in Python. For this implementation, we will use the flower recognition dataset from Kaggle.   This dataset has around 4000 images from 5 different classes, namely daisy, dandelion, rose, sunflower and tulip. 

![alt text](https://github.com/Ashutosh27ind/kerasTransferLearningCNNflowerDataset/blob/main/normalization_image.PNG?raw=true)

### Data source : https://www.kaggle.com/alxmamaev/flowers-recognition

### Platform : https://www.nimblebox.ai/  

In this notebook, we will implement transfer learning in Python using the pre-trained ResNet model. We will run two experiments - 1. Freezing the base model weights, adding a few layers to it at the end (fully connected etc.) and training the newly added layers, and 2. Freezing the first 140 layers of ResNet and retraining the rest.  

Apart from this, you will learn two important practical preprocessing techniques in this notebook - data augmentation and data generators. The notebook is dividede into the following sections:  

Importing libraries  
Splitting into train and test set  
Importing the pretrained ResNet model  
Data Generators: Preprocessing and Generating Batch-Wise Data (On the Fly)  
Training the Base Model (Using Batch-Wise Data Generation)  
Freezing the initial-n layers and training the rest  

## Conclusion from experiment:  
To summarise, we conducted two transfer learning experiments. In the first experiment, we removed the last fully connected layers of ResNet (which had learnt how to classify the 1000 ImageNet images). Instead, we added our own pooling, fully connected and a 5-softmax layer and trained only those. Notice that we got very good accuracy in just a few epochs. In case we weren't satisfied with the results, we could modify this network further (add an FC layer, modify the learning rate, replace the global average pooling layer with max pool, etc.).  
In the second experiment, we froze the first 140 layers of the model (i.e. used the pre-trained ResNet weights from layers 1-140) and trained the rest of the layers. Note that while updating the pre-trained weights, we should use a small learning rate. This is because we do not expect the weights to change drastically (we expect them to have learnt some generic patterns, and want to tune them only a little to accommodate for the new task). 
