# Contrastive Learning on 3D Point Clouds with Data Augmentation
This is contrastive learning structure for point cloud datasets that adopted transformation for data augmentation  
  
Package requirement: Python3.8, keras, tensorflow2.8.0, numpy, matplotlib, h5py, tqdm

## Dataset Download
Some sample data are provided in the repository  
To download the full dataset:
- ModelNet40 [Link](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)
- ModelNet10 [Link](https://cloud.tsinghua.edu.cn/f/b3d9fe3e2a514def8097/)  

Unzip the downloaded file and put all the training `.h5` file into `Data_Train` folder. Then put all the testing `.h5` file into `Data_Test` folder.

## How to Run code
# Training the model
Run `train.py`  
Then the trained weight will be saved in the local.

# Linear evaluation
Modify the `model_name` in `linear_evaluation.py` to the weight saved by training the model  
Run `linear_evaluation.py`  
The plot for training history and t-SNE  will be saved.

# Supervised model
Run `supervised.py`
