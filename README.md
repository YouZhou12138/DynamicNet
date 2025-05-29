# DynamicNet

This is the handwritten code for the manuscript entitled "DynamicNet: Vision Transformer with Prior Space Hypothesis for Spatiotemporal Prediction", for editors and reviewers to verify the authenticity of the method.
The code is implemented with the help of the pytorch_lightning library within the pytorch framework. You just need to install the latest versions of the libraries required by the code. Of course, for fairness, the version of skimage should be less than 0.19.2.

# Train

python train.py "path to .yaml file"  --data_name taxibj  --data_dir "The parent class of all folders (For example "/home/data")"

# Example:
python train.py D:\Pycharm_workspace\ST\Total\configs\taxibj\DynamicNet\DynamicNet7M\seed=27.yaml --data_name taxibj --data_dir E:\dataset

# Test

We’ve stored the trained model weights and parameter settings for the various datasets mentioned in the paper on Baidu Cloud (https://pan.baidu.com/s/1NlCBd0jdpiNQtef2YdT5wA?pwd=vw3c) . This makes it convenient for you to run the test.py file to verify the authenticity of the method. 

python test.py "path to .yaml file"  "path to checkpoint (weight)" --data_name taxibj  --data_dir "The parent class of all folders (For example "/home/data")" --pred_dir "Just enter the address where you want to save the performance metrics and visualization results of the model (maybe you need to create this file path first)."

# Example:
python test.py D:\Pycharm_workspace\ST\Total\configs\greenearthnet\DynamicNet\DynamicNet7M\seed=27.yaml
D:\Pycharm_workspace\ST\Total\experiments\greenearthnet\DynamicNet_multi\seed=27\epoch=35-RMSE_Veg=0.1438.ckpt
--data_name greenearthnet
--pred_dir D:\Pycharm_workspace\ST\Total\experiments\greenearthnet\DynamicNet_multi\seed=27\pred
--data_dir E:\dataset

# Eval

For the multimodal dataset GreenEarthNet, the eval.py file needs to be executed additionally to compare the regional NDVI results predicted by the model with the NDVI of the actual test set.

python eval.py  "path to Target test set address" "Predicted address" "Score saving address"
# Example:
python eval.py E:\dataset\greenearthnet\ood-t_chopped E:\writing\RDformer\experiment\greenearthnet\model_score\DynamicNet\seed=27\ood\ood-t E:\writing\RDformer\experiment\greenearthnet\model_score\DynamicNet\seed=27
