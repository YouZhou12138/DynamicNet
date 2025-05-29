# DynamicNet

This is the handwritten code for the manuscript entitled "DynamicNet: Vision Transformer with Prior Space Hypothesis for Spatiotemporal Prediction", for editors and reviewers to verify the authenticity of the method.
The code is implemented with the help of the pytorch_lightning library within the pytorch framework. You just need to install the latest versions of the libraries required by the code. Of course, for fairness, the version of skimage should be less than 0.19.2.

Train

python train.py "path to .yaml file"  --data_name taxibj  --data_dir "The parent class of all folders(For example "/home/data")"
