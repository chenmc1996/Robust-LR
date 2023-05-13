# Robust-LR

<p align="center">
<img src="src/method.jpg" width="300">
</p>


## Experiments

To obtain the results on standard learning with noisy labels benchmarks,run RoLR.py, such as

     $ python RoLR.py --data_path path-to-your-dataset --num_class 10 --dataset cifar10 --num_epochs 500 --lambda_p 2 --T 1 --r 0.8

To obtain the results on learning with long-tailed noisy labels benchmarks,run RoLR_LT.py, such as

     $ python RoLR_LT.py --data_path path-to-your-dataset --num_class 10 --dataset cifar10 --num_epochs 500 --lambda_p 2 --T 1 --imbalance 0.02 --r 0.2 --conf_mode W --class_weight freq 

## Requirements

- Python >= 3.6
- PyTorch >= 1.6
- CUDA
- Numpy


## Reference
We thank the implementation of DivideMix and FixMatch in:
* ["https://github.com/LiJunnan1992/DivideMix"](https://github.com/LiJunnan1992/DivideMix),
* ["https://github.com/LeeDoYup/FixMatch-pytorch"](https://github.com/LeeDoYup/FixMatch-pytorch).
