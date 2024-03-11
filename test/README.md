# Test Details
- Put the condensed data at ```./test/results```.
- Process the original dataset with ```../data_processing.ipynb```.
- After you have the processed dataset, for cifar10 you can refer to the ```load_cifar``` function and for imagenet you can refer to the ```load_imagenet``` function. Both functions are in ```test_glad.py```.

## Test Commands

For example, to evaluate (10 images/class) on CIFAR-10 , run
```
python test_glad.py -d cifar10 -n convnet -s glad --ipc 10 --repeat 10
```

To evaluate (10 images/class) on ImageNet-10 , run
```
python test_glad.py -d imagenet --subset f -s glad --ipc 10 --repeat 10
```
  