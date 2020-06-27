# How to get the theta argument of affine_grid? (Send me email if you need an english version)

![Paper.创意.5.png](https://i.loli.net/2019/12/14/kJ6rKD1C7lMP9xT.png)

## I have finished the detection branch and am still training the model to verify its correctness. All the features will be published to develop branch, and keep master stable. 
 - ICDAR Dataset 
 - SynthText 800K Dataset
 - detection branch (verified on the training set, It works!)
 - eval
 - multi-gpu training
 - crnn (not be verified)
 - reasonable project structure
 - val loss
 - tensorboardx visualization

# Introduction

This is a PyTorch implementation of [FOTS](https://arxiv.org/abs/1801.01671). 
 
# Questions

- Should I fix weights of the backbone network, resnet50 ?
  ```python
  for param in self.backbone.parameters():
      param.requires_grad = False
  ```
  Answer: Yes, the backbone network is used as a feature extractor, so we do not need to modify the weights.
 
- For crnn, the padding size should all be 1, since the width may less than the kernel size, and the outputs' sizes of 
conv layer in CRNN are all the same? 

# Instruction

## Requirements

1. build tools

   ```
   ./build.sh
   ```

2. prepare ICDAR Dataset


## Training

1. understand your training configuration

   ```
   {
        "name": "FOTS",
        "cuda": false,
        "gpus": [0, 1, 2, 3],
        "data_loader": {
            "dataset":"icdar2015",
            "data_dir": "/Users/luning/Dev/data/icdar/icdar2015/4.4/training",
            "batch_size": 32,
            "shuffle": true,
            "workers": 4
        },
        "validation": {
            "validation_split": 0.1,
            "shuffle": true
        },
    
        "lr_scheduler_type": "ExponentialLR",
        "lr_scheduler_freq": 10000,
        "lr_scheduler": {
                "gamma": 0.94
        },
     
        "optimizer_type": "Adam",
        "optimizer": {
            "lr": 0.0001,
            "weight_decay": 1e-5
        },
        "loss": "FOTSLoss",
        "metrics": ["my_metric", "my_metric2"],
        "trainer": {
            "epochs": 100000,
            "save_dir": "saved/",
            "save_freq": 10,
            "verbosity": 2,
            "monitor": "val_loss",
            "monitor_mode": "min"
        },
        "arch": "FOTSModel",
        "model": {
            "mode": "detection"
        }
   }

   ``` 

2. train your model

   ```
   python3 train.py -c config.json

   python3 train.py -c config.json -r ./saved/FOTS/model_best.pth.tar
   ```
   
## Evaluation

```
python eval.py -m <model.tar.gz> -i <input_images_folder> -o <output_folders>
python3 eval.py -m  ./saved/FOTS/model_best.pth.tar -d /home/ec2-user/datasets/en_20  -i ../test_image_1 -o ./output/

```



