#   OCR 文本区域检测


FOTS 算法实现


## 建立环境


## Requirements

1. build tools

```shell script

conda create -n  ocr python=3.6 pip scipy numpy ##运用conda 创建python环境
source activate ocr
pip install -r requirements.txt -i https://mirrors.163.com/pypi/simple/

```

2. prepare ICDAR Dataset

[Synth Chinese ](https://github.com/dikers/ocr_synth_text_chinese)

[icdar dataset](https://rrc.cvc.uab.es/?com=introduction)



## Training

1. understand your training configuration

   ```
   {
        "name": "FOTS",
        "cuda": false,
        "gpus": [0, 1, 2, 3],
        "data_loader": {
            "dataset":"icdar2015",
            "data_dir": "./icdar2015/4.4/training",
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
python3 eval.py -m  ./saved/FOTS/model_best.pth.tar -i ./temp/ -o ./output/

```



