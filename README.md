# CLASSIFICATION

Use images and text on products to identify products and violations of each brand or product

### TODO

- [x] Models zoo

    - [ ] MobilenetV2.

    - [x] MobileNetV3.

    - [x] ResNet.

        - [x] ResNet50.

        - [x] ResNet34.

        - [x] ResNet18.

    - [ ] DenseNet.

    - [ ] ShuffleNet.

    - [ ] EfficientNetV1.

    - [ ] EfficientNetV2.

- [x] Augmentation

    - [x] RandomAugment.

    - [x] AutoAugment.

    - [x] Photometric-augmentation.

        - [x] hist_equalize.

        - [x] invert.

        - [x] mix_up_gray_scale.

        - [x] adjust_brightness.

        - [x] solarize.

        - [x] posterize.

        - [x] contrast.

        - [x] hsv_augment.

        - [ ] mixup.

        - [ ] Blur.

        - [ ] Media-Blur.

    - [x] Geometrics-augmentation.

        - [x] shearX.

        - [x] shearY.

        - [x] translateX.

        - [x] translateY.

        - [x] Cut-out.

        - [x] Rotate.

        - [ ] Sharpness.

        - [ ] Center-crop.

- [x] Training strategy.

    - [x] Weight decay.

    - [x] Warm-up learning rate.

    - [ ] Learning Schedual.

    - [x] Early-stopping.

    - [ ] FixRes strategy. [detail](https://arxiv.org/pdf/1906.06423.pdf)

    - [ ] Progressive learning with adaptive regularization. [detail](https://arxiv.org/pdf/2104.00298.pdf)

- [x] Data-format.

    - [x] Pandas-DataFrame

    - [x] Folder-format

- [x] Visualize.

    - [x] training progress (tensorboard).

    - [x] statistic

    - [x] Confusion-matrix.

    - [x] ROC.

### REQUIREMENTS

`
    pip install -r requirement.txt
`

### FOLDER TREE

```

PROJECT

├── config/
|    ├── default/
|    |      ├── train_config.yaml
|    |      ├── data_config.yaml
|
|
├── source/
|    ├── models/
|    |      ├── mobilenetv2.py
|    |      ├── Resnet.py
|    |      ├── ...
|    |      ├── ...
|    |      ├── EfficientNet.py
|    ├── utils/
|    |      ├── dataset.py
|    |      ├── augmentations.py
|    |      ├── general.py
|    |      ├── torch_utils.py
|    |      ├── imbalance_data_handle.py
|    ├── train.py
|
|
├── result/                          
|    ├── runs_1/
|    ├── runs_2/
|    ├── ..... /
|    ├── runs_n/
|    |      ├── checkpoint/
|    |      |      ├── best.pt
|    |      |      ├── last.pt
|    |      ├── visualize/
|    |      |      ├── *.jpg
|
|
├── .gitignore
├── README.md
├── LICENSE
├── preprocess.py               
├── demo_video.py

```

## Training

See config file for more detail.
Put your whole data in DATA_FOLDER [config-file](config/default/data_config.yaml)
### 1.Pandas Format: 

colums-name: 

    path : images-relative path with DATA_FOLDER 

    label_name_1 : must be the same with *classes in data_config.yaml. values can be 'class_name' or class_index

    ...

    label_name_n:

### 2.Folder-format

image-path : DATAFOLDER / label_1/ label_2 /.../ label_n / *.jpg

### 3.Run

```
    python source/train.py \
    --cfg YOUR_PATH/train_config.yaml \
    --data YOUR_PATH/data.yaml 
```

## Reference 

[Deep Residual Learning for Image Recognition.](https://arxiv.org/abs/1512.03385)

[MobileNetV2: Inverted Residuals and Linear Bottlenecks.](https://arxiv.org/abs/1801.04381)

[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices.](https://arxiv.org/abs/1707.01083)

[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.](https://arxiv.org/abs/1905.11946v1?fbclid=IwAR15HgcBlYsePX34qTK2aHti_GiucEYpQHjben-8wsTf7O83YPhrJQgXEJ0)
    
