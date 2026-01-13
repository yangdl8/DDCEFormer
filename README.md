## 💡 Installation

🔥DDCEFormer🔥 is tested on Windows with Pytorch 1.9.1 and Python 3.8.8. 
- Create a conda environment: ```conda create -n DDCEFormer python=3.8.8```
- Install PyTorch 1.9.1 and Torchvision 0.10.1 following the [official instructions](https://pytorch.org/)
- ```pip3 install -r requirements.txt```

## 🐳 Download pretrained models

🔥DDCEFormer🔥's pretrained models can be found in [here](https://pan.baidu.com/s/1zT7Cf0CZVVSduyJ92LYuSA?pwd=a311), please download it and put it in the './pretrained' directory. 
  
## 🤖 Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website, and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## 🚅 Test the model

You can obtain the results. 

```bash
python main_run.py --test --previous_dir pretrained 
```

## ⚡ Train the model

To train the DDCEFormer model on Human3.6M:

```bash
python main_run.py
```

## 👍 Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [MHFormer](https://github.com/Vegetebird/MHFormer)

## 🔒 Licence

This project is licensed under the terms of the MIT license.
