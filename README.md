# CausalGS: Learning Physical Causality of 3D Dynamic Scenes with Gaussian Representations
<img width="1121" height="502" alt="屏幕截图 2025-12-25 194714" src="https://github.com/user-attachments/assets/4fc52fd4-f390-4d21-8c63-883f27932baa" />

## ⚙️ Installation
```shell script
git clone https://github.com/DustSettled/CausalGS.git --recursive
cd CausalGS

### CUDA 12.4
conda env create -f env.yml
conda activate CausalGS

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# install gaussian requirements
pip install submodules/depth-diff-gaussian-rasterization
pip install submodules/simple-knn
```

## 💾 Datasets
All the datasets will be uploaded soon. We organize the dataset following [D-NeRF](https://github.com/albertpumarola/D-NeRF) convention.
We split the dataset as:
- **train**: contains the frames within observed time interval, used for training the model.
- **val**: contains the frames within observed time interval but for novel views, used for evaluating *novel-view interpolation*.
- **test**: contains the frames in unobserved **future** time for both observed and novel views, used for evaluating *future extrapolation*.

Datasets can be downloaded from HuggingFace: 
- [Dynamic Objects](https://huggingface.co/datasets/scintigimcki/DynamicObjects)
- [Dynamic Indoor Scenes](https://huggingface.co/datasets/scintigimcki/DynamicIndoorScenes)
- [FreeGave-GoPro](https://huggingface.co/datasets/scintigimcki/FreeGave-GoPro)

## 🔑 Train
```
bash train_eval.sh
```
