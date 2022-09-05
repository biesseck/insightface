## MANUAL INSTALL

```bash
conda create -n pbidr python=3.7
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge scikit-image shapely rtree pyembree
pip install trimesh[all]
```

#### Install Pytorch3D

```bash
wget https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
wget https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.4.0.tar.gz
tar xzf v0.4.0.tar.gz
export TORCH_CUDA_ARCH_LIST="7.5"
cd pytorch3d-0.4.0 && pip install -e .
```

#### Install Other Requirments

```bash
pip install -r requirements.txt
```




```
# BERNARDO
conda create -y -n bjgbiesseck_pbidr python=3.8 && \
conda activate bjgbiesseck_pbidr && \
pip install torch torchvision && \
conda install -y -c conda-forge scikit-image shapely rtree pyembree && \
pip install trimesh[all]
```

```
# BERNARDO  #### Install Pytorch3D
conda install -y -c conda-forge cub && \
# conda install -y -c pytorch3d pytorch3d && \
pip install pytorch3d

# export CUDA_PATH=/usr/local/cuda && \
# export CUDA_PATH=/usr/local/cuda-10.1 && \
export CUDA_PATH=/usr/local/cuda11.1 && \
export PATH=$CUDA_PATH/bin:$CUDA_PATH/lib64:$PATH && \
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```


torch
(from versions: 1.4.0, 1.4.0+cpu, 1.4.0+cu100, 1.4.0+cu92, 1.5.0, 1.5.0+cpu, 1.5.0+cu101, 1.5.0+cu92, 1.5.1, 1.5.1+cpu, 1.5.1+cu101, 1.5.1+cu92, 1.6.0, 1.6.0+cpu, 1.6.0+cu101, 1.6.0+cu92, 1.7.0, 1.7.0+cpu, 1.7.0+cu101, 1.7.0+cu110, 1.7.0+cu92, 1.7.1, 1.7.1+cpu, 1.7.1+cu101, 1.7.1+cu110, 1.7.1+cu92, 1.7.1+rocm3.7, 1.7.1+rocm3.8, 1.8.0, 1.8.0+cpu, 1.8.0+cu101, 1.8.0+cu111, 1.8.0+rocm3.10, 1.8.0+rocm4.0.1, 1.8.1, 1.8.1+cpu, 1.8.1+cu101, 1.8.1+cu102, 1.8.1+cu111, 1.8.1+rocm3.10, 1.8.1+rocm4.0.1, 1.9.0, 1.9.0+cpu, 1.9.0+cu102, 1.9.0+cu111, 1.9.0+rocm4.0.1, 1.9.0+rocm4.1, 1.9.0+rocm4.2, 1.9.1, 1.9.1+cpu, 1.9.1+cu102, 1.9.1+cu111, 1.9.1+rocm4.0.1, 1.9.1+rocm4.1, 1.9.1+rocm4.2, 1.10.0, 1.10.0+cpu, 1.10.0+cu102, 1.10.0+cu111, 1.10.0+cu113, 1.10.0+rocm4.0.1, 1.10.0+rocm4.1, 1.10.0+rocm4.2, 1.10.1, 1.10.1+cpu, 1.10.1+cu102, 1.10.1+cu111, 1.10.1+cu113, 1.10.1+rocm4.0.1, 1.10.1+rocm4.1, 1.10.1+rocm4.2, 1.10.2, 1.10.2+cpu, 1.10.2+cu102, 1.10.2+cu111, 1.10.2+cu113, 1.10.2+rocm4.0.1, 1.10.2+rocm4.1, 1.10.2+rocm4.2, 1.11.0, 1.11.0+cpu, 1.11.0+cu102, 1.11.0+cu113, 1.11.0+cu115, 1.11.0+rocm4.3.1, 1.11.0+rocm4.5.2, 1.12.0, 1.12.0+cpu, 1.12.0+cu102, 1.12.0+cu113, 1.12.0+cu116, 1.12.0+rocm5.0, 1.12.0+rocm5.1.1, 1.12.1, 1.12.1+cpu, 1.12.1+cu102, 1.12.1+cu113, 1.12.1+cu116, 1.12.1+rocm5.0, 1.12.1+rocm5.1.1)
