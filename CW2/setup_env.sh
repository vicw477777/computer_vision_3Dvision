## create environtment
source ~/.bashrc
ENV_NAME=ns
mamba create -n $ENV_NAME python=3.10 -y


## setup activation/deactivation scripts to audo-load/unload modules and env vars atuomatically when activating/deactivating the environment
mkdir -p $CONDA_PREFIX/envs/$ENV_NAME/etc/conda/activate.d/
cat <<EOF > $CONDA_PREFIX/envs/$ENV_NAME/etc/conda/activate.d/custom_activate.sh
export OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
module load cuda/12.1
export CUDA_HOME=/usr/local/software/cuda/12.1
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
EOF

mkdir -p $CONDA_PREFIX/envs/$ENV_NAME/etc/conda/deactivate.d/
cat <<EOF > $CONDA_PREFIX/envs/$ENV_NAME/etc/conda/deactivate.d/custom_deactivate.sh
module unload cuda/12.1
unset CUDA_HOME
export LD_LIBRARY_PATH="\$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
EOF


## activate environment
mamba activate $ENV_NAME


## install PyTorch
mamba install pytorch==2.5.1=py3.10_cuda12.1_cudnn9.1.0_0 torchvision==0.20.1=py310_cu121   torchaudio==2.5.1=py310_cu121 pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y


## clone Pi3, MoGe and install dependencies
mkdir -p external
cd external
git clone https://github.com/CambridgeCVCourses/Pi3.git
git clone https://github.com/CambridgeCVCourses/MoGe.git
cd ..
pip install numpy==1.26.4 pillow opencv-python plyfile huggingface_hub safetensors gradio trimesh matplotlib scipy
pip install git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183


## install tiny-cuda-nn using this pre-built wheel
pip install ./wheels/tinycudann-2.0-cp310-cp310-linux_x86_64.whl
## or build from source
# git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
# cd tiny-cuda-nn/bindings/torch
# pip install "setuptools<70.0.0"
# python setup.py install
# cd ../../../


## install colmap (CPU-only), nerfstudio, and gsplat
mamba install colmap=3.8.0=cpuhc4e8ae7_23 -y
pip install nerfstudio
#pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
pip install gsplat

## install SAM2
# cd external
# git clone https://github.com/facebookresearch/sam2.git
# cd sam2
# pip install -e .
# cd ../..
