# MLMI17: Advanced Computer Vision - Coursework 2

Codebase for Coursework 2 on 3D Computer Vision. Instructor: [Elliott Wu](https://elliottwu.com/).

## Environment Setup
You will need to use the Cambridge HPC to run the expriments. These experiments require extensive GPU compute. It is very unlikely that you will be able to complete them on your laptop or with a small GPU. Follow the instructions to set up the environment on HPC.

The code has been tested on Cambridge HPC with NVIDIA A100 GPUs, CUDA 12.1, and PyTorch 2.5.1. You may check `environment.yml` for the specific package versions, but note that simply running `mamba env create -f environment.yml` will not work as it requires some custom built packages.


### 1.0 Install Mamba
I use [mamba](https://mamba.readthedocs.io/en/latest/index.html) to maintain Python environments. It is a drop-in replacement for [conda](https://docs.conda.io/projects/conda/en/latest/index.html) that is usually much faster at solving environment dependencies. You may use `conda` as well if you already have `conda`; simply replace `mamba` with the regular `conda` in the following commands.

To install mamba, first, we need to log into Cambridge HPC. If you have not done this before, follow the [instructions](https://docs.hpc.cam.ac.uk/hpc/user-guide/quickstart.html) to set up your HPC account. You can log into the HPC using this command in your terminal with your `[CRSID]`:
```
ssh [CRSID]@login-icelake.hpc.cam.ac.uk
```

Once you have logged into the HPC, download and install `mamba` using the following command:
```
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3.sh
```


### 1.1 Install Python Environment
This codebase depends on a number of Python packages, including [PyTorch](https://pytorch.org/), [COLMAP](https://colmap.github.io/index.html), [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [Pi3](https://github.com/yyfz/Pi3), and [SAM2](https://github.com/facebookresearch/sam2). We will install the Python environment using `mamba` or `conda`. You will need to first clone this repo:
```
git clone https://github.com/CambridgeCVCourses/CW2.git
```

You need a GPU session to set up the environment. It may be easier to do so using an interactive session. Once you have logged into the HPC, to request an interactive session, use this command with your assigned project code (which might be `MLMI-[CRSID]-SL2-GPU` as one of the students suggested):
```
sintr -t 2:0:0 -A [YOUR_ASSIGNED_PROJECT] --nodes=1 --gres=gpu:1 -p ampere
```
`-t 2:0:0` means to request a session for 2 hours.

Once you acquired an interactive session, simply run `setup_env.sh` to set up the environment:
```
bash setup_env.sh
```

Be aware that this can take **roughly 30 minutes**. Once the script has fully completed, your Python environment should be ready to use by:
```
mamba activate ns
```
You may see a few warnings about `ERROR: pip's dependency resolver` throughout the installation process, but you may ignore them. Feel free to open an issue if you encounter other issues.

### 1.2 Tips on Interactive GPU Sessions
You're highly recommended to use [tmux](https://tmuxcheatsheet.com/), which is a terminal multiplexer and is already pre-installed on the HPC. It allows you to create multiple terminal panes within a single terminal window. More importantly, these tmux sessions will remain active even if you disconnect from the HPC. This is particularly useful if you would like to keep your interactive GPU session active without maintaining an active connection to the HPC.

But you will notice that there are several [different login nodes](https://docs.hpc.cam.ac.uk/hpc/user-guide/connecting.html) `login-q-[1,2,3,4]` and you may be taken to a different node each time you run `ssh login-icelake.hpc.cam.ac.uk`. Since `tmux` sessions are tied to the specific login node on which they were created, you will need to reconnect to that same node to retrieve them. You can do so by a second `ssh`. For example, if your session was started on `login-q-1`, you can access it by simply running `ssh login-q-1` from the login node you were initially assigned to.

Another common use case of `tmux` is to open multiple terminals attached to the same interactive session on a specific GPU node. Once you have been allocated an interactive session on a GPU node (eg, `gpu-q-1`), you may want additional terminals connected to that same node, for instance, to run visualisation tools alongside model training. You can do this by creating a new `tmux` pane on the login node and then running `ssh gpu-q-1` from there.


## Task 1 – Camera Pose Estimation and Sparse Reconstruction
Task 1 involves running COLMAP and Pi3 to estimate camera poses and reconstruct sparse 3D point clouds from a video, and compare their results. See the Coursework PDF for the exact tasks. Instructions on how to run COLMAP and Pi3 are provided here.

### Running COLMAP
To run COLMAP on a video (eg, the provided example `data/static_scene.mp4`), simply use `ns-process-data` tool provided in `nerfstudio`, eg:
```
ns-process-data video --data data/static_scene.mp4 --output-dir results/colmap/static_scene --no-gpu
```
Remember to use the `--no-gpu` as we have only installed a CPU version of COLMAP.

### Visualisation
#### Using Viser
Once COLMAP has successfully estimated the camera poses, we can use [Viser](https://viser.studio/main/) to visualise the results. To initiate the visualiser, first run the following command on the server:
```
python visualise_poses.py --scene-path results/colmap/static_scene
```
This will start a webviewer on a localhost port (defaulting to `8080`) on the server. We then need to forward the port to your local computer so that you can access the webviewer from your local browser.

To do so, first, identify which GPU node you're using on the HPC (eg, `gpu-q-1`). Then, open a new terminal on your **local computer** and run the following command from your local computer:
```
ssh -L 8080:gpu-q-1:8080 [CRSID]@login-icelake.hpc.cam.ac.uk
```
You will then be able to access the viewer by opening `http://localhost:8080` in your local browser.

#### Using MeshLab
Another light-weight 3D visualisation tool is [MeshLab](https://www.meshlab.net/), which allows you to visualise 3D point clouds, meshes, etc. locally. To visualise the reconstructed sparse point cloud from COLMAP, simply drag the `.ply` file into MeshLab.

### Running Pi3
In Task 1.2, you are required to run Pi3 to estimate camera poses and compare the results to COLMAP's predictions. To run Pi3, an incomplete inference script is provided in `pi3_inference.py`. You are required to complete two small implementations inside the `write_transforms_json` function, indicated by `## YOUR CODE HERE`, which converts Pi3's predictions into the same format produced by `ns-process-data`.

Once you complete the implementation, simply run:
```
python pi3_inference.py --data_path data/static_scene.mp4 --save_path results/pi3
```
Replace the `--data_path` with the path to your own videos.

Once camera estimation has completed, you can visualise the results in the same way as the COLMAP visualisation described above.


## Task 2 – Novel View Synthesis
Task 2 involves training Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) models to render novel views of the captured 3D scenes.

### Training NeRF
To train a NeRF model, we will use `nerfstudio`'s implementation (`nerfacto`). Once the initial camera poses have been estiamted using either COLMAP or Pi3, simply run:
```
ns-train nerfacto --data results/colmap/static_scene --output-dir results/nerf/static_scene_colmap
```
A webviewer is automatically initiated during training, hosted on the default port `7007`. You can forward the port in the same way as described above to visualise the training on your local browser.

Alternatively, you can also launch a separate viewer by running the following command on the server, providing the config file generated in the training:
```
ns-viewer --load-config result/nerf/static_scene_colmap/static_scene/nerfacto/.../config.yml
```

### Training 3DGS
Similarly, you can train a 3DGS model (`splatfacto`) by running:
```
ns-train splatfacto --data results/colmap/static_scene --output-dir results/3dgs/static_scene_colmap
```
You can visualise the training in the same way above.


## Acknowledgements
This codebase was developed with the help from [Xiaoyang Lyu](https://shawlyu.github.io/). Remember to include relevant citations in your report. This codebase is built upon the following foundational works.

COLMAP:
```
@inproceedings{schoenberger2016sfm,
    author      = {Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
    title       = {Structure-from-Motion Revisited},
    booktitle   = {CVPR},
    year        = {2016}
}

@inproceedings{schoenberger2016mvs,
    author      = {Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
    title       = {Pixelwise View Selection for Unstructured Multi-View Stereo},
    booktitle   = {ECCV},
    year        = {2016}
}
```

Pi3:
```
@InProceedings{wang2025pi,
    title       = {$\pi^3$: Permutation-Equivariant Visual Geometry Learning},
    author      = {Wang, Yifan and Zhou, Jianjun and Zhu, Haoyi and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Pang, Jiangmiao and Shen, Chunhua and He, Tong},
    booktitle   = {ICLR},
    year        = {2026}
}
```

NeRF:
```
@inproceedings{mildenhall2020nerf,
    title       = {{NeRF}: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author      = {Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    booktitle   = {ECCV},
    year        = {2020}
}
```

3DGS:
```
@Article{kerbl3Dgaussians,
    author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal      = {TOG},
    number       = {4},
    volume       = {42},
    year         = {2023}
}
```

nerfstudio:
```
@inproceedings{nerfstudio,
	title        = {{Nerfstudio}: A Modular Framework for Neural Radiance Field Development},
	author       = {Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin, Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa, Angjoo},
	booktitle    = {SIGGRAPH},
	year         = 2023
}
```

Viser:
```
@article{yi2025viser,
    title       = {{Viser}: Imperative, web-based 3d visualization in python},
    author      = {Yi, Brent and Kim, Chung Min and Kerr, Justin and Wu, Gina and Feng, Rebecca and Zhang, Anthony and Kulhanek, Jonas and Choi, Hongsuk and Ma, Yi and Tancik, Matthew and Kanazawa, Angjoo},
    journal     = {arXiv preprint arXiv:2507.22885},
    year        = {2025}
}
```
