wget https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip
wget https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip
ns-download-data --dataset=blender

unzip Synthetic_NSVF.zip
unzip TanksAndTemple.zip
unzip nerf_synthetic.zip