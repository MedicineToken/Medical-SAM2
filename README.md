<h1 align="center">● Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2</h1>

<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

Medical SAM 2, or say MedSAM-2, is an advanced segmentation model that utilizes the [SAM 2](https://github.com/facebookresearch/segment-anything-2) framework to address both 2D and 3D medical
image segmentation tasks. This method is elaborated on the paper [Medical SAM 2: Segment Medical Images As Video Via Segment Anything Model 2](https://arxiv.org/abs/2408.00874).

## 🔥 A Quick Overview 
 <div align="center"><img width="880" height="350" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/framework.png"></div>
 
## 🩻 3D Abdomen Segmentation Visualisation
 <div align="center"><img width="420" height="420" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/example.gif"></div>

## 🧐 Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate medsam2``

 You can download SAM2 checkpoint from checkpoints folder:
 
 ``bash download_ckpts.sh``

 ## 🎯 Example Cases
 #### Download REFUGE or BCTV or youn own dataset and put in the ``data`` folder, creat the folder if it does not exist ⚒️
 
 ### 2D case - REFUGE Optic-cup Segmentation from Fundus Images

**Step1:** Dowaload [REFUGE](https://refuge.grand-challenge.org/) dataset manually from [here](https://huggingface.co/datasets/jiayuanz3/REFUGE/tree/main), or using command lines:

 ``wget https://huggingface.co/datasets/jiayuanz3/REFUGE/resolve/main/REFUGE.zip``

 ``unzip REFUGE.zip``

 **Step2:** Run the training and validation by:
``python train_2d.py -net sam2 -exp_name REFUGE_MedSAM2 -vis 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -out_size 1024 -b 4 -val_freq 1 -dataset REFUGE -data_path ./data/REFUGE``

 ### 3D case
 ``python train_3d.py -net sam2 -exp_name BTCV_MedSAM2 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 1 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv``


## 🚨 News
- 24-08-05. Update 2D example details and REFUGE dataset download link 🔗
- 24-08-05. Our Medical SAM 2 paper was available online 🥳
- 24-08-05. Our Medical SAM 2 code was available on Github 🥳
- 24-07-30. The SAM 2 model was released 🤩

## 📝 Cite
 ~~~
@misc{zhu_medical_2024,
	title={Medical SAM 2: Segment medical images as video via Segment Anything Model 2},
    author={Jiayuan Zhu and Yunli Qi and Junde Wu},
    year = {2024},
    eprint={2408.00874},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
 ~~~