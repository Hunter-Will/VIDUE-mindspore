# VIDUE-mindspore

### Introduction

This repository is the MindSpore implementation of VIDUE: [Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time](https://arxiv.org/abs/2303.15043).
Due to difference between MindSpore and Pytorch, this implementation does not have the exactly same behavior as the Pytorch implementation or the one described in paper. But the average difference between the outputs of this version and those of Pytorch version would be less than $1e-5$. So the precision of this implementation wouldn't have any difference with the Pytorch one. 
For any technical details of VIDUE, check original Pytorch version at [here](https://github.com/shangwei5/VIDUE/), it's also include several version of the paper. 

### Prerequisites
- Python >= 3.8, MindSpore >= 2.0.0
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm

### Datasets
Please download the GoPro datasets from [link](http://data.cv.snu.ac.kr:8008/webdav/dataset/GOPRO/GOPRO_Large_all.zip) (240FPS, GOPRO_Large_all)

Please download the Adobe datasets from [link](https://www.dropbox.com/s/pwjbbrcyk1woqxu/adobe240.zip?dl=0) (Full version)

## Dataset Organization Form
```
|--dataset
    |--train  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--test
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
         :
        |--video n
```
## Download Pre-trained Model of VIDUE-mindspore
Pre-trained exposure-aware feature extractor on GoPro and Adobe in MindSpore version and pre-trained VIDUE ckpt in MindSpore version can be downloaded from [here](https://pan.baidu.com/s/18homDHA7mI6tw7xZd1LJDw?pwd=mnxf).

Please put these models to `./experiment`.

## Getting Started

### 1) Generate Test Data
```
python generate_blur.py --videos_src_path your_data_path/GoPro_Large_all/test --videos_save_path your_data_path/GoPro_Large_all/LFR_Gopro_53  --num_compose 5  --tot_inter_frame 8
```
This is an example for generating "GoPro-5:8", please change `num_compose` and `tot_inter_frame` to generate other cases.

```
python generate_blur_adobe.py --videos_src_path your_data_path/adobe240/test --videos_save_path your_data_path/adobe240/LFR_Adobe_53  --num_compose 5*3  --tot_inter_frame 8*3
```
This is an example for generating "Adobe-5:8", please change `num_compose` and `tot_inter_frame` to generate other cases.


### 2) Testing
1.For testing the GoPro dataset ($\times$ 8 interpolation and deblurring):
```
python inference_vidue_worsu.py --default_data GOPRO --m 5(or 7) --n 3(or 1)
```
Check the python script to make sure you set the proper data path and ckpt path. Please change `args.data_path` according to `m` and `n`.
For other args, check `./code/options/template.py` to see or modify it to your needs.

2.For testing the Adobe dataset ($\times$ 8 interpolation and deblurring):
```
python inference_vidue_worsu.py --default_data Adobe --m 5(or 7) --n 3(or 1)
```

3.For testing the GoPro dataset ($\times$ 16 interpolation and deblurring):
```
python inference_vidue_worsu_16x.py --default_data GOPRO --m 9(or 11,13,15) --n 7(or 5,3,1)
```

4.For testing the real world dataset:
```
python inference_vidue_worsu_real.py
```
Change `args.model_path` to the path of our pre-trained models or you can finetune on your own dataset for testing real-world data.

### 3) Training
1.Training exposure-aware feature extractor:
```
python main_extractor_weighted_ordinalsupcon.py --template UNet_PRIOR_PREDICT_Weighted_OrdinalSupcon_Light  --save extractor_GoPro8x --process --random
```
Check default args in this python script and modify them to your needs. Please change `--template` to `UNet_PRIOR_PREDICT_Weighted_OrdinalSupcon_Light_Adobe` and `UNet_PRIOR_PREDICT_Weighted_OrdinalSupcon_Light_16x` for different tasks.

2.Training VIDUE:
```
python main_vidue_worsu_smph.py --template VIDUE_WORSU --save recon_GoPro8x --random --process
```
Please change `--template` to `VIDUE_WORSU_Adobe` and `VIDUE_WORSU_16x` for different tasks.
Please check the dataset path according to yours.



