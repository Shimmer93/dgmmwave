# dgmmwave

## Dependency Installation
1. Create a conda environment and activate it:
    ```
    conda create -n mm python=3.9
    conda activate mm
    ```
2. Install dependencies in `requirements.txt` (or `requirements_cu111.txt`):
    ```
    pip install -r requirements.txt
    ```
3. On the GPU node install pointnet2 for P4Transformer (For `requirements_cu111.txt`, 7.0 should be changed to 8.0):
    ```
    cd model/P4Transformer
    TORCH_CUDA_ARCH_LIST=7.0 python setup.py install
    ```

## Data Download and Preprocessing
### Raw Data
To be completed
### Preprocessed Data
1. Download [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zpengac_connect_ust_hk/EoCHAVUSomxOrWuWgbkUfPQBPrNUpGLXaTXg7_kXSlGVWw?e=tQ5Vhc)
2. Put the `.pkl` files under `data`

## Training and Testing
### Training
```
python main.py -g 4 -n 1 -w 5 -b 32 -e 20 --data_dir data --exp_name mmfi -c cfg/sample_mmfi.yml --version $(date +'%Y%m%d_%H%M%S')
```
### Testing
```
python main.py -g 4 -n 1 -w 5 -b 32 -e 20 --data_dir data --exp_name mmfi -c cfg/sample_mmfi.yml --version $(date +'%Y%m%d_%H%M%S') --checkpoint_path logs/mmfi/20240709_134217/p4t-epoch=145-val_mpjpe=0.1181.ckpt --test
```

## Experiments
### Lidar
```
python main.py -g 2 -n 1 -w 8 -b 64 -e 64 --exp_name hmpear -c cfg/exp/hmpear.yml --version $(date +'%Y%m%d_%H%M%S')
```
### Lidar + 10% mmWave
```
python main.py -g 2 -n 1 -w 8 -b 64 -e 64 --exp_name hmpear_mmfi -c cfg/exp/hmpear_mmfi.yml --version $(date +'%Y%m%d_%H%M%S')
```
### Lidar + 10% mmWave + Unsupervised
```
python main.py -g 2 -n 1 -w 8 -b 64 -e 64 --exp_name hmpear_mmfi_unsup -c cfg/exp/hmpear_mmfi_unsup.yml --version $(date +'%Y%m%d_%H%M%S') --ours
```