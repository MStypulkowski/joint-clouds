# Learning joint distibution of point clouds
## Setup
Dependencies are stored in `environment.yml`. To install, copy the repository and create the conda environment:
```
conda env create --file environment.yml
conda activate joint-clouds
```

Then, add path to the main folder to `PYTHONPATH`:
```
export PYTHONPATH='/path/to/the/joint-clouds'
```


We use [Hydra](https://github.com/facebookresearch/hydra) to manage all arguments for scripts. YAML configuration files are stored in `.\configs`.

## Data
### MNIST2D
Download [train](https://drive.google.com/file/d/133c5Im_J79L3clS-NbUJf4FTLLXQGQz_/view?usp=sharing) and [test](https://drive.google.com/file/d/1NqKPU-J84ugYkeDeg_f6tAQHi5d_g0fd/view?usp=sharing) datasets from authors' Google Drive.

## Scripts