# Test Time Adaptation of Medical VLMs

We present the first structured benchmark for test-time adaptation of medical VLMs, exploring strategies tailored to the unique challenges of medical imaging.



## Requirements 
- [Python 3.9](https://www.python.org/)
- [CUDA 11.7](https://developer.nvidia.com/cuda-zone)
- [PyTorch 2.0.1](https://pytorch.org/)


## Usage
### Step 1: Clone the repository


### Step 2: Setup the Environment
Create an environment and Install the requirements The `environment.yaml` file can be used to install the required dependencies:

```bash
cd TTAMedVLM
conda env create -f environment.yml
```

### Step 3: Prepare Datasets and Models

Download NCT, L25000, sicamp_mil datasets for histology and messidor, odir, fives for Retina.

Clone Open_clip and ViLReF foundation models. 

### Step 4: Adaptation



```bash
# dataset configuration
DATASET=nct   
DATA_DIR=/path/to/data/

# adaptation parameters
BATCH_SIZE=128   
LR=1e-3
BACKBONE=ViT-B/32
Model=Quilt


# Execute the adaptation process with specified parameters
python main.py --data_dir $DATA_DIR --dataset $DATASET --adapt --method $METHOD --save_dir ./save --backbone $BACKBONE --batch-size $BATCH_SIZE --lr $LR 

```



## License

This source code is released under the MIT license, which can be found [here](./LICENSE).

This project incorporates components from the following repositories. We extend our gratitude to the authors for open-sourcing their work:
- [WATT](https://github.com/Mehrdad-Noori/WATT)
- [Tent](https://github.com/DequanWang/tent) (MIT licensed)
- [CLIP](https://github.com/openai/CLIP/tree/main/clip) (MIT licensed)
- [CLIPArTT](https://github.com/dosowiechi/CLIPArTT)
