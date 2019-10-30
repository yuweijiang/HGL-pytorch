# Heterogeneous Graph Learning for Visual Commonsense Reasoning (NeurlPS 2019)

This repository contains data and PyTorch code for the paper [**Heterogeneous Graph Learning for Visual Commonsense Reasoning** (arxiv)](https://arxiv.org/abs/1910.11475).

This repo should be ready to replicate my results from the paper. If you have any issues with getting it set up though, please file a github issue. Still, the paper is just an arxiv version, so there might be more updates in the future.

This repository contains trained models and PyTorch version code for the above paper, If the paper significantly inspires you, we request that you cite our work:
#### Bibtex

```
@inproceedings{yu2019heterogeneous,
  title={Heterogeneous Graph Learning for Visual Commonsense Reasoning},
  author={Weijiang Yu and Jingwen Zhou and Weihao Yu and Xiaodan Liang and Nong Xiao},
  journal={Advances in Neural Information Processing Systems (NeurlPS)},
  year={2019}
}
```

# Background

This repository is for the new task of [Visual Commonsense Reasoning](https://visualcommonsense.com). A model is given an image, objects, a question, and four answer choices. The model has to decide which answer choice is correct. Then, it's given four rationale choices, and it has to decide which of those is the best rationale that explains *why its answer is right*.

In particular, I have code and checkpoints for the HGL model, as discussed in the [HGL paper](https://arxiv.org/abs/1910.11475).  Here's a diagram that explains what's going on:

We'll treat going from Q->A and QA->R as two separate tasks: in each, the model is given a 'query' (question, or question+answer) and 'response choices' (answer, or rationale). Essentially, we'll use BERT and detection regions to *ground* the words in the query, then *contextualize* the query with the response. We'll perform several steps of *reasoning* on top of a representation consisting of the response choice in question, the attended query, and the attended detection regions. See the paper for more details.

# Setup

* Get the dataset. Follow the steps in `data/README.md`. This includes the steps to get the pretrained BERT embeddings. You might find the dataloader useful (in dataloaders/vcr.py), as it handles loading the data in a nice way using the allennlp library.

* Install cuda 9.0 if it's not available already. You might want to follow this [this guide](https://medium.com/@zhanwenchen/install-cuda-9-2-and-cudnn-7-1-for-tensorflow-pytorch-gpu-on-ubuntu-16-04-1822ab4b2421) but using cuda 9.0. I use the following commands (my OS is ubuntu 16.04):
```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
chmod +x cuda_9.0.176_384.81_linux-run
./cuda_9.0.176_384.81_linux-run --extract=$HOME
sudo ./cuda-linux.9.0.176-22781540.run
sudo ln -s /usr/local/cuda-9.0/ /usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/
```

* Install [anaconda](https://www.anaconda.com/) if it's not available already, and create a new environment. You need to install a few things, namely, pytorch 1.0, torchvision (*from the layers branch, which has ROI pooling*), and allennlp.

```
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
conda update -n base -c defaults conda
conda create --name hgl python=3.6
source activate hgl

conda install numpy pyyaml setuptools cmake cffi tqdm pyyaml scipy ipython mkl mkl-include cython typing h5py pandas nltk spacy numpydoc scikit-learn jpeg

conda install pytorch cudatoolkit=9.0 -c pytorch
pip install git+git://github.com/pytorch/vision.git@24577864e92b72f7066e1ed16e978e873e19d13d

pip install -r allennlp-requirements.txt
pip install --no-deps allennlp==0.8.0
python -m spacy download en_core_web_sm


# this one is optional but it should help make things faster
pip uninstall pillow && CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
```

* If you don't want to download from scratch, then download my [checkpoint](https://drive.google.com/drive/folders/1ux9YG3sRmUVvsCt1nHwlB5Egw43NDsK1?usp=sharing).

* That's it! Now to set up the environment, run `source activate hgl && export PYTHONPATH=/home/yuwj/code/hgl` (or wherever you have this directory).

# Train & Val
### Setting up the configuration

```
VCR_IMAGES_DIR = '/home/yuwj/VCR_dataset/vcr1images' # directory of images
VCR_ANNOTS_DIR = '/home/yuwj/VCR_dataset/vcr1annots' # directory of annotations
DATALOADER_DIR = '/home/yuwj/code/hgl' # directory of project
BERT_DIR = '/home/yuwj/VCR_dataset/bert_presentations' # directory of bert embedding
```

You can train a model using `train.py`. This also has code to obtain model predictions. Use `eval_q2ar.py` to get validation results combining Q->A and QA->R components. You can submit to the [leaderboard](https://visualcommonsense.com/leaderboard/) using the script in `eval_for_leaderboard.py`.

### Train
- For question answering, run:
```
python train.py -params multiatt/default.json -folder answer_save -train -val
```

- for Answer justification, run
```
python train.py -params multiatt/default.json -folder reason_save -train -val -rationale
```

### Val
You can combine the validation predictions using
```
python eval_q2ar.py -answer_preds answer_save/valpreds.npy -rationale_preds reason_save/valpreds.npy
```
### Submitting to the leaderboard
```
python eval_for_leaderboard.py -params models/multiatt/default.json -answer_ckpt answer_save/best.th -rationale_ckpt reason_save/best.th -output submisson.csv
```

## Help

Feel free to open an issue if you encounter trouble getting it to work!

# Acknowledgement
Thank [@rowanz](https://github.com/rowanz) for his generously releasing nice code [r2c](https://github.com/rowanz/r2c).
