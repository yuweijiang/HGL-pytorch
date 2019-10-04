import os
USE_IMAGENET_PRETRAINED = True # otherwise use detectron, but that doesnt seem to work?!?

# Change these to match where your annotations and images are
# VCR_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'vcr1images')
VCR_IMAGES_DIR = '/mnt/lustre21/yuweijiang/VCR/vcr1images'
VCR_ANNOTS_DIR = '/mnt/lustre21/yuweijiang/VCR/vcr1annots'
DATALOADER_DIR = '/mnt/lustre21/yuweijiang/code/r2c'
BERT_DIR = '/mnt/lustre21/yuweijiang/VCR/bert_presentations'


# VCR_ANNOTS_DIR = os.path.join(os.path.dirname(__file__), 'data')

if not os.path.exists(VCR_IMAGES_DIR):
    raise ValueError("Update config.py with where you saved VCR images to.")
