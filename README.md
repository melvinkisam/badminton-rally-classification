# badminton-rally-classification

A system where users can upload full match videos and receive a compilation of segmented clips categorised by key events, such as rallies, shuttle changes, player breaks, and umpire interactions, while also removing dead time where no meaningful activity occurs.

![Example](img/correct.gif)

## Dataset Creation Procedure
1. **Collect match data** (`.mp4` format)  
2. **Run** `blockScore.py` to pseudonymise players  
3. **Run** `tagVideo.py` to annotate data  
4. **Run** `generateClips.py` to generate clips and their labels  
5. **Run** `generateVideoInfo.py` to generate video information as a `.txt` file  

## Model Training
There are two versions, 1 and 2. Refer to version 2 for the latest approach.
- **Run** `train_model2.py` to train model on training set
- **Run** `test_model2.py` to perform inference on validation set

## Automated Clip Generation
- **Run** `match_classification.py` to segment a full match to defined classes (rally, not rally, shuttle change, floor mopping, set break)
- **Run** `match_classification_text.py` to classify a full match to defined classes and output a full match with predictions

## Installation

### 1. Clone repository
```
git clone https://github.com/melvinkisam/badminton-rally-classification.git
```
### 2. Create virtual environment (if needed)
```
python -m venv venv
```
Then activate:
```
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### 3. Install dependencies
```
pip install -r requirements.txt
```

## GluonCV Deprecated Version Notes

Due to a deprecated GluonCV version, some adjustments are needed.

### ERROR FIXING

### 1. `multigrid_helper.py`

File: `~/python3.10/site-packages/gluoncv/torch/data/video_cls/multigrid_helper.py`

Replace:

```
from torch._six import int_classes as _int_classes
```

with:

```
_int_classes = (int, )
```

### 2. `transform.py`

File: `~/python3.10/site-packages/gluoncv/torch/data/transforms/instance_transforms/transform.py`

At line 876, update:

```
interp=Image.LINEAR
```

to:

```
interp=Image.BILINEAR
```

---

### Fix for Error: `ImportError: cannot import name 'int_classes' from 'torch._six'`

Edit the file:

```
nano ~/miniconda3/envs/CVInBadminton/lib/python3.9/site-packages/gluoncv/torch/data/video_cls/multigrid_helper.py
```

Replace:

```
from torch._six import int_classes as _int_classes
```

with:

```
_int_classes = (int, )
```

Clear cache:

```
find ~/miniconda3/envs/CVInBadminton/ -name "__pycache__" -type d -exec rm -rf {} +
find ~/miniconda3/envs/CVInBadminton/ -name "*.pyc" -delete
```

---

### HPC Cluster Conda Environment Installation

(Older CUDA to bypass glibc issue)

Using conda:

```
conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cudatoolkit=10.2 -c pytorch
```

Or using pip:

```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```
