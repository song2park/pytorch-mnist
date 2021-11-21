# pytorch-mnist
MNIST handwriting data classification with pytorch

## Dataset
from [Kaggle MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)

## Structure
```
-root
    +digit-recognizer
    |   ├sample_submission.csv : submission template
    |   ├test.csv : test dataset
    |   └train.csv : train dataset
    |
    └train.py : train code
    └test.py : test code
    └customclasses.py : custom dataset class, custom model class pkg
```

## Usage
```bash
python train.py
python test.py
```