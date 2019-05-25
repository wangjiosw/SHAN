# SHAN


Keras implementation of [Syntax-Directed Hybrid Attention Network for Aspect-Level Sentiment Analysis](https://ieeexplore.ieee.org/document/8561296)

## Prerequisites
1. keras
2. spacy (pip install -U spacy)
3. pyenchant （pip install pyenchant）
4. python 2.7
5. nltk (pip install nltk)
6. download pretrained glove [glove.840B.300d.zip](https://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and unzip it to dir *GLOVE_MODEL*

## Preprocess

run preprocess.ipynb to preprocess data

## Usage

For training, `python Main.py train`
<br>
For testing, `python Main.py test`

## Result

### train data
set validation_split = 0.2
- 80% data training
- 20% data validating

categorical_accuracy:  87.234%
<br>
val_categorical_accuracy: 74.064%

### test data
accuracy: 0.84375

## Pretrained model
[no pretrained model](wwww.github.com)


## References
[Syntax-Directed Hybrid Attention Network for Aspect-Level Sentiment Analysis](https://ieeexplore.ieee.org/document/8561296)

