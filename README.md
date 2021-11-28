# Semantic analyzer
A semantic analyzer of sentences and questions based on Language processing algorithms (with Pytorch :) ).

### 1. 'Are the questions similar?' problem
In this problem, we compare similarity between two questions.
The dataset used here comes from the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) challenge.

Our solution with a pretrained [BERT](https://arxiv.org/abs/1810.04805) model built with Pytorch.

How does it work?
```shell
python3 same_analyze.py "Am I wrong?" "Do you love ice scream?"
Same at 0.15%
```

```shell
python3 same_analyze.py "How do I save videos from twitter?" "How do you upload videos from your camera roll onto Twitter?"
Same at 10.42%
```

```shell
python3 same_analyze.py "How do I save videos from twitter?" "How do you upload videos from your camera roll onto Twitter?"
Same at 97.04%
```

Details are in the notebook `qqp_BERT.ipynb`

### 2. 'Is this comment positive?' problem
Here, we evaluate how positive is a comment sent.
The dataset used here to train the model come from [The Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) dataset

Our solution is a [BiLSTM](https://www.sciencedirect.com/science/article/abs/pii/S0893608005001206) model trained on a negative-positive classification task. We embed words with the Word2Vec [Gensim](https://radimrehurek.com/gensim/index.html) model trained with the [glove-wiki-gigaword-50](https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models) corpus.

How does it work?
```shell
python3 sent_analyze.py "I love this movie"
Positive at 100.0%
```

```shell
python3 sent_analyze.py "A great idea becomes a not-great movie."
Positive at 0.07%
```

Details are in the notebook `sentiment_analysis_BiLSTM_v2.ipynb`
