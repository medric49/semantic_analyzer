# Semantic analyzer
A semantic analyzer of sentences and questions based on Language processing algorithms (with Pytorch :) ).

### 1. 'Are the similar questions' problem
In this issue, we try to analyze similarity between two questions.
The dataset used here comes from the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) challenge.

Our solution with a pretrained [BERT](https://arxiv.org/abs/1810.04805) model built with Pytorch.

How does it work?
```shell
$ python3 analyze.py "Am I wrong?" "Do you love ice scream?"
Same at 0.15%
```

```shell
$ python3 analyze.py "How do I save videos from twitter?" "How do you upload videos from your camera roll onto Twitter?"
Same at 10.42%
```

```shell
$ python3 analyze.py "How do I save videos from twitter?" "How do you upload videos from your camera roll onto Twitter?"
Same at 97.04%
```

### 2. 'Is this sentence semantically correct' problem

To be continued...