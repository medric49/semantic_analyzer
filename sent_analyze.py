import argparse
import sys

import nltk
import numpy as np
import torch
from gensim import models

import utils
from models import SentimentClassifier

sentence_size = 60


def analyze(model, wv, sentence):
    sentence = ' '.join(nltk.word_tokenize(sentence))
    sentences = [sentence.lower().split()]
    sentences, lengths = utils.vectorize_sentences(sentences, wv, sentence_size)
    sentences = np.array(sentences)
    sentences = torch.tensor(sentences, dtype=torch.float, device=utils.device)
    lengths = torch.tensor(lengths, dtype=torch.int, device=utils.device)

    output = model(sentences, lengths)
    output = torch.softmax(output, dim=1)
    return output[0, 1].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('sentence', type=str, help='Sentence to analyze ^^')

    args, _ = parser.parse_known_args(sys.argv[1:])

    root_dir = 'logs/sentiment_analysis'
    wv_model_file = root_dir + '/' + 'wv_bilstm.pth'
    wv = models.KeyedVectors.load(wv_model_file)

    wv.add_vectors(
        ['<unk>', '<eos>'],
        [np.zeros(wv.vector_size), np.ones(wv.vector_size)]
    )

    torch.no_grad()

    model = SentimentClassifier(vector_size=50, hidden_size=256, num_layers=2, bidirectional=True).to(utils.device)
    utils.load_model(model, 'logs/sentiment_analysis/lstm_model_v2.pth')
    model.eval()

    output = analyze(model, wv, args.sentence)
    print(f"Positive at {round(output*100, 2)}%")
