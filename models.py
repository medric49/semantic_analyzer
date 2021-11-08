import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertForSequenceClassification


class BERTQuestionAnalyzer(nn.Module):

    def __init__(self):
        super(BERTQuestionAnalyzer, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea


class SentimentClassifier(nn.Module):

    def __init__(self, vector_size, hidden_size, num_layers, bidirectional):
        super(SentimentClassifier, self).__init__()
        lstm_dim = hidden_size * 2 * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(input_size=vector_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True
                            )
        self.dropout = nn.Dropout(p=0.5)

        self.fcnn_1 = nn.Linear(in_features=lstm_dim, out_features=64)

        self.fcnn_2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, sentences, lengths):
        sentences = pack_padded_sequence(sentences, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h_lstm, _ = self.lstm(sentences)
        output, _ = pad_packed_sequence(h_lstm, batch_first=True)

        avg_pool = torch.mean(output, 1)
        max_pool, _ = torch.max(output, 1)

        output = torch.cat([avg_pool, max_pool], 1)
        output = self.dropout(output)

        output = self.fcnn_1(output)
        output = torch.relu(output)

        output = self.fcnn_2(output)

        return output
