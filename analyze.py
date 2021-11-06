import argparse
import sys
from sys import stdin
import torch
from transformers import BertTokenizer
import utils
from models import BERT


def analyze(model, tokenizer, question1, question2):
    question = torch.tensor(tokenizer.encode(question1 + '|||' + question2), dtype=torch.long, device=utils.device).unsqueeze(0)
    label = torch.tensor(0, dtype=torch.long, device=utils.device).unsqueeze(0)
    _, output = model(question, label)
    output = torch.softmax(output, dim=1)
    return output[0, 1].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('question1', type=str, help='Glossary file where to find right words.')
    parser.add_argument('question2', type=str, help='Glossary file where to find right words.')

    args, _ = parser.parse_known_args(sys.argv[1:])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    torch.no_grad()

    best_model = BERT().to(utils.device)
    utils.load_checkpoint('logs/qqp/model_bert.pt', best_model)
    best_model.eval()

    output = analyze(best_model, tokenizer, args.question1, args.question2)
    print('')
    print(f"Same at {round(output*100, 2)}%")
