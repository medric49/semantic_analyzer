import pandas as pd
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def transform_tsv_to_fasttext_format(tsv_file, output_file):
    csv_data = pd.read_csv(tsv_file, sep='\t', header=None)

    with open(output_file, 'w') as output_file:
        for i, data in csv_data.iterrows():
            output_file.write(f'__label__{data[1]} {data[3]}\n')


def save_checkpoint(save_path, model, valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name):
    return model.load_state_dict(torch.load(file_name))


def vectorize_sentences(sentences, wv, sentence_size):
    voc = wv.key_to_index.keys()
    unk = wv['<unk>']
    eos = wv['<eos>']
    lengths = []
    for i, sentence in enumerate(sentences):
        lengths.append(len(sentence))

        for i, token in enumerate(sentence):
            if token in voc:
                sentence[i] = wv[token]
            else:
                sentence[i] = unk

        while len(sentence) < sentence_size:
            sentence.append(eos)

    return sentences, lengths
