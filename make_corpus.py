

if __name__ == '__main__':

    root_dir = '1g-word-1m-benchmark-r13output/training-monolingual.tokenized.shuffled'
    corpus_root = 'corpus'

    i = 5

    new_corpus_file = f'{corpus_root}/corpus.txt'
    new_corpus = open(new_corpus_file, 'a')

    for j in range(1, i + 1):
        num = str(j).zfill(5)
        corpus_file = f'{root_dir}/news.en-{num}-of-00100'
        with open(corpus_file, 'r') as corpus_file:
            corpus = corpus_file.read()
        new_corpus.write(corpus)
    new_corpus.close()
