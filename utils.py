import pandas as pd


def transform_tsv_to_fasttext_format(tsv_file, output_file):
    csv_data = pd.read_csv(tsv_file, sep='\t', header=None)

    with open(output_file, 'w') as output_file:
        for i, data in csv_data.iterrows():
            output_file.write(f'__label__{data[1]} {data[3]}\n')
