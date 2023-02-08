import json
import os
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd



def expand_sentences(sents):
    return [sent['raw'] for sent in sents]


def sent_len(sent):
    return len(sent.split())


def prepare_df(base_dir, js_file_name):
    js_path = os.path.join(base_dir, js_file_name)
    with open(js_path, 'r') as js_file:
        rsicd = json.load(js_file)

    images = rsicd['images']
    df = pd.DataFrame(images)
    # df['sent1'], df['sent2'], df['sent3'], df['sent4'], df['sent5'] = zip(*df['sentences'].apply(expand_sentences))
    df['sent1'], df['sent2'], df['sent3'], df['sent4'], df['sent5'] = zip(*df['sentences'].apply(lambda sents: [sent['raw'] for sent in sents]))
    for i in range(1, 6):
        df[f'sent{i}_len'] = df[f'sent{i}'].apply(lambda sent: len(sent.split()))

    df['class'] = df['filename'].apply(lambda x: x.split('_')[0])
    df.drop(['sentences', 'sentids'], axis=1, inplace=True)
    df.to_csv(js_path.replace('.json', '.csv'), index=False)
    return df


def main():
    base_dir = '/home/asebaq/RSICD_optimal'
    js_file_name = 'dataset_rsicd.json'
    df = prepare_df(base_dir, js_file_name)
    df = pd.read_csv(os.path.join(base_dir, 'dataset_rsicd.csv'))
    for i in range(1, 6):
        print(f'sent{i}_len', df[f'sent{i}_len'].max())
    sent1 = df.sent1[0]
    print(sent1)
    print(len(sent1.split()))

    print(df['class'][df['class'].str.isalpha().values].value_counts())
    df['class'][df['class'].str.isalpha().values].value_counts().plot(kind='bar')
    plt.show()

    filename = df.filename[8551]
    print(filename)



if __name__ == '__main__':
    main()
