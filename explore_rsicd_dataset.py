import json
import os
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
from transformers import T5Tokenizer, T5Model

tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=256)
model = T5Model.from_pretrained('t5-small')


def expand_sentences(sents):
    return [sent['raw'] for sent in sents]


def embed_sentences(sent):
    input_ids = tokenizer(
        sent, return_tensors='pt').input_ids  # Batch size 1
    decoder_input_ids = tokenizer(
        'Studies show that', return_tensors='pt').input_ids  # Batch size 1

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = model._shift_right(decoder_input_ids)

    # forward pass
    outputs = model(input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0]


def sent_len(sent):
    return len(sent.split())


def prepare_df(base_dir, js_file_name):
    js_path = os.path.join(base_dir, js_file_name)
    with open(js_path, 'r') as js_file:
        rsicd = json.load(js_file)

    images = rsicd['images']
    df = pd.DataFrame(images)
    # df['sent1'], df['sent2'], df['sent3'], df['sent4'], df['sent5'] = zip(*df['sentences'].apply(expand_sentences))
    df['sent1'], df['sent2'], df['sent3'], df['sent4'], df['sent5'] = zip(
        *df['sentences'].apply(lambda sents: [sent['raw'] for sent in sents]))

    for i in range(5):
        df[f'sent{i+1}_emb_t5'] = df[f'sent{i+1}'].apply(embed_sentences)

    for i in range(1, 6):
        df[f'sent{i}_len'] = df[f'sent{i}'].apply(lambda sent: len(sent.split()))

    df['class'] = df['filename'].apply(lambda x: x.split('_')[0])
    df.drop(['sentences', 'sentids'], axis=1, inplace=True)
    df.to_csv(js_path.replace('.json', '.csv'), index=False)
    return df


def main():
    tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=256)
    model = T5Model.from_pretrained('t5-small')
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
