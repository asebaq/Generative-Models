from transformers import BertTokenizer, BertModel
from transformers import T5Tokenizer, T5Model

import torch
import matplotlib.pyplot as plt


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    sent1 = 'many planes are parked next to a long building in an airport .'
    marked_text = "[CLS] " + sent1 + " [SEP]"

    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Display the words with their indeces.
    for tup in zip(tokenized_text, indexed_tokens):
        print('{:<12} {:>6,}'.format(tup[0], tup[1]))

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    print(segments_ids)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)

        # Evaluating the model will return a different number of objects based on
        # how it's  configured in the `from_pretrained` call earlier. In this case,
        # becase we set `output_hidden_states = True`, the third item will be the
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]


    # `hidden_states` is a Python list.
    print('      Type of hidden_states: ', type(hidden_states))
    # Each layer in the list is a torch tensor.
    print('Tensor shape for each layer: ', hidden_states[0].size())
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    print(token_embeddings.size())

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    print(token_embeddings.size())

    token_embeddings_sum = torch.zeros((token_embeddings.shape[0], token_embeddings.shape[-1]))
    for i in range(token_embeddings.shape[0]):
        print(i, token_embeddings[i, -4:].shape, torch.sum(token_embeddings[i, -4:], dim=0).shape)
        token_embeddings_sum[i] = torch.sum(token_embeddings[i, -4:], dim=0)
    print(token_embeddings_sum.shape)

    # TODO: Pad T5 embeddings
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5Model.from_pretrained("t5-small")

    input_ids = tokenizer(sent1, return_tensors="pt").input_ids  # Batch size 1
    decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

    # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
    # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
    decoder_input_ids = model._shift_right(decoder_input_ids)

    # forward pass
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    last_hidden_states = outputs.last_hidden_state
    print('last_hidden_states.shape =', last_hidden_states.shape)


if __name__ == '__main__':
    main()
