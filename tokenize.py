from transformers import BertTokenizer
from transformers import DataCollatorForLanguageModeling
import pandas as pd
import warnings
import torch
from tqdm import tqdm
import numpy as np
import gc


def read_csv(file_paths, url_data):
    for file in file_paths:
        df = pd.read_csv(file)
        column = df['domain']
        url_data.append(column)
    return url_data


if __name__ == "__main__":
    bert_tokenizer = BertTokenizer(vocab_file="./bert_tokenizer/vocab.txt")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bert_tokenizer, mlm=True, mlm_probability=0.15
    )
    file_paths = ["dataset_path"]
    url_data = []
    train_data = read_csv(file_paths, url_data)
    train_data = pd.concat(train_data, ignore_index=True)
    train_data = train_data.to_numpy()
    filename = "val_data_path"
    df_val = pd.read_csv(filename)
    val_data = df_val["domain"].values

    warnings.filterwarnings('ignore')

    # file data max_num
    max_num = 200000
    max_num_val = 40000
    for i, data in enumerate(tqdm(np.array_split(train_data, len(train_data)/max_num), desc="Split Train_Data and Tokenize")):
        input_ids_train = []
        token_type_ids_train = []
        attention_masks_train = []
        labels_train = []

        for sent in tqdm(data, desc="Tokenizing Train_Data"):
            encoded_dict = bert_tokenizer.encode_plus(
                sent,                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = 64,           # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
            )
            # Add the encoded sentence to the list.
            input_ids_train.append(data_collator.torch_mask_tokens(encoded_dict["input_ids"])[0])
            token_type_ids_train.append(encoded_dict["token_type_ids"])
            attention_masks_train.append(encoded_dict["attention_mask"])
            labels_train.append(data_collator.torch_mask_tokens(encoded_dict["input_ids"])[1])

        torch.save(input_ids_train, "./tokenized_data/train/train_input_ids{}.pt".format(i))
        torch.save(token_type_ids_train, "./tokenized_data/train/train_token_type_ids{}.pt".format(i))
        torch.save(attention_masks_train, "./tokenized_data/train/train_attention_mask{}.pt".format(i))
        torch.save(labels_train, "./tokenized_data/train/train_labels{}.pt".format(i))

        del data
        gc.collect()


    for i, data in enumerate(tqdm(np.array_split(val_data, len(val_data)/max_num_val), desc="Split Val_Data and Tokenize")):
        input_ids_val = []
        token_type_ids_val = []
        attention_masks_val = []
        labels_val = []

        for sent in tqdm(data, desc="Tokenizing Val_Data"):
            encoded_dict = bert_tokenizer.encode_plus(
                sent,                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = 64,           # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
            )
            # Add the encoded sentence to the list.
            input_ids_val.append(data_collator.torch_mask_tokens(encoded_dict["input_ids"])[0])
            token_type_ids_val.append(encoded_dict["token_type_ids"])
            attention_masks_val.append(encoded_dict["attention_mask"])
            labels_val.append(data_collator.torch_mask_tokens(encoded_dict["input_ids"])[1])

        torch.save(input_ids_val, "./tokenized_data/val/val_input_ids{}.pt".format(i))
        torch.save(token_type_ids_val, "./tokenized_data/val/val_token_type_ids{}.pt".format(i))
        torch.save(attention_masks_val, "./tokenized_data/val/val_attention_mask{}.pt".format(i))
        torch.save(labels_val, "./tokenized_data/val/val_labels{}.pt".format(i))

        del data
        gc.collect()
