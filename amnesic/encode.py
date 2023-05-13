import sys
sys.path.append("../")

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from datasets import load_dataset
import argparse
import os
from typing import List
#main function
import tqdm
import pickle


def main():
    pass


def tokenize(tokenizer, tokens):
    # recive a list of tokens and return a list of token ids as well as a mapping from token ids to tokens

    
    token_ids = []
    token_id_to_token = {}
    all_bert_tokens = []

    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    for i,original in enumerate(tokens):
        bert_tokens = tokenizer.tokenize(original)
        for token in bert_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            token_ids.append(token_id)
            token_id_to_token[len(token_ids)-1] = i
            all_bert_tokens.append(token)

        
    #  token to token_id mapping
    token_to_token_id = {}
    # since each token can be split into multiple tokens, we need to map each token to a list of token_ids

    
    return tokens, all_bert_tokens,token_ids, token_id_to_token


def encode(model, tokenizer, sentences: List[List[str]], labels: List[List[str]], device="cpu", layer=-1):
    data = []
    for sentence, label_seq in tqdm.tqdm(zip(sentences, labels), total=len(sentences)):

        tokens, all_bert_tokens,token_ids, token_id_to_token = tokenize(tokenizer, sentence)
        with torch.no_grad():
            hidden_states = model(torch.tensor([token_ids]).to(device), output_hidden_states=True, return_dict=True).hidden_states[layer if layer < 13 else -1]
            if layer == 13 or layer == -1:
                hidden_states = model.cls.predictions.transform(hidden_states).detach().cpu().numpy()[0]

        for i,h in enumerate(hidden_states):
            if i == 0 or i == len(hidden_states)-1:
                continue   

            token_id = token_id_to_token[i] - 1
            label = label_seq[token_id]
            data.append({"h": h, "label": label, "token": all_bert_tokens[i]})
    
    return data


def get_tokens_and_labels(dataset_split, labels):

    tokens = [d["tokens"] for d in dataset_split]
    labels = [d[labels] for d in dataset_split]
    return tokens, labels

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="bert-base-uncased")
    args.add_argument("--labels", type=str, default = "upos")
    args.add_argument("--device", type=str, default = "cuda:0")
    args.add_argument("--specific_tags", type=str, default = None)
    args.add_argument("--layer", type=int, default = 12)
    args = args.parse_args()

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    #load dataset

    dataset = load_dataset("universal_dependencies", "en_gum")
    train = dataset["train"]
    test = dataset["test"]

    upos_tags = ["NOUN","PUNCT","ADP","NUM","SYM","SCONJ","ADJ","PART","DET","CCONJ","PROPN","PRON","X","_","ADV","INTJ","VERB","AUX"]

    train_tokens, train_labels = get_tokens_and_labels(train, args.labels)
    test_tokens, test_labels = get_tokens_and_labels(test, args.labels)

    if args.labels == "upos":

        train_labels = [[upos_tags[label] for label in labels] for labels in train_labels]
        test_labels = [[upos_tags[label] for label in labels] for labels in test_labels]

    train_data = encode(model, tokenizer, train_tokens, train_labels, args.device, args.layer)
    test_data = encode(model, tokenizer, test_tokens, test_labels, args.device, args.layer)

    if args.specific_tags is not None:
        specific_tags = args.specific_tags.split("-")
        train_data = [t for t in train_data if t[1] in specific_tags]

    # create dir interim if it does not exist
    if not os.path.exists("interim"):
        os.makedirs("interim")
    with open("interim/data_{}_{}{}_layer-{}.pickle".format(args.model, args.labels, ("_"+args.specific_tags) if args.specific_tags is not None else "", args.layer), "wb") as f:
        pickle.dump((train_data, test_data), f)