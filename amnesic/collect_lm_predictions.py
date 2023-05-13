import sys
sys.path.append("../")

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from concept_eraser import ConceptEraser
import numpy as np
from datasets import load_dataset
import argparse
import os
from typing import List
#main function
import tqdm
import pickle



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="bert-base-uncased")
    args.add_argument("--labels", type=str, default = "upos_NOUN-ADJ")
    args.add_argument("--device", type=str, default = "cpu")
    args.add_argument("--method", type=str, default = "concept_erasure")
    args = args.parse_args()

    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    with open("interim/data_projected_{}_{}_{}.pickle".format(args.method, args.model, args.labels), "rb") as f:
        data = pickle.load(f)

    # collect LM predictions
    lm_final_embddings = model.cls.predictions.decoder.to(args.device)

    X = torch.tensor(data["x"]).float()
    X_projected = torch.tensor(data["x_projected"]).float()
    #y = torch.tensor(data["y"]).float()
    #z = torch.tensor(data["z"]).float()

    logits_over_X = lm_final_embddings(X)
    logits_over_X_projected = lm_final_embddings(X_projected)

    word_predictions = logits_over_X.argmax(-1)
    word_predictions_projected = logits_over_X_projected.argmax(-1)
    word_predictions_tokens = np.array(tokenizer.convert_ids_to_tokens(word_predictions))
    word_predictions_projected_tokens = np.array(tokenizer.convert_ids_to_tokens(word_predictions_projected))

    y = np.array(data["y"])

    acc_original = (word_predictions_tokens == y).mean().item()
    acc_projected = (word_predictions_projected_tokens == y).mean().item()
    id2z = data["id_to_z"]
    z = np.array([id2z[i] for i in data["z"]])

    print("Accuracy original: {}".format(acc_original))
    print("Accuracy projected: {}".format(acc_projected))

    # collect acc per POS

    pos_to_acc_original = {}
    pos_to_acc_projected = {}

    for pos in set(z):
        pos_to_acc_original[pos] = (word_predictions_tokens[z == pos] == y[z == pos]).mean().item()
        pos_to_acc_projected[pos] = (word_predictions_projected_tokens[z == pos] == y[z == pos]).mean().item()
    
    # print acc per POS, sorted from high to low

    pos_to_change_in_acc = {}
    for pos in pos_to_acc_original:
        pos_to_change_in_acc[pos] = pos_to_acc_original[pos] - pos_to_acc_projected[pos]
    
    # sort by change in acc

    sorted_pos_to_change_in_acc = {k: v for k, v in sorted(pos_to_change_in_acc.items(), key=lambda item: item[1], reverse=True)}
    for pos in sorted_pos_to_change_in_acc:
        print("{}: {} -> {}".format(pos, pos_to_acc_original[pos], pos_to_acc_projected[pos]))