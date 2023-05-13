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
from sklearn.utils import shuffle
import random
from sklearn.preprocessing import OneHotEncoder

def forward_from_specific_layer(model, layer_number: int, layer_representation: torch.Tensor):
    """
   :param model: a BertForMaskedLM model
   :param layer_representation: a torch tensor, dims: [1, seq length, 768]
   Return: 
           states, a numpy array. dims: [#LAYERS - layer_number, seq length, 768]
           last_state_after_batch_norm: np array, after batch norm. dims: [seq_length, 768]
   """

    layers = model.bert.encoder.layer[layer_number:]
    h = layer_representation
    states = []

    for layer in layers:
        h = layer(h)[0]
        states.append(h)

    last_state_after_batch_norm = model.cls.predictions.transform(states[-1]).detach().cpu().numpy()[0]

    for i, s in enumerate(states):
        states[i] = s.detach().cpu().numpy()[0]

    return np.array(states), last_state_after_batch_norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="upos")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--method", type=str, default="concept_erasure")
    parser.add_argument("--layer", type=int, default=7)
    args.add_argument("--device", type=str, default = "cuda:0")
    args = parser.parse_args()

    with open("data_{}_{}_layer-{}.pickle".format(args.model, args.labels, args.layer), "rb") as f:
        train_data, test_data = pickle.load(f)


    model = AutoModelForMaskedLM.from_pretrained(args.model).to(args.device)
    
    random.seed(0)

    #train_data = shuffle(train_data, random_state=0)
    #test_data = shuffle(test_data, random_state=0)

    train_data = train_data[:100000]
    test_data = test_data[:100000]

    train_x = np.array([d["h"] for d in train_data])
    train_z = np.array([d["label"] for d in train_data])
    train_y = np.array([d["token"] for d in train_data])

    test_x = np.array([d["h"] for d in test_data])
    test_z = np.array([d["label"] for d in test_data])
    test_y = np.array([d["token"] for d in test_data])
 
    z_name_to_id = {z: i for i, z in enumerate(sorted(set(train_z).union(set(test_z))))}
    id_z_to_name = {i: z for z, i in z_name_to_id.items()}
    train_z = np.array([z_name_to_id[z] for z in train_z])
    test_z = np.array([z_name_to_id[z] for z in test_z])
    # convert to one-hot
    
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_z.reshape(-1, 1))
    train_z = enc.transform(train_z.reshape(-1, 1)).toarray()
    print("train size:", train_z.shape, "test size:", test_z.shape)

    if args.method == "concept_erasure":
 
        X_t = torch.from_numpy(train_x)
        Z_t = torch.from_numpy(train_z)
        X_test_t = torch.from_numpy(test_x)

        eraser = ConceptEraser.fit(X_t, Z_t)
        test_x_projected = eraser(X_test_t)
        
        last_state_after_batch_norm_projected = []
        last_state_after_batch_norm_original = []
        
        for h_proj, h in zip(test_x_projected, X_t):
        _, h_final_projected = forward_from_specific_layer(model, args.layer, h_proj.to(args.device))
        _, h_final_original = forward_from_specific_layer(model, args.layer, X_t.to(args.device))
        print("Saving...")
        
        # saving the last hidden representations (just before the LM head) for the projecdted and original representations
        with open("interim/data_projected_{}_{}_{}.pickle".format(args.method, args.model, args.labels), "wb") as f:
            pickle.dump({"x_projected": last_state_after_batch_norm_projected, "x": last_state_after_batch_norm_original, "y": test_y, "z": test_z,
                         "id_to_z": id_z_to_name}, f)
    
    elif args.method == "inlp":
        raise NotImplementedError