import os
import json
import torch
import openai
import logging
import argparse
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from tensor_parallel import TensorParallelPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM

openai.api_key = "XXX"  # your API key
logging.getLogger().setLevel(logging.INFO)


def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tar_mod", type=str, default="pythia-2.8B")
    parser.add_argument("--ref_mod", type=str, default="pythia-70M")
    parser.add_argument("--data", type=str, default="WikiMIA_128")
    parser.add_argument("--max_tok", type=int, default=1024)
    parser.add_argument("--key_nam", type=str, default="input")
    parser.add_argument("--gpu_num", type=int, default=4)
    arg = parser.parse_args()
    return arg


def load_model(tar_mod_nam, ref_mod_nam):
    devices = []
    for i in range(args.gpu_num):
        devices.append(f"cuda:{i}")

    tar_mod = AutoModelForCausalLM.from_pretrained(tar_mod_nam, return_dict=True, trust_remote_code=True)
    tar_mod = TensorParallelPreTrainedModel(tar_mod, devices)
    tar_mod.eval()
    tar_tok = AutoTokenizer.from_pretrained(tar_mod_nam, trust_remote_code=True)

    ref_mod = AutoModelForCausalLM.from_pretrained(ref_mod_nam, return_dict=True, trust_remote_code=True)
    ref_mod = TensorParallelPreTrainedModel(ref_mod, devices)
    ref_mod.eval()
    ref_tok = AutoTokenizer.from_pretrained(ref_mod_nam, trust_remote_code=True)
    return tar_mod, ref_mod, tar_tok, ref_tok


def cal_ppl(text, model, tok):
    device = model.device
    input_ids = tok.encode(text, max_length=args.max_tok, truncation=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = model(input_ids, labels=input_ids)

    logit = output[1]

    # Apply softmax to the logits to get probabilities
    prob_weight = torch.nn.functional.softmax(logit, dim=-1)[0][:-1]
    prob = torch.nn.functional.log_softmax(logit, dim=-1)[0][:-1]
    prob_mu = (prob_weight * prob).sum(-1)
    prob_sigma = (prob_weight * torch.square(prob)).sum(-1) - torch.square(prob_mu)
    input_ids = input_ids[0][1:]

    probs = prob[torch.arange(len(prob)).to(device), input_ids].tolist()
    input_ids = input_ids.tolist()
    mu = prob_mu.tolist()
    sigma = prob_sigma.tolist()
    return probs, input_ids, mu, sigma


def inference(text, label, tar_mod, ref_mod, tar_tok, ref_tok):
    response = {}
    tar_prob, input_ids, mu, sigma = cal_ppl(text, tar_mod, tar_tok)
    low_prob, _, _, _ = cal_ppl(text.lower(), tar_mod, tar_tok)
    ref_prob, _, _, _ = cal_ppl(text, ref_mod, ref_tok)

    response["input_ids"] = input_ids
    response["tar_prob"] = tar_prob
    response["low_prob"] = low_prob
    response["ref_prob"] = ref_prob
    response["tar_prob_mu"] = mu
    response["tar_prob_sigma"] = sigma
    response["text"] = text
    response["label"] = label
    return response

def tok_pro_dis(dat, key_nam, tar_mod, ref_mod, tar_tok, ref_tok):
    responses = []
    for example in tqdm(dat):
        text = example[key_nam]
        label = example["label"]
        responses.append(inference(text, label, tar_mod, ref_mod, tar_tok, ref_tok))

    return  responses

if __name__ == '__main__':
    args = get_arg()
    logging.info(f"compute token probability distribution from {args.tar_mod} on {args.data}")

    out_dir = "output/pro_dis"
    out_path = os.path.join(out_dir, args.data)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    dat_dir = "data/"
    dat_pat = os.path.join(dat_dir, f"{args.data}.jsonl")
    with open(dat_pat, 'r') as f:
        dataset = [json.loads(line) for line in f]

    mod_dir = "hf_model/"
    tar_mod_name = os.path.join(mod_dir, args.tar_mod)
    ref_mod_name = os.path.join(mod_dir, args.ref_mod)
    tar_model, ref_model, tar_tokenizer, ref_tokenizer = load_model(tar_mod_name, ref_mod_name)
    pro_dis = tok_pro_dis(dataset, args.key_nam, tar_model, ref_model, tar_tokenizer, ref_tokenizer)

    with open(f"{out_path}/{args.tar_mod}.pkl", "wb") as f:
        pkl.dump(pro_dis, f)