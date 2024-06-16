import os
import gzip
import json
import torch
import logging
import argparse
import jsonlines
import pickle as pkl
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ref_dat", type=str, default="C4")
    parser.add_argument("--fil_num", type=int, default=15)
    parser.add_argument("--model", type=str, default="pythia-2.8B")
    parser.add_argument("--max_tok", type=int, default=1024)
    parser.add_argument("--vob_siz", type=int, default=50304, help="50304 for pythia")
    parser.add_argument("--sav_int", type=int, default=1e4)
    arg = parser.parse_args()
    return arg


def fre_dis(ref_dat, tok, fre_dis, max_tok, k):
    """
    token frequency distribution
    ref_dat: reference dataset
    tok: tokenizer
    """
    for i, e in enumerate(tqdm(ref_dat, desc=f"{k+1} sub-dataset")):
        text = e["text"]
        input_ids = tok.encode(text)[:max_tok]
        for input_ids in input_ids:
            ran_dis[input_ids] += 1

if __name__ == "__main__":
    args = get_arg()
    logging.info(f"compute token frequency distribution for {args.model} using {args.fil_num} files of {args.ext_dat}")

    out_dir = "output/fre_dis"
    out_path = os.path.join(out_dir, args.ran_dis)
    Path(out_path).mkdir(parents=True, exist_ok=True)

    mod_dir = "hf_model/"
    mod_pat = os.path.join(mod_dir, args.model)
    tokenizer = AutoTokenizer.from_pretrained(mod_pat, trust_remote_code=True)
    fre_dis = [0] * args.vob_siz

    if args.ext_dat == "C4":
        for i in range(args.fil_num):
            iter = i
            while len(str(i)) < 5:
                i = "0" + str(i)
            fil_nam = f"c4-train.{i}-of-01024.json.gz"
            ref_dat_pat = os.path.join(args.ref_dat, fil_nam)
            with open(ref_dat_pat, "r+", encoding="utf8") as f:
                sub_dataset = gzip.open(ref_dat_pat, "rb")
                examples = []
                for example in tqdm(sub_dataset):
                    example = json.loads(example)
                    examples.append(example)
                fre_dis(examples, tokenizer, fre_dis, args.max_tok, iter)
    else:
        for i in range(args.fil_num):
            iter = i
            while len(str(i)) < 4:
                i = "0" + str(i)
            fil_nam = f"part-{i}.jsonl"
            ref_dat_pat = os.path.join(ref_dat_dir, args.ref_dat, fil_nam)
            with open(ref_dat_pat, "r+", encoding="utf8") as f:
                examples = []
                for example in tqdm(jsonlines.Reader(f)):
                    examples.append(example)
                fre_dis(examples, tokenizer, fre_dis, args.max_tok, iter)

    with open(f"{out_path}/{args.model}.pkl", "wb") as f:
        pkl.dump(fre_dis, f)