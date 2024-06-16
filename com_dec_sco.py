import os
import zlib
import jieba
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from scipy import stats
from collections import defaultdict
from transformers import AutoTokenizer
from sklearn.metrics import auc, roc_curve

def get_arg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tar_mod", type=str, default="pythia-2.8B")
    parser.add_argument("--data", type=str, default="WikiMIA_128")
    parser.add_argument("--max_cha", type=int, default=512, desc="text length")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--a", type=str, default="0.01", desc="the hyperparameter of DC-PDD")
    arg = parser.parse_args()
    return arg


def sweep(x, score):
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc


def evaluate(es, fpr_threshold=0.05):
    answers = []
    metric2predictions = defaultdict(list)
    for e in es:
        answers.append(e["label"])
        for metric in e["pred"].keys():
            metric2predictions[metric].append(e["pred"][metric])

    for metric, predictions in metric2predictions.items():
        fpr, tpr,  auc, acc = sweep(np.array(answers, dtype=bool), np.array(predictions))
        low = tpr[np.where(fpr < fpr_threshold)[0][-1]]
        print("Attack %s AUC %.4f, Accuracy %.4f, TPR@5FPR of %.4f\n" %(metric, auc, acc, low))


def cal_met(pro_dis, lang, max_cha, a):
    es = []
    for i, t in enumerate(tqdm(pro_dis)):
        e = {}
        pred = {}
        if lang == "cn":
            text = "".join(jieba.lcut(t["text"])[:max_cha])
        else:
            text = " ".join(t["text"].split()[:max_cha])

        tar_tok_num = len(tokenizer.encode(text))
        tar_ppl = np.exp(-np.mean(t["tar_prob"][:tar_tok_num]))
        pred["ppl"] = tar_ppl  # larger for nonmember

        low_tok_num = len(tokenizer.encode(text.lower()))
        low_ppl = np.exp(-np.mean(t["low_prob"][:low_tok_num]))
        pred["tar_ppl/low_ppl"] = np.log(tar_ppl)/np.log(low_ppl)  # larger for nonmember

        ref_ppl = np.exp(-np.mean(t["ref_prob"][:tar_tok_num]))
        pred["tar_ppl/ref_ppl"] = np.log(tar_ppl)/np.log(ref_ppl)  # larger for nonmember

        z_lib = len(zlib.compress(bytes(text, 'utf-8')))
        pred["tar_ppl/zlib"] = np.log(tar_ppl) / z_lib  # larger for nonmember

        k = int(tar_tok_num * 0.2)
        min_k_pro = np.sort(t["tar_prob"][:tar_tok_num])[:k]
        pred[f"min_20% prob"] = -np.mean(min_k_pro).item()  # larger for nonmember

        mu = np.array(t["tar_prob_mu"])
        sigma = np.array(t["tar_prob_sigma"])
        token_log_probs = np.array(t["tar_prob"])
        mink_plus = (token_log_probs - mu) / np.sqrt(sigma)
        k = int(len(mink_plus) * 0.2)
        min_k_plus_pro = np.sort(mink_plus)[:k]
        pred['min_20%++ prob'] = -np.mean(min_k_plus_pro).item()  # larger for nonmember

        probs = np.exp(t["tar_prob"])
        input_ids = t["input_ids"]

        # tokens with first occurance in text
        indexes = []
        current_ids = []
        for i, input_id in enumerate(input_ids):
            if input_id not in current_ids:
                indexes.append(i)
                current_ids.append(input_id)

        x_pro = probs[indexes]
        x_fre = fre_dis[input_ids[indexes]]  # 加 1 法
        ce = x_pro * np.log(1 / x_fre)
        ce[ce > a] = a
        pred["DC-PDD"] = -np.mean(ce)

        e["pred"] = pred
        e["label"] = t["label"]
        es.append(e)

    return es


if __name__ == "__main__":
    args = get_arg()
    tok_dir = "hf_model/"
    tok_path = os.path.join(tok_dir, args.tar_mod)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    pro_dis_dir = "output/pro_dis"
    fre_dis_dir = "output/fre_dis"
    pro_dis_pat = os.path.join(pro_dis_dir, args.data, f"{args.tar_mod}.pkl")
    with open(pro_dis_pat, "rb") as f:
        pro_dis = pkl.load(f)

    fre_dis_pat = os.path.join(fre_dis_dir, f"{args.tar_mod}.pkl")
    with open(fre_dis_pat, "rb") as f:
        fre_dis = pkl.load(f)

    fre_dis_npy = np.array(fre_dis)
    fre_dis_smo = (fre_dis_npy + 1) / (sum(fre_dis_npy) + len(fre_dis_npy))

    examples = cal_met(pro_dis, args.lang, args.max_cha, args.a)
    evaluate(examples)