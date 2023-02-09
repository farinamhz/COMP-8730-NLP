import argparse, os, pickle
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
import pytrec_eval
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


def load(input, output):
    print('\nLoading and preprocessing ...')
    corpus = []
    try:
        print('\nLoading file ...')
        with open(f'{output}/corpus.pkl', 'rb') as f:
            corpus = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print('\nLoading corpus file failed! ...')
        with tqdm(total=os.path.getsize(input)) as pbar, open(input, "r", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                pbar.update(len(line))
                corpus.append(line.strip())
        with open(f'{output}/corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Dataset: {len(corpus)} entries')
    return corpus


def preprocess(dataset):
    corr = []
    ms = []
    sents = []
    for d in dataset:
        r = d.split()
        if len(r) > 1:
            ms.append(r[0])
            corr.append(r[1])
            sents.append(r[2:])
    bd = brown.words(categories='news')
    brown_dataset = [word.lower() for word in bd]
    dic = np.asarray(list(set(brown_dataset)))
    return ms, corr, sents, brown_dataset, dic


def train(n, browm_dataset):
    train_data, padded_sents = padded_everygram_pipeline(n, [browm_dataset])
    mdl = MLE(n)  # train a n-grams model
    mdl.fit(train_data, padded_sents)
    return mdl


def evaluation(top_list, gt, metrics_set, output, k):
    try:
        print('\nLoading qrel and run ...')
        with open(f'{output}/qrel_with_k{k}.pkl', 'rb') as f:
            qrel = pickle.load(f)
        with open(f'{output}/run_with_k{k}.pkl', 'rb') as f:
            run = pickle.load(f)
    except (FileNotFoundError, EOFError) as e:
        print('\nLoading qrel and run files failed! ...')
        qrel = {}
        run = {}
        i = 0
        for t in top_list:
            run[f'q{i}'] = {}
            qrel[f'q{i}'] = {}
            corr_word = gt[i]
            qrel[f'q{i}'][corr_word] = 1
            for j in range(len(top_list[t])):
                run[f'q{i}'][top_list[t][j]] = len(top_list[t]) - j
            i += 1
        with open(f'{output}/qrel_with_k10.pkl', 'wb') as f:
            pickle.dump(qrel, f)
        with open(f'{output}/run_with_k10.pkl', 'wb') as f:
            pickle.dump(run, f)
    # print("qrel", qrel)
    # print("run", run)
    print(f'Calling pytrec_eval for {metrics_set} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(run))
    print(f'Averaging ...')
    df_mean = df.mean(axis=1).to_frame('mean')
    return df_mean


def visualization(result_path, output):
    r = pd.read_csv(result_path)
    r.columns = ['k', 'mean']
    r = r.replace('_', ' @', regex=True)
    ax = sns.barplot(y="mean", x="k", data=r, estimator=sum)
    for i in ax.containers:
        ax.bar_label(i, )
    # ax.set(ylim=(0.0, 1.0))
    plt.title("Success @K", fontsize=20, pad=15)
    plt.xlabel('Metric', fontsize=15, labelpad=5)
    plt.ylabel('Value', fontsize=15, labelpad=5)
    plt.savefig(f'{output}result.png')
    plt.clf()


def main(args):
    if not os.path.isdir(f'{args.output}'): os.makedirs(f'{args.output}')
    ms_dataset = load(args.data, args.output)
    ms, gt, sents, brown_dataset, dic = preprocess(ms_dataset)
    ngram_models = dict()
    k = 10
    for n in [1, 2, 3, 5, 10]:  # [1, 2, 3, 5, 10]
        ngram_models[n] = train(n, brown_dataset)
    for n in ngram_models:
        top_list = dict()
        for s_idx, sent in enumerate(sents):
            start_idx = sent.index("*")
            text = sent[max(0, start_idx - n + 1):start_idx]
            scores = []
            for word in dic:
                score = ngram_models[n].score(word, text)
                scores.append(score)
            scores = np.asarray(scores)
            top_list_idx = np.flip(scores.argsort()[(0 - k):])
            top_list[s_idx] = dic[top_list_idx].tolist()
        metrics_set = {'success_1,5,10'}
        model_path = f'{args.output}/{n}-gram'
        if not os.path.isdir(f'{model_path}'): os.makedirs(f'{model_path}')
        df_mean = evaluation(top_list, gt, metrics_set, model_path, k)
        # print(f'Dataset: {len(dataset)} Entries, {len(set(gt))} unique correct and unique {len(ms)} misspelled words')
        # print(f"Brown dictionary has {len(dic)} unique words")
        # print(f'Most similar words to {ms[0]}: {top_list[ms[0]]}')
        result_path = f'{model_path}/pred.eval.mean.csv'
        df_mean.to_csv(result_path)
        visualization(result_path, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Spell Correction')
    parser.add_argument('--data', dest='data', type=str, default='birkbeck-corpus/APPLING1DAT.643',
                        help='Dataset file path, e.g., birkbeck-corpus/APPLING1DAT.643')
    parser.add_argument('--output', dest='output', type=str, default='output', help='output path, e.g., ../output/')
    args = parser.parse_args()

    main(args)
