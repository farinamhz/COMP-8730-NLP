import argparse, os, pickle, multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
import pytrec_eval
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import seaborn as sns

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
    corr_word = ''
    for word in dataset:
        if word[0] == '$':
            corr_word = word[1:]
        else:
            corr.append(corr_word.lower())
            ms.append(word.lower())
    dic = []
    for w in tqdm(wordnet.all_synsets()):
        dic.append(w.name().split('.')[0])
    dic = np.unique(np.asarray(dic))
    return np.asarray(corr), np.asarray(ms), dic


def med_algorithm(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1],  # Insert
                                   dp[i - 1][j],  # Remove
                                   dp[i - 1][j - 1])  # Replace

    return dp[m][n]


def evaluation(top_list, metrics_set, output, k):
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
        for i, ms_word in enumerate(top_list):
            run[f'q{i}'] = {}
            qrel[f'q{i}'] = {}
            corr_word, mswords_list = top_list[ms_word]
            qrel[f'q{i}'][corr_word] = 1
            for j in range(len(mswords_list)):
                run[f'q{i}'][mswords_list[j][0]] = len(mswords_list) - j
        with open(f'{output}/qrel_with_k10.pkl', 'wb') as f:
            pickle.dump(qrel, f)
        with open(f'{output}/run_with_k10.pkl', 'wb') as f:
            pickle.dump(run, f)

    print(f'Calling pytrec_eval for {metrics_set} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(run))
    print(f'Averaging ...')
    df_mean = df.mean(axis=1).to_frame('mean')
    return df_mean


def get_topk(ms, gt, dic, k, output):
    # print('\nLoading top list file failed! ...')
    top_list = dict()
    for i in tqdm(range(len(ms))):
        distances = []
        for d_idx, d in enumerate(dic):
            distances.append((d, med_algorithm(ms[i], d, len(ms[i]), len(d))))
        distances.sort(key=lambda x: x[1])
        top_list[ms[i]] = (gt[i], distances[:k])
    with open(f'{output}/toplist.pkl', 'ab') as file:
        pickle.dump(top_list, file)
    return top_list


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


def main(args):
    if not os.path.isdir(f'{args.output}'): os.makedirs(f'{args.output}')

    print('\nLoading top list file ...')
    dataset = load(args.data, args.output)
    gt, ms, dic = preprocess(dataset)
    k = 10
    # chunks = np.array_split(ms, len(ms) / 50)
    # top_list = Parallel(n_jobs=-1, prefer="processes")(delayed(get_topk)(i, gt, dic, k, args.output) for i in chunks)

    top_list_list = list()
    with open(f'{args.output}/toplist.pkl', 'rb') as f:
        while True:
            try:
                top_list_list.append(pickle.load(f))
            except EOFError:
                break
    top_list = dict()
    for t in top_list_list:
        top_list.update(t)

    # top_list = get_topk(ms, gt, dic, k, args.output)
    print(f'Dataset have {len(dataset)} entries and {len(set(gt))} unique correct words and unique {len(ms)} misspelled words')
    print(f"Wordnet dictionary has {len(dic)} unique words")
    # print(f'Most similar words to {ms[0]}: {top_list[ms[0]]}')
    metrics_set = {'success_1,5,10'}
    df_mean = evaluation(top_list, metrics_set, args.output, k)
    result_path = f'{args.output}/pred.eval.mean.csv'
    df_mean.to_csv(result_path)
    visualization(result_path, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Spell Correction')
    parser.add_argument('--data', dest='data', type=str, default='birkbeck-corpus/ms.dat',
                        help='Dataset file path, e.g., birkbeck-corpus/ms.dat')
    parser.add_argument('--output', dest='output', type=str, default='output', help='output path, e.g., ../output/')
    args = parser.parse_args()

    main(args)
