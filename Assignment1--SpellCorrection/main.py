import argparse, os, pickle, multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
import pytrec_eval
from nltk.corpus import wordnet


def load(input, output):
    print('\nLoading and preprocessing ...')
    print('#' * 50)
    t_s = time()
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
    print(f'Time elapsed: {(time() - t_s)}')
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


def evaluation(top_list, ms, gt, metrics_set, output, k):
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
        for i in range(len(gt)):
            misspelled = ms[i]
            qrel[misspelled] = {}
            qrel[misspelled][gt[i]] = 1
        for i in range(len(ms)):
            misspelled = ms[i]
            run[misspelled] = {}
            words = top_list[i]
            for j in range(len(words)):
                run[misspelled][words[j]] = 1
        with open(f'{output}/qrel_with_k10.pkl', 'wb') as f:
            pickle.dump(qrel, f)
        with open(f'{output}/run_with_k10.pkl', 'wb') as f:
            pickle.dump(run, f)

    print(f'Calling pytrec_eval for {metrics_set} ...')
    df = pd.DataFrame.from_dict(pytrec_eval.RelevanceEvaluator(qrel, metrics_set).evaluate(run))
    print(f'Averaging ...')
    df_mean = df.mean(axis=1).to_frame('mean')
    return df_mean


def get_topk(ms, dic, k, output):
    top_list = []
    for ms_word in tqdm(ms):
        distances = []
        for d_idx, d in enumerate(dic):
            distances.append(med_algorithm(ms_word, d, len(ms_word), len(d)))
        distances = np.asarray(distances)
        top_k = dic[np.asarray(distances.argpartition(range(k))[:k])]
        top_list.append(top_k)
    with open(f'{output}/toplist.pkl', 'wb') as file:
        pickle.dump(top_list, file)
    return top_list


def main(args):
    if not os.path.isdir(f'{args.output}'): os.makedirs(f'{args.output}')
    dataset = load(args.data, args.output)
    gt, ms, dic = preprocess(dataset)
    k = 10
    chunks = np.array_split(ms, len(ms) / 100)
    top_list = Parallel(n_jobs=-1, prefer="processes")(delayed(get_topk)(i, dic, k, args.output) for i in chunks)
    # top_list = get_topk(ms, dic, k)
    metrics_set = {'success_1,5,10'}
    df_mean = evaluation(top_list, ms, gt, metrics_set, args.output, k)
    df_mean.to_csv(f'{args.output}/pred.eval.mean.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latent Aspect Detection')
    parser.add_argument('--data', dest='data', type=str, default='birkbeck-corpus/ms.dat',
                        help='Dataset file path, e.g., birkbeck-corpus/ms.dat')
    parser.add_argument('--output', dest='output', type=str, default='output', help='output path, e.g., ../output/')
    args = parser.parse_args()

    main(args)
