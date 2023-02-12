import argparse, os, pickle
from tqdm import tqdm
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


def visualization(result_path, metric_list, topk):
    x = pd.read_csv(f'{result_path}/{metric_list[0]}-gram/pred.eval.mean.csv')
    x.columns = ['Metric', 'mean']
    merged = x['Metric']
    for m in metric_list:
        lm = pd.read_csv(f'{result_path}/{m}-gram/pred.eval.mean.csv')
        merged = pd.concat([merged, lm['mean']], axis=1)
    column_names = ['Metric']
    for l in [f'{m}-gram' for m in metric_list]:
        column_names.append(l)
    merged.columns = column_names
    print(merged)
    melted_query = merged.melt('Metric', var_name='language_models', value_name='Values')
    p = sns.lineplot(x='Metric', y='Values', hue='language_models', palette='Set2', linewidth=5, data=melted_query)
    p.lines[1].set_linestyle("solid")
    p.lines[2].set_linestyle("dashed")
    p.lines[3].set_linestyle("dashdot")
    p.lines[4].set_linestyle("dotted")
    plt.legend(loc='upper right')
    plt.title('Success @k')
    plt.savefig(f'{result_path}/Success_outputresult.png')
    plt.clf()


def main(args):
    if not os.path.isdir(f'{args.output}'): os.makedirs(f'{args.output}')
    ms_dataset = load(args.data, args.output)
    ms, gt, sents, brown_dataset, dic = preprocess(ms_dataset)

    print(f'Dataset: {len(sents)} sentences, {len(set(gt))} unique correct and unique {len(set(ms))} misspelled words')
    print(f"Brown dictionary has {len(dic)} unique words")

    ngram_models = dict()
    k = 10
    metric_list = [1, 2, 3, 5, 10]
    for n in metric_list:  # [1, 2, 3, 5, 10]
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
        result_path = f'{model_path}/pred.eval.mean.csv'
        df_mean.to_csv(result_path)
        # print(f'{n}-gram model:')
        # for i in range(len(ms)):
        #     print(f'Most similar words to {ms[i]} with correct format {gt[i]}: {top_list[i]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Spell Correction')
    parser.add_argument('--data', dest='data', type=str, default='birkbeck-corpus/APPLING1DAT.643',
                        help='Dataset file path, e.g., birkbeck-corpus/APPLING1DAT.643')
    parser.add_argument('--output', dest='output', type=str, default='output', help='output path, e.g., ../output/')
    args = parser.parse_args()

    main(args)

    metric_list = [1, 2, 3, 5, 10]
    visualization(args.output, metric_list, [1, 5, 10])
