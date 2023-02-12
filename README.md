# COMP-8730-NLP
This is a repository for the assignments of COMP-8730 course (Natural Language Processiong & Understanding).

## Setup
This work has been developed on `Python 3.8` and can be installed by `pip`:

```bash
git clone https://github.com/farinamhz/COMP-8730-NLP.git
pip install -r requirements.txt
```
### Assignment 1
Additionally, you need to install the following library from its source:
  
- ``wordnet`` as a requirement in ``nltk`` library with the following command:
  
  ```bash
  python -m nltk.downloader wordnet
  ```
### Assignment 2
Additionally, you need to install the following library from its source:
  
- ``brown`` corpus as a requirement in ``nltk`` library with the following command:
  
  ```bash
  python -m nltk.downloader brown
  ```

## Quickstart

You can run the code for all the assignments using this script: [``Colab Notebook``](https://colab.research.google.com/drive/1YWSMIdIpJkouFadmZEKr8K2pYoUDChvg?usp=sharing)

### Assignment 1


You can run the code via [`./Assignment1-SpellCorrection/main.py`](./Assignment1-SpellCorrection/main.py) with following command:

```bash
cd Assignment1-SpellCorrection
python main.py --data birkbeck-corpus/ms.dat --output output
```
where the input arguments are:

- `data`: Dataset file path
- `output`: Output path

### Assignment 2


You can run the code via [`./Assignment2-SpellCorrectionUsingLM/main.py`](./Assignment2-SpellCorrectionUsingLM/main.py) with following command:

```bash
cd Assignment2-SpellCorrectionUsingLM
python main.py --data birkbeck-corpus/APPLING1DAT.643 --output output
```
where the input arguments are:

- `data`: Dataset file path
- `output`: Output path
