# COMP-8730-NLP
A repository for the assignments of COMP-8730 course

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

### Assignment 1

**Run**

You can run the code via [`./Assignment1-SpellCorrection/main.py`](./src/Assignment1-SpellCorrection/main.py) with following command:

```bash
cd Assignment1-SpellCorrection
python main.py --data birckbeck/ms.dat --output output
```
where the input arguements are:

- `data`: Dataset file path
- `output`: Output path

### Assignment 2

**Run**

You can run the code via [`./Assignment2-SpellCorrection/main.py`](./src/Assignment2-SpellCorrection/main.py) with following command:

```bash
cd Assignment2-SpellCorrection
python main.py --data birkbeck-corpus/APPLING1DAT.643 --output output
```
where the input arguements are:

- `data`: Dataset file path
- `output`: Output path
