# COMP-8730-NLP
A repository for the assignments of COMP-8730 course

## Setup
This work has been developed on `Python 3.8` and can be installed by `pip`:

```bash
git clone https://github.com/farinamhz/COMP-8730-NLP.git
pip install -r requirements.txt
```

Additionally, you need to install the following library from its source:
  
- ``wordnet`` as a requirement in ``nltk`` library with the following command:
  
  ```bash
  python -m nltk.downloader wordnet
  ```
## Quickstart
We have a misspelled datasets at [`./birckbeck/ms.dat`](./birckbeck/ms.dat).

### Run
You can run the code via [`./Assignment1--SpellCorrection/main.py`](./src/Assignment1--SpellCorrection/main.py) with following command:

```bash
cd Assignment1--SpellCorrection
python main.py
```
where the input arguements are:

- `data`: dataset file path
- `output`: Output path
