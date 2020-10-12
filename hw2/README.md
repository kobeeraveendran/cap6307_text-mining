# Hidden Markov Models and Binary Text Classification on 20 Newsgroups

## Setting up the dataset

### Manual (Windows)

1. Download the [20Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz).
2. Move the `.tar.gz` file to this project's root directory and extract its contents.
3. Rename `20news-bydate-train/` to `train/` and `20news-bydate-test/` to `test/` (you can now delete `20news-bydate.tar.gz`).
4. In the train and test folders, remove all folders except `rec.autos` and `rec.sport.hockey`.

### Automatic (Linux/Mac)

1. **Ensure you are in this project's root directory (the same as this README)** and run the setup script (requires `curl`) with:

```bash
bash setup.sh
```

## Usage

In order to run the program, you can use my provided conda environment or install the necessary libraries yourself.

Activate my conda environment by running the following in your terminal at the project's root directory:

```bash
conda env create -f nlp_environment.yml
conda activate nlp
```

OR:

Install the following packages (optionally with exact versions specified) using your package manager of choice:

* spacy == 2.3.1
* python == 3.7
* numpy == 1.19.1
* scikit-learn == 0.23.2

Regardless of which method you chose, you will also need to install spaCy's English language NLP model, which can be done by running:

```bash
python -m spacy download en_core_web_sm
```

Once complete, from the `src/` folder run `python text_classifier.py`.
