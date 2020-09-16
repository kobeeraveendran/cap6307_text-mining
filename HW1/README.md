# Text Classification and Naive Bayes

## Setup
To run this code, ensure that you have an environment set up with the necessary libraries. If you have `conda` installed, create an environment using my provided `environment.yml` file by running the following in your terminal from the `assets/` directory:

```
conda env create -f environment.yml
```

If you run into issues, consider providing an absolute path for the `prefix` variable in `environment.yml` with your local conda installation's prefix. Otherwise, you may also set up this environment manually by running:

```
conda env create -n nlp python=3.7
```

Now, activate the environment with `conda activate nlp`. Then, install the libraries under `dependencies:` in `environment.yml` using the syntax `conda install <package_name>` (though for this project specifically, you should only require `numpy==1.19.1`).

Once done testing the code, if you wish you can remove the environment with:

```
conda remove --name nlp --all
```

## Usage
Once in the root directory (with the conda environment activated), run `python run_classifier.py`. By default, I use a smoothing parameter of 0 (no smoothing) and a train/test split of 80%/20%. You can also configure these two parameters yourself with command-line arguments, as shown below. Note that, when specifying the train/test split, you should supply the percentage of the dataset you want to make up the training set; the test set is allocated the remainder of the data.

```bash
# run with a smoothing param of 1 and an 80/20 train/test split
python run_classifier.py --smoothing 1

# run with no smoothing and a 50/50 train/test split
python run_classifier.py --train_split 0.5

# run with a smoothing param of 3 and a 70/30 train/test split
python run_classifier.py --smoothing 3 train_split 0.7
```