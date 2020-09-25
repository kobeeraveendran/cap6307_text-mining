## Hidden Markov Models and Binary Text Classification on 20 Newsgroups

### Setting up the dataset

1. Download the 20Newsgroups [dataset](http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz).
2. Move the `.tar.gz` file to this project's root directory and extract its contents.
3. Rename `20news-bydate-train/` to `train/` and `20news-bydate-test/` to `test/` (you can now delete `20news-bydate.tar.gz`).
4. In the train and test folders, remove all folders except `rec.autos` and `rec.sport.hockey`.


**OR:**

1. **Ensure you are in this project's root directory (the same as this README)** and run the following in your terminal (requires `curl`):

```bash
curl -O http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar -xf 20news-bydate.tar.gz
rm 20news-bydate.tar.gz
mv 20news-bydate-train/ train/ && mv 20news-bydate-test/ test/
shopt -s extglob
cd train/ && rm -r !(rec.sport.hockey|rec.autos)
cd ../test/ && rm -r !(rec.sport.hockey|rec.autos) && cd ..
```