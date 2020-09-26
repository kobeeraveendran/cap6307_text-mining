curl -O http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
tar -xf 20news-bydate.tar.gz
rm 20news-bydate.tar.gz
mv 20news-bydate-train/ train/ && mv 20news-bydate-test/ test/
shopt -s extglob
cd train/ && rm -r !(rec.sport.hockey|rec.autos)
cd ../test/ && rm -r !(rec.sport.hockey|rec.autos) && cd ..