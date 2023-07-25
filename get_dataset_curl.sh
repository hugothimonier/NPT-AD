#!/bin/bash

#abalone
curl -o data/abalone.zip https://archive.ics.uci.edu/static/public/1/abalone.zip
unzip data/abalone.zip -d data/abalone && rm data/abalone.zip

#annthyroid
mkdir -p data/annthyroid
curl -o data/annthyroid/annthyroid.mat https://www.dropbox.com/s/aifk51owxbogwav/annthyroid.mat?dl=1

#arrhythmia
curl -o data/arrhythmia.zip https://archive.ics.uci.edu/static/public/5/arrhythmia.zip
unzip data/arrhythmia.zip -d data/arrhythmia && rm data/arrhythmia.zip

#breastw
mkdir -p data/breastw
curl -o data/breastw/breastw.mat https://www.dropbox.com/s/g3hlnucj71kfvq4/breastw.mat?dl=1

#cardio
mkdir -p data/cardio
curl -o data/cardio/cardio.mat https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1

#ecoli
curl -o data/ecoli.zip https://archive.ics.uci.edu/static/public/39/ecoli.zip
unzip data/ecoli.zip -d data/ecoli && rm data/ecoli.zip

#forestcover
mkdir -p data/forest_cover
curl -o data/forest_cover/forest_cover.mat https://www.dropbox.com/s/awx8iuzbu8dkxf1/cover.mat?dl=1

#glass
mkdir -p data/glass
curl -o data/glass/glass.mat https://www.dropbox.com/s/iq3hjxw77gpbl7u/glass.mat?dl=1

#ionosphere
mkdir -p data/ionosphere
curl -o data/ionosphere/ionosphere.mat https://www.dropbox.com/s/lpn4z73fico4uup/ionosphere.mat?dl=1

#kdd
mkdir -p data/kdd
mkdir -p data/kddrev
curl -o data/kdd/kddcup.names http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
curl -o data/kdd/kddcup.data_10_percent.gz http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
curl -o data/kddrev/kddcup.names http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
curl -o data/kddrev/kddcup.data_10_percent.gz http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz

#letter
mkdir -p data/letter
curl -o data/letter/letter.mat https://www.dropbox.com/s/rt9i95h9jywrtiy/letter.mat?dl=1

#lympho
mkdir -p data/lympho
curl -o data/lympho/lympho.mat https://www.dropbox.com/s/ag469ssk0lmctco/lympho.mat?dl=1

#mammography
mkdir -p data/mammography
curl -o data/mammography/mammography.mat https://www.dropbox.com/s/tq2v4hhwyv17hlk/mammography.mat?dl=1

#mnist
mkdir -p data/mnist
curl -o data/mnist/mnist.mat https://www.dropbox.com/s/n3wurjt8v9qi6nc/mnist.mat?dl=1

#mulcross
mkdir -p data/mulcross
curl -o data/mulcross/mulcross.arff https://www.openml.org/data/download/16787460/phpGGVhl9

#musk
mkdir -p data/musk
curl -o data/musk/musk.mat https://www.dropbox.com/s/we6aqhb0m38i60t/musk.mat?dl=1

#optdigits
mkdir -p data/optdigits
curl -o data/optdigits/optdigits.mat https://www.dropbox.com/s/w52ndgz5k75s514/optdigits.mat?dl=1

#pendigits
mkdir -p data/pendigits
curl -o data/pendigits/pendigits.mat https://www.dropbox.com/s/1x8rzb4a0lia6t1/pendigits.mat?dl=1

#pima
mkdir -p data/pima
curl -o data/pima/pima.mat https://www.dropbox.com/s/mvlwu7p0nyk2a2r/pima.mat?dl=1

#satellite
mkdir -p data/satellite
curl -o data/satellite/satellite.mat https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1

#satimage
mkdir -p data/satimage
curl -o data/satimage/satimage.mat https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=1

#seismic
mkdir -p data/seismic
curl -o data/seismic/seismic.arff https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff

#shuttle
mkdir -p data/shuttle
curl -o data/shuttle/shuttle.mat https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1

#speech
mkdir -p data/speech
curl -o data/speech/speech.mat https://www.dropbox.com/s/w6xv51ctea6uauc/speech.mat?dl=1

#thyroid
mkdir -p data/thyroid
curl -o data/thyroid/thyroid.mat https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1

#vertebral
mkdir -p data/vertebral
curl -o data/vertebral/vertebral.mat https://www.dropbox.com/s/5kuqb387sgvwmrb/vertebral.mat?dl=1

#vowels
mkdir -p data/vowels
curl -o data/vowels/vowels.mat https://www.dropbox.com/s/pa26odoq6atq9vx/vowels.mat?dl=1

#wbc
mkdir -p data/wbc
curl -o data/wbc/wbc.mat https://www.dropbox.com/s/ebz9v9kdnvykzcb/wbc.mat?dl=1

#wine
mkdir -p data/wine
curl -o data/wine/wine.mat https://www.dropbox.com/s/uvjaudt2uto7zal/wine.mat?dl=1