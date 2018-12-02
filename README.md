# CF-Project

The main file is Main.ipynb.
First, run Main.ipynb, it will read the yelp-dataset and write the data into 2 .dat files, train.dat and test.dat with aan 80-20 split. It also trains the Matrix-factorization model over this dataset.
Then, run ConvertDataset.ipyb. It will convert the .dat file into TFRECORD files used for traing.
Then, run TrainDAE.ipynb. It will train the data over a deep autoencoder.
