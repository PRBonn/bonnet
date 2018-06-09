# general_preprocess
This script is to generate a bonnet dataset from images and labels
Arguments:
- --img: path for the original images folder.
- --lbl: path for the label images folder.
- --dis: (optinal) for the path of save the merged data.
- --v:   (float/optinal) the present of the validation set over all data
- --t:   (float/optinal) the present of the test set over all data

The dataset structure will like this:
```
Bonnet_dataset
  |-- test
        |-- img
        `-- lbl
  |-- traing
        |-- img
        `-- lbl
  `-- valid
        |-- img
        `-- lbl
```
