# Bonnet Dataset Creator
Scripts for helping to create a dataset for Bonnet framwork.

## Mergefiles.py
is for marge images from diffrent forlders in same structre.
parameters:
- --img: path for the original images folder.
- --lbl: path for the label images folder.
- --dis: (optinal) for the path of save the merged data.
```
python mergefiles.py --img c:\Users\user\Downloads\Iniana_dataset\images --lbl c:\Users\user\Downloads\Iniana_dataset\segmenter --dis c
:\Users\user\Downloads\Iniana_dataset

```
For example:
```
original_images
  |-- folder1
  `-- folder2
  
label_images
  |-- folder1
  `-- folder2
````  
the folder after running the script will be like this:
```
Merged_data_images
|-- OriginalImages
`-- LabelImages
```

## create_bonnet_dataset.py
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
