# UriPred
A method for predicting urinary proteins

# Introduction
UriPred is developed for predicting, mapping and scanning urinary proteins or peptides. This page provide information about standalone version of UriPred.

## Conda Installation
Conda forge version is also available for easy installation and usage of this tool. The following command is required to install the package 
```
conda install conda-forge::uripred
```
To know about the available option for the conda package, type the following command:
```
uripred -h
```

# Standalone

Standalone version of UriPred is written in python3 and the following libraries are necessary for a successful run:

- pycaret
- scikit-learn
- pandas
- numpy
- blastp

**USAGE** 

To know about the available option for the standalone, type the following command:
```
uripred.py -h
```
To run the example, type the following command:
```
uripred.py -i protein.fa

```
where protein.fa is a input FASTA file. This will predict urinary proteins in FASTA format. It will use other parameters by default. It will save output in "outfile.csv" in CSV (comma separated variables).

**Full Usage**: 
```
Following is complete list of all options, you may get these options
usage: uripred.py [-h] 
                     [-i INPUT]
                     [-o OUTPUT]
                     [-t THRESHOLD]
                     [-m {1,2}] 
                     [-d {1,2}]
```
```
Please provide following arguments

optional arguments:

  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence in FASTA format or
                        single sequence per line in single letter code
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold: Value between 0 to 1 by default 0.6
  -m {1,2}, -- model Model
                        Model: 1: AAC based SVM, 2: Hybrid, by default 1
  -d {1,2}, --display {1,2}
                        Display: 1:Urinary proteins, 2: All proteins, by
                        default 2

```

**Input File**: It allow users to provide input in two format; i) FASTA format (standard) (e.g. peptide.fa) and ii) Simple Format. In case of simple format, file should have one peptide sequence in a single line in single letter code (eg. peptide.seq). 

**Output File**: Program will save result in CSV format, in case user do not provide output file name, it will be stored in outfile.csv.

**Threshold**: User should provide threshold between 0 and 1, please note score is proportional to urinary potential of proteins/peptide.

**Models**: In this program, two models have been incorporated;  
  i) Model1 for predicting given input peptide/protein sequence as urinary and non-urinary peptide/proteins using SVM-RBF based on amino-acid composition of the peptide/proteins; 

  ii) Model2 for predicting given input peptide/protein sequence as urinary and non-urinary peptide/proteins using Hybrid approach, which is the ensemble of Support Vector Machine + BLAST + MERCI. It combines the scores generated from machine learning (SVM), MERCI, and BLAST as Hybrid Score, and the prediction is based on Hybrid Score.


UriPred Package Files
=======================
It contain following files, brief description of these files given below

LICENSE       	: License information

envfile : This file provide the path information for BLAST and MERCI commands ,and data 
          required to run BLAST and MERCI

Database: This folder contains the blast database

progs : This folder contains the program to run MERCI

README.md     	: This file provide information about this package

uripred.py 	: Main python program 

SVM_model        : Model file required for running Machine-learning model


protein.fa	: Example file contain protein sequences in FASTA format 

# Reference
Dr. Amouda's Lab
