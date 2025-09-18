# UriPred (Urinary Protein Predictor)
A tool to predict the urinary proteins.

# Introduction
UriPred is developed for predicting, mapping and scanning urinary proteins or peptides. This page provides information about standalone version of the tool.

# Standalone

Python3 is used to develop a standalone version of UriPred with the following necessary libraries for a successful run:

- pycaret
- scikit-learn
- pandas
- numpy
- blastp

**USAGE** 

Type the following commands to explore the options of the tool:
```
uripred.py -h
```
To run the example, type the following command:
```
uripred.py -i protein.fa

```
Where protein.fa is a input FASTA file  will predict the urinary proteins with  default parameters. The output file  "outfile.csv"  (comma separated variables).

**Full Usage**: 
```
Following is complete list of all options, you may get these options
usage: uripred.py [-h] 
                     [-i INPUT]
                     [-o OUTPUT]
                     [-t THRESHOLD]
                     [-m {1,2,3,4}] 
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
                        Model: 1: AAC based SVM, 2: ML+BLAST, 3: ML+MERCI (positive motifs) 4: Hybrid,
                        by default 1
  -d {1,2}, --display {1,2}
                        Display: 1:Urinary proteins, 2: All proteins, by
                        default 2

```

> **Input File**: Allows users to provide input in two format; i) FASTA format (standard) (e.g. peptide.fa). ii) If input is not in FASTA,  each line is considered to be a peptide/ Protein sequence.

> **Output File**: If user do not name the output file, by default the output file is named as outfile.csv. 
Program will save result in CSV format, in case user do not provide output file name, it will be stored in outfile.csv.

> **Threshold**: The default threshold = 0.6 and the users may assign the threshold between 0 and 1. 
User should provide threshold between 0 and 1, please note score is proportional to urinary potential of proteins/peptide.

> **Models**: The tool consists of  two models: 
>  * **i) Model-1** _predicts the given input sequences (peptide/protein) as urinary and non-urinary using SVM-RBF based on amino-acid composition of the sequences_.
>  * **ii) Model-2** _predicts the given input sequences (peptide/protein) as urinary and non-urinary using Hybrid approach(SVM + BLAST). The prediction is based on the hybrid score (combined scores of  SVM, and BLAST)_.
>  * **iii) Model-3** _predicts the given input sequences using Hybrid approach(SVM + MERCI (positive motifs)). The prediction is based on the hybrid score (combined scores of  SVM, and MERCI)_.
>  * **iv) Model-4** _predicts the given input sequences using Hybrid approach(SVM + MERCI + BLAST). The prediction is based on the hybrid score (combined scores of  SVM, MERCI, and BLAST)_.


UriPred Package Files
=======================
The brief descriptions of the files of the tool are given below:

LICENSE       	: License information

envfile : The file provide the path information for BLAST and MERCI commands ,and data required to run BLAST and MERCI

Database: The folder contains the blast database

progs : The folder contains the program to run MERCI

README.md     	: The file provide information about the package

uripred.py 	: Main python program

SVM_model        : Model file required for running Machine-learning model


protein.fa	: The file contain protein sequences in FASTA format

# Reference
Dr. Amouda's Lab
