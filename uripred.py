#====================================================================================================
# UriPred: A tool for predicting urinary and non-urinary proteins from their primary sequence.
# Developed by Dr. V. Amouda's group
#====================================================================================================
import argparse  
import warnings
import os
import re
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from pycaret.classification import load_model 

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='Please provide following arguments. Please make the suitable changes in the envfile provided in the folder.') 

# Read Arguments from command
parser.add_argument("-i", "--input", type=str, required=True, help="Input: protein or peptide sequence in FASTA format or single sequence per line in single letter code")
parser.add_argument("-o", "--output",type=str, help="Output: File for saving results by default outfile.csv")
parser.add_argument("-t","--threshold", type=float, help="Threshold: Value between 0 to 1 by default 0.6")
parser.add_argument("-m","--model",type=int, choices = [1, 2], help="Model: 1: AAC based SVM-RBF, 2: Hybrid, by default 1")
parser.add_argument("-d","--display", type=int, choices = [1,2], help="Display: 1:Urinary, 2: Both Urinary & Non-Urinary, by default 2")
args = parser.parse_args()

# Define constants
STD_AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"
DEFAULT_THRESHOLD = 0.6
DEFAULT_MODEL = 1
DEFAULT_DISPLAY = 2
DEFAULT_OUTPUT_FILENAME = "outfile.csv"

#--------------------------------- AAC Calculation ---------------------------------
def aac_comp(file,out):
    std = list(STD_AMINO_ACIDS)
    df1 = pd.DataFrame(file, columns=["Seq"])
    dd = []
    for j in df1['Seq']:
        cc = []
        for i in std:
            count = 0
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
                composition = (count/len(j))*100
            cc.append(composition)
        dd.append(cc)
    df2 = pd.DataFrame(dd)
    head = []
    for mm in std:
        head.append('AAC_'+mm)
    df2.columns = head
    df2.to_csv(out, index=None, header=True)

#--------------------------------- Normalization ---------------------------------  
def standardize_aac(input_file, output_file):
    df = pd.read_csv(input_file)
    scaler = joblib.load('./progs/aac_scaler.pkl')
    X_scaled = scaler.transform(df.values)
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns)
    df_scaled.to_csv(output_file, index=False)

#---------------------------------Prediction---------------------------------    
def prediction(inputfile, model, out):
    clf = load_model(model)  
    data_test = pd.read_csv(inputfile)
    probabilities = clf.predict_proba(data_test)[:, 1]
    pd.DataFrame(probabilities).to_csv(out, index=False, header=False)

#---------------------------------Class Assignment---------------------------------
def class_assignment(file1,thr,out):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(0,len(df1)):
        if df1['ML Score'][i]>=float(thr):
            cc.append('Urinary')
        else:
            cc.append('Non-Urinary')
    df1['Prediction'] = cc
    df1 =  df1.round(3)
    df1.to_csv(out, index=None)

#---------------------------------MERCI Processor for Positive Motifs---------------------------------
def MERCI_Processor_p(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('Non-Urinary')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        #total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])  # Not used
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('Urinary')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Non-Urinary')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_p(merci_processed,final_merci_p):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if int(df5['Hits'][i]) > 0:
            kk.append(0.5)
        else:
            kk.append(0)
    df5["MERCI Score Pos"] = kk
    df5 = df5[['Subject','MERCI Score Pos']]
    df5.to_csv(final_merci_p, index=None)

#---------------------------------MERCI Processor for Negative Motifs---------------------------------
def MERCI_Processor_n(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['1']))
            kk.append('Urinary')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        #total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])  # Not used
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('Non-Urinary')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Urinary')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_n(merci_processed,final_merci_n):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if int(df5['Hits'][i]) > 0:
            kk.append(-0.5)
        else:
            kk.append(0)
    df5["MERCI Score Neg"] = kk
    df5 = df5[['Subject','MERCI Score Neg']]
    df5.to_csv(final_merci_n, index=None)

#---------------------------------BLAST Processor (unchanged)---------------------------------
def BLAST_processor(blast_result,blast_processed,name1):
    if os.stat(blast_result).st_size != 0:
        df1 = pd.read_csv(blast_result, sep="\t",header=None)
        df2 = df1.iloc[:,:2]
        df2.columns = ['Subject','Query']
        df3 = pd.DataFrame()
        for i in df2.Subject.unique():
            df3 = pd.concat([df3, df2.loc[df2.Subject==i][0:5]], ignore_index=True)
        cc= []
        for i in range(0,len(df3)):
            cc.append(df3['Query'][i].split("_")[0])
        df3['label'] = cc
        dd = []
        for i in range(0,len(df3)):
            if df3['label'][i] == 'P':
                dd.append(1)
            else:
                dd.append(-1)
        df3["vote"] = dd
        ff = []
        gg = []
        for i in df3.Subject.unique():
            ff.append(i)
            gg.append(df3.loc[df3.Subject==i]["vote"].sum())
        df4 = pd.concat([pd.DataFrame(ff),pd.DataFrame(gg)],axis=1)
        df4.columns = ['Subject','Blast_value']
        hh = []
        for i in range(0,len(df4)):
            if df4['Blast_value'][i] >0:
                hh.append(0.5)
            elif df4['Blast_value'][i] == 0:
                hh.append(0)
            else:
                hh.append(-0.5)
        df4['BLAST Score'] = hh
        df4 = df4[['Subject','BLAST Score']]
    else:
        ss = []
        vv = []
        for j in name1:
            ss.append(j)
            vv.append(0)
        df4 = pd.concat([pd.DataFrame(ss),pd.DataFrame(vv)],axis=1)
        df4.columns = ['Subject','BLAST Score']
    df4.to_csv(blast_processed, index=None)

#--------------------------------- Hybrid (with both MERCI scores) ---------------------------------
def hybrid(ML_output,name1,merci_output_p, merci_output_n,blast_output,threshold,final_output):
    df6_2 = pd.read_csv(ML_output,header=None)
    df6_1 = pd.DataFrame(name1)
    df5 = pd.read_csv(merci_output_p)
    df4 = pd.read_csv(merci_output_n)
    df_blast = pd.read_csv(blast_output)
    df6 = pd.concat([df6_1,df6_2],axis=1)
    df6.columns = ['Subject','ML Score']
    df6['Subject'] = df6['Subject'].str.replace('>','')
    df7 = pd.merge(df6,df5, how='outer',on='Subject')
    df8 = pd.merge(df7,df4, how='outer',on='Subject')
    df9 = pd.merge(df8,df_blast, how='outer',on='Subject')
    df9.fillna(0, inplace=True)
    numeric_columns = ['ML Score', 'MERCI Score Pos', 'MERCI Score Neg', 'BLAST Score']
    df9['Hybrid Score'] = df9[numeric_columns].sum(axis=1)
    df9 = df9.round(3)
    ee = []
    for i in range(0,len(df9)):
        if df9['Hybrid Score'][i] > float(threshold):
            ee.append('Urinary')
        else:
            ee.append('Non-Urinary')
    df9['Prediction'] = ee
    df9.to_csv(final_output, index=None)

#================================== Read input file ==================================
f=open(args.input,"r")
len1 = f.read().count('>')
f.close()

with open(args.input) as f:
    records = f.read()
records = records.split('>')[1:]
seqid = []
seq = []
for fasta in records:
    array = fasta.split('\n')
    name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
    seqid.append(name)
    seq.append(sequence)
if len(seqid) == 0:
    f=open(args.input,"r")
    data1 = f.readlines()
    for each in data1:
        seq.append(each.replace('\n',''))
    for i in range (1,len(seq)+1):
        seqid.append("Seq_"+str(i))

seqid_1 = list(map(">{}".format, seqid))
CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
f.close()

# Output file 
if args.output == None:
    result_filename= DEFAULT_OUTPUT_FILENAME
else:
    result_filename = args.output
         
# Threshold 
if args.threshold == None:
    Threshold = DEFAULT_THRESHOLD
else:
    Threshold= float(args.threshold)
# Model
if args.model == None:
    Model = DEFAULT_MODEL
else:
    Model = int(args.model)
# Display
if args.display == None:
    dplay = DEFAULT_DISPLAY
else:
    dplay = int(args.display)

#====================================================================================================
print(" UriPred: Urinary and Non-Urinary Protein Predictor")
print(" Developed by Dr. V. Amouda's group")
print('====================================================================================================')
print('Parameters Summary:')
print(f"  Input File:  {args.input}")
print(f"  Model:       {'Hybrid' if Model == 2 else 'ML'}")
print(f"  Threshold:   {Threshold}")
print(f"  Display:     {'Urinary Only' if dplay == 1 else 'Both Urinary & Non-Urinary'}")
print(f"  Output File: {result_filename}")
print('====================================================================================================')

#================================ Prediction Module start from here ================================
if Model==1:
    print("Running ML model...")
    aac_comp(seq,'seq.aac')
    os.system("perl -pi -e 's/,$//g' seq.aac")
    standardize_aac('seq.aac', 'seq.aac')
    prediction('seq.aac','./progs/SVM_model','seq.pred')
    class_assignment('seq.pred',Threshold,'seq.out')
    df1 = pd.DataFrame(seqid)
    df2 = pd.DataFrame(seq)
    df3 = pd.read_csv("seq.out")
    df3 = round(df3,3)
    df4 = pd.concat([df1,df2,df3],axis=1)
    df4.columns = ['ID','Sequence','ML_Score','Prediction']
    if dplay == 1:
        df4 = df4.loc[df4.Prediction=="Urinary"]
    df4.to_csv(result_filename, index=None)
    os.remove('seq.aac')
    os.remove('seq.pred')
    os.remove('seq.out')
else:
    print("Running Hybrid model...")
    if os.path.exists('envfile'):
        with open('envfile', 'r') as file:
            data = file.readlines()
        output = []
        for line in data:
            if not "#" in line:
                output.append(line)
        if len(output) >= 5: 
            paths = []
            for i in range(len(output)):
                paths.append(output[i].split(':',1)[1].replace('\n',''))
            blastp = paths[0].strip()
            blastdb = paths[1].strip()
            merci = paths[2].strip()
            motifs_p = paths[3].strip()
            motifs_n = paths[4].strip()
        else:
            print("====================================================================================================")
            print("Error: Please provide paths for BLAST, MERCI and required files", file=sys.stderr)
            print("====================================================================================================")
            sys.exit()
    else:
        print("====================================================================================================")
        print("Error: Please provide the '{}', which comprises paths for BLAST and MERCI".format('envfile'), file=sys.stderr)
        print("====================================================================================================")
        sys.exit()
    aac_comp(seq,'seq.aac')
    os.system("perl -pi -e 's/,$//g' seq.aac")
    standardize_aac('seq.aac', 'seq.aac')
    prediction('seq.aac','./progs/SVM_model','seq.pred')
    print("Running BLAST...")
    os.system(f"{blastp} -task blastp -db {blastdb} -query Sequence_1 -out RES_1_6_6.out -outfmt 6 -evalue 0.000001")
    print("Running MERCI (positive motifs)...")
    os.system(f"{merci} -p Sequence_1 -i {motifs_p} -o merci_p.txt")
    print("Running MERCI (negative motifs)...")
    os.system(f"{merci} -p Sequence_1 -i {motifs_n} -o merci_n.txt")
    MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
    Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
    MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
    Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
    BLAST_processor('RES_1_6_6.out','blast_hybrid.csv',seqid)
    hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv','blast_hybrid.csv',Threshold,'final_output')
    df44 = pd.read_csv('final_output')

    # Add sequence data to the result DataFrame
    df_seq = pd.DataFrame({'Subject': seqid, 'Sequence': seq})
    df44 = pd.merge(df44, df_seq, on='Subject', how='left')

    # Reorder columns to have Sequence after Subject
    cols = list(df44.columns)
    seq_idx = cols.index('Sequence')
    cols.insert(1, cols.pop(seq_idx))
    df44 = df44[cols]

    if dplay == 1:
        df44 = df44.loc[df44.Prediction=="Urinary"]
    df44 = round(df44,3)
    df44.to_csv(result_filename, index=None)
    os.remove('seq.aac')
    os.remove('seq.pred')
    os.remove('final_output')
    os.remove('RES_1_6_6.out')
    os.remove('merci_output_p.csv')
    os.remove('merci_output_n.csv')
    os.remove('merci_hybrid_p.csv')
    os.remove('merci_hybrid_n.csv')
    os.remove('blast_hybrid.csv')
    os.remove('merci_p.txt')
    os.remove('merci_n.txt')
    os.remove('Sequence_1')

print("="*100)
print('\nThanks for using UriPred. Your results are stored in file :',result_filename,'\n')
print('Please cite: UriPred\n')