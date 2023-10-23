import numpy as np
import pandas as pd
from  dtw_cf import *
import sys
from sklearn import preprocessing



def encode_label(data):
    # Assuming that the input data in a dataframe with the first column being the label, encode the labels with int (0,1,2,3...) and not string
    le = preprocessing.LabelEncoder()
    le.fit(data[0])
    data[0] = le.transform(data[0])
    print("Labels encoded")
    return data

def load_ds(ds, folder = "data/"):
    # Load a dataset from a csv file
    data = pd.read_csv(folder + "/{}/{}_TRAIN.tsv".format(ds,ds) ,sep='\t', header =None)
    data_test = pd.read_csv(folder + "/{}/{}_TEST.tsv".format(ds,ds) ,sep='\t', header =None)
    
    data = encode_label(data)
    data_test = encode_label(data_test)
    return data, data_test
    

def raise_usage():
    usage = """
Run command : python main.py <OPTION>

<OPTION> :
    --get_info_dataset : affiche infos des fichiers d'entrainement (nb de samples, compte pour chaque label, imbalence degree).
    --make_tab <DATASET_NAME> <MODEL>("RF","NN","DTW_NEIGBOURS","TS-RF") <NB_ITERATION> : Construit les tableaux pour un dataset et un model donné
    --multi_test : make_tab itéré sur différents datasets et différents model (modifier les listes dans le fichier main.py)
    --draw_example_minority <DATASET_NAME> : plot min(3,cnt_minority) exemples de la time series de la classe minoritaire du dataset
            """
    print(usage)
    exit()



if __name__ == '__main__':
    
    try :
        option = sys.argv[1]
    except Exception as e:
        raise_usage()
        
    if '--ds' in option:
        ds_name = sys.argv[2]
    else:
        ds_name = 'ECG200'
        
    
    
    data, data_test = load_ds(ds_name)
    cf_augment(data, initializer = "random", mode = 1, draw_pca=True, draw_tsne=True) 