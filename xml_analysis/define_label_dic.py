import pickle

import xml.etree.ElementTree as ET 
import os 

# get dir path list of annotation_data
os.chdir('../')
current_path = os.getcwd()
dirs_path_list = [i for i in os.listdir(os.path.join(current_path, r"annotation_data"))]

# define dic for label
label_dic = {}

# think one dir
for dir_path in dirs_path_list:
    files_path_list = [i for i in os.listdir(os.path.join(os.path.join(os.path.join(current_path, r"annotation_data"), dir_path), "Annotations"))]

    # think one files
    for file_path in files_path_list:
        # analysis xml file
        tree = ET.parse(os.path.join(os.path.join(os.path.join(os.path.join(current_path, r"annotation_data"), dir_path), "Annotations"), file_path)) 

        # XMLを取得
        root = tree.getroot()

        for ele in root.iter("name"):
            if ele.text in label_dic.keys():
                pass
            else:
                label_dic[ele.text] = len(label_dic) 

# save label dic
with open("xml_analysis/label_dic.pickle", "wb") as f:
    pickle.dump(label_dic,f)
    