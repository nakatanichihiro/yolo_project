import xml.etree.ElementTree as ET 
import os 
import pickle

# read label dictionary
with open("label_dic.pickle", "rb") as f:
    label_dic = pickle.load(f)

# get dir path list of annotation_data
os.chdir('../')
current_path = os.getcwd()
dirs_path_list = [i for i in os.listdir(os.path.join(current_path, r"annotation_data"))]

# think one dir
for dir_path in dirs_path_list:
    files_path_list = [i for i in os.listdir(os.path.join(os.path.join(os.path.join(current_path, r"annotation_data"), dir_path), "Annotations"))]

    # think one files
    for file_path in files_path_list:
        # analysis xml file
        tree = ET.parse(os.path.join(os.path.join(os.path.join(os.path.join(current_path, r"annotation_data"), dir_path), "Annotations"), file_path)) 

        # save coordinate value
        coord_dic = {}

        # XMLを取得
        root = tree.getroot()

        for ele_1 in root.iter("xmin"):
            xmin = ele_1.text
            coord_dic["xmin"] = xmin

        for ele_2 in root.iter("xmax"):
            xmax = ele_2.text
            coord_dic["xmax"] = xmax

        for ele_3 in root.iter("ymin"):
            ymin = ele_3.text
            coord_dic["ymin"] = ymin

        for ele_4 in root.iter("ymax"):
            ymax = ele_4.text
            coord_dic["ymax"] = ymax

        for label_ele in root.iter("name"):
            label = label_dic[label_ele.text]

        save_path = os.path.join(os.path.join(os.path.join(os.path.join(current_path, r"annotation_data"), dir_path), "JPEGImages"), file_path)
        save_txt = save_path + " " + coord_dic["xmin"] + "," + coord_dic["ymin"] + "," + coord_dic["xmax"] + "," + coord_dic["ymax"] + ","  + str(label)

        with open("xml_analysis/train_data.txt", "a") as d:
            d.write(save_txt + "\n")
        