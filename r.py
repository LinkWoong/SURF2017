import os
import re
import shutil
import csv

#extract the photos into a folder named "Container"
photoRegex = re.compile(r'.png|.jpg')
root_path = '/media/linkwong/Work/PixivUtil2/filter'
list_dir = os.listdir(root_path)
Container_path = os.path.join(root_path,'Container')
if not os.path.exists(Container_path):
    os.mkdir(os.path.join(root_path,'Container'))
for folder in list_dir:
    if folder != 'Container':
        folder_path = os.path.join(root_path,folder)
        if os.path.isdir(folder_path):
            folder_dir = os.listdir(folder_path)
            for file in folder_dir:
                file_path = os.path.join(folder_path,file)
                if os.path.isfile(file_path) and photoRegex.search(file) != None:
                    shutil.copy(file_path, os.path.join(root_path,'Container'))

#Rename the photos by deleting non-digit parts
#some bugs, move this script to the Content folder
renameRegex = re.compile(r'^\d*')
Container_dir = os.listdir(Container_path)
for image in Container_dir:
    new_name = renameRegex.search(image)
    os.rename(image,new_name.group()+image[-4:])

#name the front entry with the back entry
data_file = open(os.path.join(Container_path,'01.csv'))
data_reader = csv.reader(data_file)
data_list = list(data_reader)
row_count = len(data_list)
for i in range(row_count):
    colored = data_list[i][0]
    sketch = data_list[i][1]
    if (os.path.exists(os.path.join(Container_path,colored+'.jpg'))):
        os.rename(colored+'.jpg',sketch+'.jpg')
    elif (os.path.exists(os.path.join(Container_path,colored+'.png'))):
        os.rename(colored+'.png',sketch+'.png')
    else:
        continue

#remove the photos that has only one entry
file_dir = os.listdir(Container_path)
dic = {}
for file in file_dir:
    if photoRegex.search(file) != None:
        if file[:-4] not in dic:
            dic[file[:-4]] = 1
        else:
            dic[file[:-4]] = dic[file[:-4]] + 1
for (entry,freq) in dic.items():
    if freq != 2:
        if os.path.exists(os.path.join(Container_path,entry+'.jpg')):
            os.remove(entry+'.jpg')
        else:
            os.remove(entry+'.png')

print("Succeed!")