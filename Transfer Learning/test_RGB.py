from PIL import Image     
import os       

path = 'dataset/pizza-hut/train/' 

for file in os.listdir(path):      
    split_file = file.split('.')
    
    extension = split_file[-1]

    if extension != 'xml':
        fileLoc = path+file
        
        with Image.open(fileLoc) as img:
            img_mode = img.mode
        
        if img_mode != 'RGB':
            print('\033[93mRemoving not RGB file\033[0m: ',file+', '+img_mode)

            os.remove(path+file)

            split_file[-1] = 'xml'
            fileXML = '.'.join(split_file)
            print(fileXML)
            os.remove(path+fileXML)