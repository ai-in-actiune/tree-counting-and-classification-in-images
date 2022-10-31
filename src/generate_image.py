import os
import cv2 
import cvzone 
import logging
import pandas as pd
from random import randrange

def generate_image(backgroundPath, objectImagePath, destinationFolder=None, label=None):
    '''
    Parameters
    ----------
    backgroundPath: str,  background image path
    objectImagePath: str, object image path
    destinationFolder: str, destination folder for generted images
    label: str, image label
    
    Function returns
    ----------------
    - Image with added object  
    - Dataframe with image name, position, label
    
    Examples
    --------
    backgroundPath = 'C:/path/to/background/image'
    objectImagePath = 'C:/path/to/tree/image'
    
    data = add_tree(backgroundPath, objectImagePath)
         image_path        xmin    ymin    xmax    ymax    label
    0    OSBS_24100.jpg    120     155     204     243     Tree
    1    OSBS_6174.jpg     308     255     392     343     Tree
    2    OSBS_33967.jpg    259     284     343     372     Tree
    '''
    
    if label is None: label = 'Tree'
    if destinationFolder is None: destinationFolder = f'{os.getcwd()}\\Out_folder'
    if not os.path.exists(destinationFolder):
        os.makedirs(destinationFolder)
        
    # Create lists for image data
    image_paths = []
    xmin_l = []
    ymin_l = []
    xmax_l = [] 
    ymax_l = [] 
    labels = []
    
    # Iterate backgrounds
    for bg in os.listdir(backgroundPath):
        try:
            # Open background image
            imgback = cv2.imread(os.path.join(backgroundPath,bg)) 
            
            # Get background shape
            bg_h, bg_w, bg_channels = imgback.shape

            # Iterate trees
            for tree in os.listdir(objectImagePath):
                try:
                    # Open tree image
                    imgfront = cv2.imread(os.path.join(objectImagePath,tree), cv2.IMREAD_UNCHANGED) 
                    
                    # Get tree shape
                    fg_h, fg_w, fg_channels = imgfront.shape
                    
                    # Get positions
                    x_position = randrange(bg_w - fg_w) 
                    y_position = randrange(bg_h - fg_h)
                    
                    # Magic
                    imgresult = cvzone.overlayPNG(imgback, imgfront,[x_position,y_position] ) 
                    
                    # Save image
                    new_name = f'OSBS_{randrange(50000)}'
                    cv2.imwrite(os.path.join(destinationFolder, f'{new_name}.jpg'), imgresult)
                    
                    # Append new data
                    image_paths.append(f'{new_name}.jpg')
                    xmin_l.append(x_position)
                    ymin_l.append(y_position)
                    xmax_l.append(x_position + fg_w) 
                    ymax_l.append(y_position + fg_h) 
                    labels.append(label)
                     
                except:
                    logging.warning(f'Error loading file {tree}')
                    pass
                
        except:
            logging.warning(f'Error loading file {bg}')
            pass
            
            
    data = {
        'image_path': image_paths, 
        'xmin': xmin_l,
        'ymin': ymin_l, 
        'xmax': xmax_l,
        'ymax': ymax_l, 
        'label': labels,
    }
        
    return pd.DataFrame(data)