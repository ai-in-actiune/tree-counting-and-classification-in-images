from dict2xml import dict2xml
from typing import Union
from pathlib import Path
import pandas as pd
from glob import glob
from tqdm import tqdm

from deepforest import get_data, utilities
    

def __annotation_row_to_dict(annotation_row):
    """
    Converts row from the dataframe representing the bbox of an object
    to a dict. A dict which is ready to be converted to xml,
    for LabelImg usage.
    
    <object>
		<name>tree</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>718</xmin>
			<ymin>603</ymin>
			<xmax>792</xmax>
			<ymax>705</ymax>
		</bndbox>
	</object>
    
    """
    annotation_object = {
        'name': annotation_row['label'],
        'bndbox': {
            'xmin': int(annotation_row['xmin']),
            'ymin': int(annotation_row['ymin']),
            'xmax': int(annotation_row['xmax']),
            'ymax': int(annotation_row['ymax'])
        }
    }
    return dict2xml(annotation_object, wrap='object', indent="   ")


def annotations_to_xml(annotations_df: pd.DataFrame, image_path: Union[str, Path],
                       write_file=True) -> str:
    """
    Load annotations from dataframe (retinanet output format) and
    convert them into xml format (e.g. RectLabel editor / LabelImg).
    Args:
        annotations_df (DataFrame): Format [xmin,ymin,xmax,ymax,label,...]
        image_path: string/Path path to the file where these bboxes are found
        write_file: Writes the xml at the same path as the image it describes.
                    Overwrites the existent file, if any.
    Returns:
        XML
    
    <annotation>
        <folder>unlabeled_imgs</folder>
        <filename>autumn-forest-from-above-2210x1473.jpeg</filename>
        <path>/work/trees/unlabeled_imgs/autumn-forest-from-above-2210x1473.jpeg</path>
        <source>
            <database>Unknown</database>
        </source>
        <size>
            <width>2210</width>
            <height>1473</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
 
        <object>
            <name>tree</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>718</xmin>
                <ymin>603</ymin>
                <xmax>792</xmax>
                <ymax>705</ymax>
            </bndbox>
    	</object>
    
    </annotation>
    """
    image_path = Path(image_path)
    out_dict = {
        'folder': image_path.parent.name,
        'filename': image_path.name,
        'path': str(image_path),
        'segmented': 0
    }
    
    xml_out = '<annotation>\n'
    xml_out += dict2xml(out_dict, indent="   ") + '\n'
    xml_out += "\n".join([__annotation_row_to_dict(row) for _, row in annotations_df.iterrows()])
    xml_out += '\n</annotation>\n'
    
    if write_file:
        # annotations file should be near its image
        file_path = image_path.parent / f'{image_path.stem}.xml'
        with open(file_path, 'w+') as the_file:
            the_file.write(xml_out)
    
    return xml_out


def xml_to_annotations(xml_file_path):
    try:
        return utilities.xml_to_annotations(xml_file_path)
    except:
        print("ERROR. defaultig to NO LABELS in image")
        return pd.DataFrame()


def xmls_to_csv_for_train(from_folder_path, to_csv_file):
  xmls_paths = sorted(glob(f"{str(from_folder_path)}/*.xml"))
  accumulator_bboxes_dfs = []
  for xml_path_str in tqdm(xmls_paths,
                           desc=f"Converting xmls to {Path(to_csv_file).name} for train eval"):
    xml_path = Path(xml_path_str)
    xml_as_df = xml_to_annotations(str(xml_path))
    accumulator_bboxes_dfs.append(xml_as_df)
  folder_bboxes_df = pd.concat(accumulator_bboxes_dfs)
  folder_bboxes_df.to_csv(to_csv_file, index=False)
