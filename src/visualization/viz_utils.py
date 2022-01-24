from dict2xml import dict2xml
from typing import Union
from pathlib import Path
import pandas as pd
    

def annotation_row_to_dict(annotation_row):
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
    xml_out += "\n".join([annotation_row_to_dict(row) for _, row in annotations_df.iterrows()])
    xml_out += '\n</annotation>\n'
    
    if write_file:
        # annotations file should be near its image
        file_path = image_path.parent / f'{image_path.stem}.xml'
        with open(file_path, 'w+') as the_file:
            the_file.write(xml_out)
    
    return xml_out
