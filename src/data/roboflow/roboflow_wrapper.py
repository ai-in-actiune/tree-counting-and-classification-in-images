import argparse
import yaml
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import io
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
from roboflow import Roboflow

from utils import AttributeDict
from data.config.config_keys import CFG


class RoboflowWrapper:
    CURRENT_PROJECT_NAME = "treecounts"
    
    @classmethod
    def get_proj_name_from_credentials(cls, credentials):
        return credentials.get(CFG.project_name, cls.CURRENT_PROJECT_NAME)
    
    @classmethod
    def get_workspace_name_from_credentials(cls, credentials):
        return credentials.get(CFG.workspace_name, cls.CURRENT_PROJECT_NAME)
    
    @classmethod
    def get_project(cls, credentials):
        # gain access to your workspace
        rf = Roboflow(api_key=credentials.key)
        workspace_name = cls.get_workspace_name_from_credentials(credentials)
        project_name = cls.get_proj_name_from_credentials(credentials)
        workspace = rf.workspace(workspace_name)
        # you can obtain your model path from your project URL, it is located
        # after the name of the workspace within the URL - you can also find your
        # model path on the Example Web App for any dataset version trained
        # with Roboflow Train
        # https://docs.roboflow.com/inference/hosted-api#obtaining-your-model-endpoint
        project = workspace.project(project_name)
        return project

    @classmethod
    def download_dataset(cls, credentials, format="voc"):
        project = cls.get_project(credentials)
        if credentials.version is None:
            proj_version = project.versions()[0]
        else:
            proj_version = project.version(credentials.version)
        dataset = proj_version.download(format)
        return dataset
    
    @classmethod
    def upload_xml(cls, credentials, annotation_filename, rbflw_img_id):
        # https://docs.roboflow.com/adding-data/upload-api
        # Read Annotation as String
        annotation_str = open(annotation_filename, "r").read()
        # Construct the URL
        upload_url = "".join([
            f"https://api.roboflow.com/dataset/{cls.get_proj_name_from_credentials(credentials)}/annotate/{rbflw_img_id}",
            f"?api_key={credentials.key}",
            "&name=", str(annotation_filename)
        ])
        # POST to the API
        r = requests.post(upload_url, data=annotation_str, headers={
            "Content-Type": "text/plain"
        })
    
    @classmethod
    def upload_image(cls, credentials, image_filename, retry_count=1):
        image = Image.open(image_filename).convert("RGB")
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")
        # Construct the URL
        upload_url = "".join([
            f"https://api.roboflow.com/dataset/{cls.get_proj_name_from_credentials(credentials)}/upload",
            f"?api_key={credentials.key}"
        ])
        m = MultipartEncoder(fields={'file': (image_filename.name, buffered.getvalue(), "image/jpeg")})
        r = requests.post(upload_url, data=m, headers={'Content-Type': m.content_type})
        if r.status_code == 200:
            r = r.json()
            if r.get('success') == True or r.get('duplicate') == True:
                return r.get('id', None)
            return None
        retry_count -= 1
        if retry_count > 0:
            return cls.upload_image(credentials, image_filename, retry_count)
        return None
    
    @classmethod
    def upload_the_new_pretagged_batches_from_path(cls, credentials, from_path: Path):
        project = cls.get_project(credentials)
        from_path = Path(from_path)
        for img_path in tqdm(list(from_path.iterdir()), desc=f"Uploading from path {from_path.name}"):
            item_suffix = img_path.suffix.lower()
            if item_suffix in ['.jpg', '.png', '.jpeg']:
                # upload image & get the result. from that result,
                # extract the image id from roboflow; then use it to link the annotation file
                rbflw_img_id = cls.upload_image(credentials, img_path, retry_count=3)
                if rbflw_img_id is None:
                    raise Exception(f"Img {img_path} did not receive an upload id from Roboflow")
                annotation_file = img_path.parent / f"{img_path.stem}.xml"
                if annotation_file.exists():
                    cls.upload_xml(credentials, annotation_file, rbflw_img_id)

    @staticmethod
    def fix_roboflow_specific_issues_on_csv(csv_at_path):
        anno_csv = pd.read_csv(csv_at_path)
        anno_csv[['xmin', 'xmax', 'ymin', 'ymax']] -= 1
        anno_csv.to_csv(csv_at_path, index=False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config_path", type=str, required=True)
    parser.add_argument("-u", "--upload_folder_path", type=str, required=False, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config_path = Path(args.config_path)
    upload_folder_path = Path(args.upload_folder_path) if args.upload_folder_path else None
    
    loaded_config = AttributeDict(yaml.safe_load(open(config_path)))
    if upload_folder_path:
        RoboflowWrapper.upload_the_new_pretagged_batches_from_path(loaded_config.roboflow_credentials,
                                                                   from_path=loaded_config.pretagged_crops_path)
