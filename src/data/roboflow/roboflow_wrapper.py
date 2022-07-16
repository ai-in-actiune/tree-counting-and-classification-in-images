from pathlib import Path
from glob import glob
import pandas as pd
from roboflow import Roboflow

from data.config.config_keys import CFG


class RoboflowWrapper:
    CURRENT_PROJECT_NAME = "treecounts"
    
    @classmethod
    def get_project(cls, credentials):
        # gain access to your workspace
        rf = Roboflow(api_key=credentials.key)
        workspace_name = credentials.get(CFG.workspace_name, cls.CURRENT_PROJECT_NAME)
        project_name = credentials.get(CFG.project_name, cls.CURRENT_PROJECT_NAME)
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
        if credentials.version:
            proj_version = project.version(credentials.version)
        else:
            proj_version = project.versions()[0]
        dataset = proj_version.download(format)
        return dataset
    
    @classmethod
    def upload_the_new_tagging_batches(cls, credentials, from_path: Path):
        project = cls.get_project(credentials)
        for item in glob(from_path):
            project.upload(item, num_retry_uploads=3)

    def fix_roboflow_specific_issues_on_csv(csv_at_path):
        anno_csv = pd.read_csv(csv_at_path)
        anno_csv[['xmin', 'xmax', 'ymin', 'ymax']] -= 1
        anno_csv.to_csv(csv_at_path, index=False)
