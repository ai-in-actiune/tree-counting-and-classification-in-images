import os
import argparse
import neptune.new as neptune
from pytorch_lightning.loggers import NeptuneLogger


NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN", default="")
NEPTUNE_PROJ_NAME = os.getenv("NEPTUNE_PROJ_NAME", default="octavf/tree-counting-and-classif")


class NeptuneWrapper():
    def __init__(self, proj_name, api_token) -> None:
        self.proj_name = proj_name
        self.api_token = api_token
    
    def init_neptune_project(self, read_mode="read-only"):
        nep_project = neptune.init_project(name=self.proj_name,
                                           api_token=self.api_token,
                                           mode=read_mode)
        return nep_project

    # DATASET

    def upload_dataset(self, from_local_path='./dataset/', to_neptune_path='dataset/version/'):
        nep_project = self.init_neptune_project(read_mode="async")
        nep_project[to_neptune_path].track_files(from_local_path)
        nep_project.wait()
        nep_project["dataset/latest"] = nep_project[to_neptune_path].fetch()
        nep_project.stop()
    
    def download_dataset_version(self, from_neptune_path="dataset/0.1", to_local_path='./dataset'):
        nep_project = self.init_neptune_project()
        nep_project.wait()
        nep_project[from_neptune_path].download(to_local_path)
        nep_project.stop()
        
    def download_latest_dataset(self, to_local_path='./dataset'):
        self.download_dataset_version(from_neptune_path="dataset/latest", to_local_path=to_local_path)


    # EXPERIMENTS
    
    @staticmethod
    def get_pytorch_lightning_logger(proj_name=NEPTUNE_PROJ_NAME, api_token=NEPTUNE_API_TOKEN):
        neptune_logger = NeptuneLogger(api_key=api_token, project=proj_name,
                                       tags=["training", "deeptree"])
        return neptune_logger
    
    def upload_experiment(self, model, precision, recall):
        nep_project = self.init_neptune_project(read_mode="async")
        nep_project['metrics/valid/precision'] = precision
        nep_project['metrics/valid/recall'] = recall
        nep_project.wait()
        nep_project.stop()


def run(proj_name, api_token,
        upload_ds=False, download_ds=False,
        from_path=None, to_path=None):
    neptunewrap = NeptuneWrapper(proj_name,
                                 api_token)
    if upload_ds:
        neptunewrap.upload_dataset(from_local_path=from_path,
                                   to_neptune_path=to_path)
    elif download_ds:
        if from_path is None:
            neptunewrap.download_latest_dataset(to_local_path=to_path)
        else:
            neptunewrap.download_dataset_version(from_neptune_path=from_path,
                                                 to_local_path=to_path)  
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-proj", "--proj_name", type=str, required=True)
    parser.add_argument("-tok", "--api_token", type=str, required=True)
    
    parser.add_argument("-dd", "--download_dataset", type=str, action="store_true",
                        help="Just a flag argument. Where action='store_true' implies default=False.")
    parser.add_argument("-ud", "--upload_dataset", type=str, action="store_true",
                        help="Just a flag argument. Where action='store_true' implies default=False.")
    parser.add_argument("-from", "--from_path", type=str, required=False)
    parser.add_argument("-to", "--to_path", type=str, required=False)
    
    return parser.parse_args()


if __name__ == '__main__':
    # run_test()
    args = get_args()
    run(args.proj_name, args.api_token,
        upload_ds=args.upload_dataset, download_ds=args.download_dataset,
        from_path=args.from_path, to_path=args.to_path)
    