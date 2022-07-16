class CFG:
    # Input Train
    pretrained_path="pretrained_path"
    gpus = "gpus"
    epochs = "epochs"
    batch_size = "batch_size"
    save_snapshot = "save-snapshot"
    train_annotations = "train_annotations"
    valid_annotations = "valid_annotations"
    crops_path = "crops_path"
    
    # Output Train
    model_output_folder_path = "model_output_folder_path"
    # Predict
    unlabeled_crops_path = "unlabeled_crops_path"
    pretagged_crops_path = "pretagged_crops_path"
    
    # Dataset
    splits = "splits"
    
    # Neptune
    neptune_enabled = "neptune_enabled"
    neptune_proj_name = "neptune_proj_name"
    neptune_api_token = "neptune_api_token"
    
    # LabelBox
    labelbox_enabled = "labelbox_enabled"
    labelbox_credentials = "labelbox_credentials"

    # Roboflow
    roboflow_enabled = "roboflow_enabled"
    roboflow_credentials = "roboflow_credentials"
    workspace_name = "workspace_name"
    project_name = "project_name"
    version = "version"
