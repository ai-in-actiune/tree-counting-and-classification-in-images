# How to use Data Version Control (DVC):
###  Install dvc:
```
pip install dvc[gdrive]
```
### Set up a remote storage:
> The next command is used only once
```
dvc remote add -d name gdrive://g_drive_folder_id
```
###  Start tracking a file or directory:
```
dvc add path/to/file
```
### Upload dataset/model to remote storage:
```
dvc push
```
> To download dvc-tracked data and/or models run:
```
dvc pull
```