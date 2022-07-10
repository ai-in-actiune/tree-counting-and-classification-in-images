tree-counting-and-classification-in-images
==============================
proiect bazat pe învățare automată care își propune să detecteze și să clasifice copacii din imagini făcute de drone. 

problema: mulți copaci verzi sunt tăiați ilegal, copacii uscați nu sunt tăiați la timp din cauză că nu sunt resurse umane pentru a monitoriza bucăți mari de pădure.

solutie: crearea unui sistem care sa invete sa detecteze si sa clasifice copaci rapid avand un set de date nou fara etichete. Sistemul va fi folosit:
* pentru a eticheta semi-automat datele folosind modelele ML existente
* sa antreneze modele noi folosind datele noi etichetate
* sa identifice etichetele gresite

## resurse utile:
- model(pytorch): https://github.com/weecology/DeepForest
- [colab model](https://colab.research.google.com/drive/1gKUiocwfCvcvVfiKzAaf6voiUVL2KK_r?usp=sharing#scrollTo=f8MKNC3_Zrxk)
- set de date: https://github.com/weecology/NeonTreeEvaluation
- project template: https://github.com/drivendata/cookiecutter-data-science

## get started

### install
#### virtualenv
```shell
apt install virtualenv
virtualenv -p python3 treedetect
source treedetect/bin/activate
pip install -r requirements.txt
```
#### docker
```shell
docker build -t treedetect:1 -f Dockerfile .
docker run -it -d -p 8800:8888 -p 6000:6006  -v "$(pwd)":/work --name treedetectbox treedetect:1
```
go to http://localhost:8800

#### download data
```shell
python src/make_dataset.py
```
## Contributing

This project is built by amazing volunteers and you can be one of them! Here's a list of ways in [which you can contribute to this project](CONTRIBUTING.md).

If you want to make any change to this repository, please **make a fork first**.

If you see something that doesn't quite work the way you expect it to, open an Issue. Make sure to describe what you _expect to happen_ and _what is actually happening_ in detail.

If you would like to suggest new functionality, open an Issue and mark it as a __[Feature request]__. Please be specific about why you think this functionality will be of use. If you can, please include some visual description of what you would like the UI to look like, if you are suggesting new UI elements. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


## Run & Development
### Base(main) environment 
#### Requirements
```
python 3.6
pip
```
#### Installation
```shell
pip install -U virtualenv

python -m virtualenv -p=python tree-count
```
`./tree-count/bin/activate` **linux based systems**

`./tree-count/Scripts/activate` **win32 based systems**
```shell
pip install -r requirements.txt 
```

### Image processing

For image processing we use [`labelImg`](https://github.com/tzutalin/labelImg).

#### Installation

In order not to pollute global python environment,
it is advised using virtual environment tools(`virtualenv, pipenv, venv, anaconda etc.`)

There can be used th main project environment or there can be created a new separate environment.

##### Create new environment

`python -m virtualenv -p=python labelImg`

`pip install labelImg`

##### Activate base environment

`path/to/base/environment/bin/activate` **linux based systems**

`path\to\base\environment\Scrips\activate` **win32 based systems**

```
pip install labelImg
python -m labelImg
```
## Feedback

* Request a new feature on GitHub.
* Vote for popular feature requests.
* File a bug in GitHub Issues.
* Email us with other feedback aiinactiune@gmail.md

## License

This project is licensed under the MPL 2.0 License - see the [LICENSE](LICENSE) file for details

## About IAinActiune

Started in 2020, IAinActiune is a civic tech NGO. We have a community of over 100 volunteers (ml engineer, entrepreneurs, programmers, project managers and more) who work pro-bono for developing ML solutions to solve social problems. #mlforsocialgood. If you want to learn more details about our projects [visit our site](https://www.iainactiune.md/) or if you want to talk to one of our staff members, please e-mail us at aiinactiune@gmail.md.

Last, but not least, we rely on donations to ensure the infrastructure, logistics and management of our community that is widely spread across 11 timezones, coding for social change to make Moldova and the world a better place. If you want to support us, [you can do it here](https://iainactiune.md/).


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
