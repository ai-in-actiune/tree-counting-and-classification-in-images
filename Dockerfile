FROM python:3.8
MAINTAINER Mosnoi Ion <moshnoi2000@gmail.com>

RUN apt-get update

WORKDIR "/work"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#
# Jupyter Notebook
#
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN pip3 --no-cache-dir install jupyter && \
    mkdir /root/.jupyter && \
    echo "c.NotebookApp.ip = '*'" \
         "\nc.NotebookApp.open_browser = False" \
         "\nc.NotebookApp.token = ''" \
         > /root/.jupyter/jupyter_notebook_config.py
         
 
EXPOSE 8888

RUN pip3 install --upgrade deepforest albumentations pyyaml

CMD ["jupyter", "notebook", "--allow-root"]
