FROM python:3.8-slim-buster

COPY requirements.txt ./

RUN pip install -r requirements.txt

ENV BASE_DIR="/"

RUN mkdir ${BASE_DIR}/docker_app_data
RUN mkdir /usr/local/lib/nltk_data

COPY app.py ${BASE_DIR}/docker_app_data/
COPY score.py ${BASE_DIR}/docker_app_data/
COPY lr_best_model.sav ${BASE_DIR}/docker_app_data/
COPY word_vec.sav ${BASE_DIR}/docker_app_data/
COPY tf_idf_trans.sav ${BASE_DIR}/docker_app_data/
COPY ./template ${BASE_DIR}/docker_app_data/template
COPY ./data ${BASE_DIR}/docker_app_data/data
COPY ./nltk_data /usr/local/lib/nltk_data

WORKDIR ${BASE_DIR}/docker_app_data

ENTRYPOINT python ${BASE_DIR}/docker_app_data/app.py
# CMD ["/bin/bash"]