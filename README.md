# Stroke Prediction
In this project, we attempt to classify patients to find out if they will have a stroke or not.

## Dataset
This dataset can be obtained from Kaggle and can be found in this [link](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). In the dataset, we have 5110 datapoints (rows) and each row provides the following information on each patient:
-  id: unique identifier
- gender: "Male", "Female" or "Other"
- age: age of the patient
- hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
- heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
- ever_married: "No" or "Yes"
- work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
- Residence_type: "Rural" or "Urban"
- avg_glucose_level: average glucose level in blood
- bmi: body mass index
- smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
- stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

## Repo Structure
- `dcerejio-players-scores`: full dataset downloaded from data.world
- `scripts`: contains formating script for py-files
- `Makefile`: Makefile where you can create a virtual env or create a docker container to run the notebook
- `Dockerfile`: Dockerfile to build a docker image with GPU support (tested on NVIDIA TITAN RTX with Ubuntu 18.04). More information on how to download docker can be found [here](https://docs.docker.com/get-docker/). nvidia-docker installation can also be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- `StrokePrediction.ipynb`: Notebook containing all the code
- `poetry.lock`/`pyproject.toml`: Python Poetry files to install a development environment with all dependencies handled
- `envrionment.yml`: YAML file containing conda exported environment for development (you can use either poetry or conda)

## Running the Notebook
Clone and `cd` in the repository before running any of the commands:
```bash
git clone git@github.com:charbel-a-hC/ups-ml-stroke-prediction.git
cd ups-ml-stroke-prediction
```
You also need to install `python3.8` locally if you wish to run the notebook on a **local** environment. For Ubuntu:
```bash
sudo apt-get install python3.8 \
    python3-pip \
    python3.8-venv \
    python3.8-dev \
    python3.8-distutils
```
And you need to update your `pip`:
```bash
/usr/bin/python3.8 -m pip install --upgrade pip
```
### Docker
If you have docker installed:
```bash
docker build . -t ups-ml-stroke-prediction
docker run -it --rm -v --runtime=nvidia ${PWD}:/ups-ml-stroke-prediction ups-ml-stroke-prediction bash
```
After launching the container, a notebook will be open in the following ip/port; `localhost:8888`.

### Local Environment (Ubuntu-18.04) - Poetry

Simply run the make command in the repository:
```bash
make env
```
A virtual environment will be created after running the above command. In the same shell, run:
```bash
poetry shell
```
This will activate the environment and then you can open a jupyter notebook:
```bash
jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```

### Local Environment - Conda
You can download Anaconda [here](https://docs.anaconda.com/anaconda/install/index.html).
After the download, open an anaconda navigator prompt if you're on windows and run the following commands:
```bash
conda env create -f environment.yml
conda activate ml
```
**Note**: If you're on Linux, you can open a normal terminal and run the following command before creating the environment:
```bash
conda activate base
```
You can then open a jupyter notebook similarly: 
```bash
jupyter notebook
```
### Google Colaboratory
You can open the notebook in Google Colab here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()
