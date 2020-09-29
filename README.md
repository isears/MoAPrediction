## First-time Setup

Make sure you're running some version of python3:
```bash
$ python --version
Python 3.8.5
```

Install dependencies (might help to do this in a [virtualenv](https://docs.python.org/3/library/venv.html) if things don't work out-of-the-box the first time):
```bash
$ pip install -r requirements.txt
```

Make data/logs/models directory
```bash
$ mkdir data/
$ mkdir data/cache/
$ mkdir models/
$ mkdir logs/
```

Put .npy files in `data/cache/`, put .h5 files in `models/`

## Run

```bash
$ python NonscoredEnsemble.py
...
```

This step may take several hours and should consume significant GPU resources. If the GPU isn't running, tensorflow may have to be specifically configured to use it.

Hyperparameters can be modified by changing the capitalized variables at the top of `NonscoredEnsemble.py`
