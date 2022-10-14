# CSE 591 project


### Installation 

`git submodule update --init --recursive` to initialize submodules

Install [Python 3.10](https://www.python.org/downloads/)  
Optionally install [poetry](https://python-poetry.org/) for package management.  
Optionally run `poetry update` to initialize and install all dependencies.  
Alternatively, you can use `poetry export -f requirements.txt --output requirements.txt` to install with conda, pip, or venv directly.  

### Usage

The main entrypoint is the `./run` bash script. By default, this runs the _script_ `scripts/main.py`, or if provided with an argument, such as `./run download`, then it runs the corresponding script, e.x. `scripts/download.py`.

To test SATNet, first use `./run download` and then `./run`
