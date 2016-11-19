#!/usr/bin/env bash

# Checkout location.
repo="repo"

virtualenv .venv --clear --system-site-packages > script-noise.log
. .venv/bin/activate >> script-noise.log
pip install --upgrade pip >> script-noise.log
pip install -r ${repo}/requirements.txt >> script-noise.log

export PYTHONPATH=${repo}:${PYTHONPATH}

# The report is consumed by the "Post-Build" XUnit action in Jenkins.
cd ${repo}
python setup.py build_ext --inplace
pytest --junitxml report.xml test/
