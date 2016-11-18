#!/usr/bin/env bash

# Setup.
virtualenv .venv --python=python2.7 --clear > script-noise.log
. .venv/bin/activate >> script-noise.log
pip install pylint >> script-noise.log

# Repo Location.
repo="repo"

# Files to be linted.
find ${repo} \
    | grep "\.py$" \
    | grep -v "_pb2\.py$" \
    > pylint_targets.log
echo -e "\n\e[1mLinting $(cat pylint_targets.log | wc -l) files\e[0m"

# Lint.
cat pylint_targets.log \
    | xargs pylint \
        --reports=no \
        --output-format=colorized \
        --rcfile=${repo}/scripts/continuous_integration/jenkins/.pylintrc_all \
    | tee pylint_warnings.log

# Summarize.
if [ -s pylint_warnings.log ]; then
    failure_count=$(cat pylint_warnings.log | grep -v "\* Module" | wc -l)
    echo -e "\n\e[31mFound ${failure_count} lint failures.\e[0m"
    echo "failure_count=${failure_count}" > build.properties
    # Jenkins determines success via failure_count in build.properties.
else
    echo -e "\e[32mNo problems found.\e[0m"
    touch build.properties
fi
