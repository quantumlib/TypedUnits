#!/usr/bin/env bash

# Repo Location.
repo="."

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
        --rcfile=${repo}/.pylintrc_all
