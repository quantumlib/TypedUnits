#!/usr/bin/env bash

# Repo Location.
repo="."

# Files to be linted.
find ${repo} \
    | grep "\.py$" \
    | grep -v "_pb2\.py$" \
    > pylint_targets.log
echo -e "\n\e[1mLinting $(wc -l < pylint_targets.log) files\e[0m"

# Lint.
xargs -a pylint_targets.log pylint \
        --reports=no \
        --output-format=colorized \
        --rcfile=${repo}/.pylintrc_all
