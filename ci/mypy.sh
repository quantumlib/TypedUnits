#!/bin/bash

set -Eeuo pipefail

shopt -s globstar

mypy ./**/*.py
