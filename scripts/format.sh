#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
readonly REPO_ROOT

args=( "$@" )
if [ "$#" -eq 0 ]; then
  args=("$REPO_ROOT")
elif [ "$#" -eq 1 ] && [ "$1" = "--check" ]; then
  args=("--check" "$REPO_ROOT")
fi

black "${args[@]}"
