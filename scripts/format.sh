#!/usr/bin/env bash

set -e

readonly repo_root=$(git rev-parse --show-toplevel)

args=( "$@" )
if [ "$#" -eq 0 ]; then
  args=("$repo_root")
elif [ "$#" -eq 1 ] && [ "$1" = "--check" ]; then
  args=("--check" "$repo_root")
fi

black "${args[@]}"
