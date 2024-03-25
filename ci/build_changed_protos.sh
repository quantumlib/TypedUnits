#!/usr/bin/env bash

################################################################################
# Compares the auto generated code from protos to what should be generated.
#
# Usage:
#     ./ci/build_changed_protos.sh
################################################################################

# Get the working directory to the repo root.
thisdir="$(dirname "${BASH_SOURCE[0]}")" || exit $?
topdir="$(git -C "${thisdir}" rev-parse --show-toplevel)" || exit $?
cd "${topdir}" || exit $?

rev="origin/main"

base="$(git merge-base "${rev}" HEAD)"
if [ "$(git rev-parse "${rev}")" == "${base}" ]; then
    echo -e "Comparing against revision '${rev}'." >&2
else
    echo -e "Comparing against revision '${rev}' (merge base ${base})." >&2
    rev="${base}"
fi

# All the protos.
echo "Building protos in $PWD"

dev_tools/build_protos.sh

# Filenames with spaces will be ugly (each part will be listed separately)
# but the error logic will still work.
uncommitted=$(git status --porcelain 2>/dev/null | grep -E "^?? src/proto" | cut -d " " -f 3)

if [[ -n "$uncommitted" ]]; then
    echo -e "\033[31mERROR: Uncommitted generated files found! Please generate and commit these files using dev_tools/build-protos.sh:\033[0m"
    for generated in $uncommitted
    do
        echo -e "\033[31m   ${generated}\033[0m"
    done
    exit 1
fi
