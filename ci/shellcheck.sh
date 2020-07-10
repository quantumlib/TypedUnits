#!/bin/bash

set -Eeuo pipefail

# -printf to strip leading ./
readarray -t shell_files < <(find . -iname '*.sh' -printf '%P\n')

echo "Shellchecking these files:"
echo "${shell_files[@]}" | tr ' ' '\n' | sed 's/^/    /'

echo

/usr/local/bin/shellcheck --external-sources "${shell_files[@]}"
