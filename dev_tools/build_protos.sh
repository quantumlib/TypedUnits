#!/usr/bin/env bash

################################################################################
# Generates python from protobuf definitions, including mypy stubs.
#
# Usage:
#     ./dev_tools/build-protos.sh
################################################################################

set -e
trap "{ echo -e '\033[31mFAILED\033[0m'; }" ERR

# Get the working directory to the repo root.
cd "$(dirname "${BASH_SOURCE[0]}")"
cd "$(git rev-parse --show-toplevel)"

python -m grpc_tools.protoc -I=. --python_out=. --mypy_out=. src/proto/*.proto
