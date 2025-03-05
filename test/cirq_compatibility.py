# Copyright 2025 The TUnits Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import tunits as tu
import cirq


@pytest.mark.cirq
def test_to_json() -> None:
    assert (
        cirq.to_json(tu.ns * 3)
        == """{
  "cirq_type": "tunits.Value",
  "value": 3,
  "unit": "ns"
}"""
    )

    assert (
        cirq.to_json(tu.GHz * [1, 2, 3, -1])
        == """{
  "cirq_type": "tunits.ValueArray",
  "value": [
    1.0,
    2.0,
    3.0,
    -1.0
  ],
  "unit": "GHz"
}"""
    )
