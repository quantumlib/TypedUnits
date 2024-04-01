# Copyright 2024 The TUnits Authors
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

from typing import TypeVar
import numpy as np

from src.proto import tunits_pb2

T = TypeVar('Value', bound='Value')


class Value(WithUnit):
    """A floating-point value with associated units."""

    @classmethod
    def from_proto(cls: type[T], msg: tunits_pb2.Value) -> T:
        if not msg.HasField('real_value'):
            raise ValueError(f"{msg=} doesn't have a value.")
        return cls(msg.real_value, _proto_to_units(msg.units))

    def to_proto(self, msg: Optional[tunits_pb2.Value] = None) -> tunits_pb2.Value:
        if msg is None:
            msg = tunits_pb2.Value()
        msg.real_value = self.value
        msg.units.extend(_units_to_proto(self.display_units))
        return msg
