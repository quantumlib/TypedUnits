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

from pyparsing import (
    Word,
    Literal,
    Optional,
    alphas,
    nums,
    alphanums,
    stringStart,
    stringEnd,
    Combine,
    Group,
    ZeroOrMore,
    ParserElement,
)


def _maybe_parens(token: ParserElement) -> ParserElement:
    """
    :param ParserElement token:
    :return ParserElement:
    """
    return token ^ ('(' + token + ')')


scalar_combine = Combine(
    Word('+-' + nums, nums)
    + Optional('.' + Optional(Word(nums)))
    + Optional('e' + Word('+-' + nums, nums))
)
scalar = scalar_combine.setParseAction(lambda s, l, t: [float(t[0])])('factor')

number = Word(nums).setParseAction(lambda s, l, t: [int(t[0])])
name = Word(alphas, alphanums)

negatable = Optional(Literal('-'))('neg')
exponent = _maybe_parens(negatable + _maybe_parens(number('num') + Optional('/' + number('denom'))))

single_unit = name('name') + Optional('^' + exponent)
head = Group(single_unit).setResultsName('posexp', True)
times_unit = Group('*' + single_unit).setResultsName('posexp', True)
over_unit = Group('/' + single_unit).setResultsName('negexp', True)

unit_regex = (
    stringStart
    + Optional(scalar)
    + Optional(head)
    + ZeroOrMore(times_unit | Optional('1') + over_unit)
    + stringEnd
)
