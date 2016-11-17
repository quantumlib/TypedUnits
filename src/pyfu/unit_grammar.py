from __future__ import absolute_import
from pyparsing import (Word,
                       Literal,
                       Forward,
                       Optional,
                       alphas,
                       nums,
                       alphanums,
                       stringStart,
                       stringEnd,
                       Combine)


def out(key, token):
    """
    :param str key:
    :param ParserElement token:
    :return ParserElement:
    """
    return token.setResultsName(key)


def all_out(key, token):
    """
    :param str key:
    :param ParserElement token:
    :return ParserElement:
    """
    return token.setResultsName(key, listAllMatches=True)


def maybe_parens(token):
    """
    :param ParserElement token:
    :return ParserElement:
    """
    return token ^ ('(' + token + ')')

scalar = Combine(Word('+-' + nums, nums) +
                 Optional('.' + Optional(Word(nums))) +
                 Optional('e' + Word('+-' + nums, nums)))
scalar = out('factor', scalar.setParseAction(lambda s, l, t: [float(t[0])]))

number = Word(nums).setParseAction(lambda s, l, t: [int(t[0])])
name = Word(alphas, alphanums)

negatable = Optional(out('neg', Literal('-')))
exponent = maybe_parens(negatable +
                        maybe_parens(out('num', number) +
                                     Optional('/' + out('denom', number))))

single_unit = out('name', name) + Optional('^' + exponent)
head = all_out('posexp', single_unit)
times_unit = all_out('posexp', '*' + single_unit)
over_unit = all_out('negexp', '/' + single_unit)

tail = Forward()
tail <<= (times_unit | over_unit) + Optional(tail)

unit = Forward()
unit <<= (stringStart +
          Optional(scalar) +
          Optional((head | Optional('1') + over_unit) + Optional(tail)) +
          stringEnd)
