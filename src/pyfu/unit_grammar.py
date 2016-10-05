from __future__ import absolute_import
from pyparsing import (Word,
                       Literal,
                       Forward,
                       Optional,
                       alphas,
                       nums,
                       alphanums,
                       stringEnd)


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

number = Word(nums).setParseAction(lambda s, l, t: [int(t[0])])
# degree and mu
name = Word(alphas + '%"\'\xE6\xF8', alphanums + '%"\'\xE6\xF8')

negatable = Optional(out('neg', Literal('-')))
exponent = maybe_parens(negatable + maybe_parens(out('num', number) + Optional('/' + out('denom', number))))

single_unit = out('name', name) + Optional('^' + exponent)
bare_unit = all_out('posexp', single_unit)
times_unit = all_out('posexp', '*' + single_unit)
over_unit = all_out('negexp', '/' + single_unit)

later_units = Forward()
later_units <<= (times_unit | over_unit) + Optional(later_units)

unit = Forward()
unit <<= (bare_unit | '1' + over_unit) + Optional(later_units) + stringEnd
