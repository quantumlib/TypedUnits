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
)


def maybe_parens(token):
    """
    :param ParserElement token:
    :return ParserElement:
    """
    return token ^ ('(' + token + ')')


scalar = Combine(
    Word('+-' + nums, nums)
    + Optional('.' + Optional(Word(nums)))
    + Optional('e' + Word('+-' + nums, nums))
)
scalar = scalar.setParseAction(lambda s, l, t: [float(t[0])])('factor')

number = Word(nums).setParseAction(lambda s, l, t: [int(t[0])])
name = Word(alphas, alphanums)

negatable = Optional(Literal('-'))('neg')
exponent = maybe_parens(negatable + maybe_parens(number('num') + Optional('/' + number('denom'))))

single_unit = name('name') + Optional('^' + exponent)
head = Group(single_unit).setResultsName('posexp', True)
times_unit = Group('*' + single_unit).setResultsName('posexp', True)
over_unit = Group('/' + single_unit).setResultsName('negexp', True)

unit = (
    stringStart
    + Optional(scalar)
    + Optional(head)
    + ZeroOrMore(times_unit | Optional('1') + over_unit)
    + stringEnd
)
