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

"""
Exposes standard units as module variables.

For example, '1/units.millisecond' is equal to 'units.kHz'.
"""

import tunits.core as core

# Explicitly expose units.
A = core.default_unit_database.known_units['A']
Ang = core.default_unit_database.known_units['Ang']
Bohr = core.default_unit_database.known_units['Bohr']
Bq = core.default_unit_database.known_units['Bq']
C = core.default_unit_database.known_units['C']
EA = core.default_unit_database.known_units['EA']
EBq = core.default_unit_database.known_units['EBq']
EC = core.default_unit_database.known_units['EC']
EF = core.default_unit_database.known_units['EF']
EGauss = core.default_unit_database.known_units['EGauss']
EH = core.default_unit_database.known_units['EH']
EHz = core.default_unit_database.known_units['EHz']
EJ = core.default_unit_database.known_units['EJ']
EK = core.default_unit_database.known_units['EK']
EN = core.default_unit_database.known_units['EN']
EOhm = core.default_unit_database.known_units['EOhm']
EPa = core.default_unit_database.known_units['EPa']
ES = core.default_unit_database.known_units['ES']
ET = core.default_unit_database.known_units['ET']
EV = core.default_unit_database.known_units['EV']
EW = core.default_unit_database.known_units['EW']
EWb = core.default_unit_database.known_units['EWb']
Ecd = core.default_unit_database.known_units['Ecd']
Eg = core.default_unit_database.known_units['Eg']
El = core.default_unit_database.known_units['El']
Elm = core.default_unit_database.known_units['Elm']
Elx = core.default_unit_database.known_units['Elx']
Em = core.default_unit_database.known_units['Em']
Emol = core.default_unit_database.known_units['Emol']
Erad = core.default_unit_database.known_units['Erad']
Es = core.default_unit_database.known_units['Es']
Esr = core.default_unit_database.known_units['Esr']
F = core.default_unit_database.known_units['F']
G = core.default_unit_database.known_units['G']
GA = core.default_unit_database.known_units['GA']
GBq = core.default_unit_database.known_units['GBq']
GC = core.default_unit_database.known_units['GC']
GF = core.default_unit_database.known_units['GF']
GGauss = core.default_unit_database.known_units['GGauss']
GH = core.default_unit_database.known_units['GH']
GHz = core.default_unit_database.known_units['GHz']
GJ = core.default_unit_database.known_units['GJ']
GK = core.default_unit_database.known_units['GK']
GN = core.default_unit_database.known_units['GN']
GOhm = core.default_unit_database.known_units['GOhm']
GPa = core.default_unit_database.known_units['GPa']
GS = core.default_unit_database.known_units['GS']
GT = core.default_unit_database.known_units['GT']
GV = core.default_unit_database.known_units['GV']
GW = core.default_unit_database.known_units['GW']
GWb = core.default_unit_database.known_units['GWb']
Gauss = core.default_unit_database.known_units['Gauss']
Gcd = core.default_unit_database.known_units['Gcd']
Gg = core.default_unit_database.known_units['Gg']
Gl = core.default_unit_database.known_units['Gl']
Glm = core.default_unit_database.known_units['Glm']
Glx = core.default_unit_database.known_units['Glx']
Gm = core.default_unit_database.known_units['Gm']
Gmol = core.default_unit_database.known_units['Gmol']
Grad = core.default_unit_database.known_units['Grad']
Gs = core.default_unit_database.known_units['Gs']
Gsr = core.default_unit_database.known_units['Gsr']
H = core.default_unit_database.known_units['H']
Hartree = core.default_unit_database.known_units['Hartree']
Hz = core.default_unit_database.known_units['Hz']
J = core.default_unit_database.known_units['J']
K = core.default_unit_database.known_units['K']
MA = core.default_unit_database.known_units['MA']
MBq = core.default_unit_database.known_units['MBq']
MC = core.default_unit_database.known_units['MC']
MF = core.default_unit_database.known_units['MF']
MGauss = core.default_unit_database.known_units['MGauss']
MH = core.default_unit_database.known_units['MH']
MHz = core.default_unit_database.known_units['MHz']
MJ = core.default_unit_database.known_units['MJ']
MK = core.default_unit_database.known_units['MK']
MN = core.default_unit_database.known_units['MN']
MOhm = core.default_unit_database.known_units['MOhm']
MPa = core.default_unit_database.known_units['MPa']
MS = core.default_unit_database.known_units['MS']
MT = core.default_unit_database.known_units['MT']
MV = core.default_unit_database.known_units['MV']
MW = core.default_unit_database.known_units['MW']
MWb = core.default_unit_database.known_units['MWb']
Mcd = core.default_unit_database.known_units['Mcd']
Mg = core.default_unit_database.known_units['Mg']
Ml = core.default_unit_database.known_units['Ml']
Mlm = core.default_unit_database.known_units['Mlm']
Mlx = core.default_unit_database.known_units['Mlx']
Mm = core.default_unit_database.known_units['Mm']
Mmol = core.default_unit_database.known_units['Mmol']
Mrad = core.default_unit_database.known_units['Mrad']
Ms = core.default_unit_database.known_units['Ms']
Msr = core.default_unit_database.known_units['Msr']
N = core.default_unit_database.known_units['N']
Nav = core.default_unit_database.known_units['Nav']
Ohm = core.default_unit_database.known_units['Ohm']
PA = core.default_unit_database.known_units['PA']
PBq = core.default_unit_database.known_units['PBq']
PC = core.default_unit_database.known_units['PC']
PF = core.default_unit_database.known_units['PF']
PGauss = core.default_unit_database.known_units['PGauss']
PH = core.default_unit_database.known_units['PH']
PHz = core.default_unit_database.known_units['PHz']
PJ = core.default_unit_database.known_units['PJ']
PK = core.default_unit_database.known_units['PK']
PN = core.default_unit_database.known_units['PN']
POhm = core.default_unit_database.known_units['POhm']
PPa = core.default_unit_database.known_units['PPa']
PS = core.default_unit_database.known_units['PS']
PT = core.default_unit_database.known_units['PT']
PV = core.default_unit_database.known_units['PV']
PW = core.default_unit_database.known_units['PW']
PWb = core.default_unit_database.known_units['PWb']
Pa = core.default_unit_database.known_units['Pa']
Pcd = core.default_unit_database.known_units['Pcd']
Pg = core.default_unit_database.known_units['Pg']
Pl = core.default_unit_database.known_units['Pl']
Plm = core.default_unit_database.known_units['Plm']
Plx = core.default_unit_database.known_units['Plx']
Pm = core.default_unit_database.known_units['Pm']
Pmol = core.default_unit_database.known_units['Pmol']
Prad = core.default_unit_database.known_units['Prad']
Ps = core.default_unit_database.known_units['Ps']
Psr = core.default_unit_database.known_units['Psr']
R_k = core.default_unit_database.known_units['R_k']
S = core.default_unit_database.known_units['S']
T = core.default_unit_database.known_units['T']
TA = core.default_unit_database.known_units['TA']
TBq = core.default_unit_database.known_units['TBq']
TC = core.default_unit_database.known_units['TC']
TF = core.default_unit_database.known_units['TF']
TGauss = core.default_unit_database.known_units['TGauss']
TH = core.default_unit_database.known_units['TH']
THz = core.default_unit_database.known_units['THz']
TJ = core.default_unit_database.known_units['TJ']
TK = core.default_unit_database.known_units['TK']
TN = core.default_unit_database.known_units['TN']
TOhm = core.default_unit_database.known_units['TOhm']
TPa = core.default_unit_database.known_units['TPa']
TS = core.default_unit_database.known_units['TS']
TT = core.default_unit_database.known_units['TT']
TV = core.default_unit_database.known_units['TV']
TW = core.default_unit_database.known_units['TW']
TWb = core.default_unit_database.known_units['TWb']
Tcd = core.default_unit_database.known_units['Tcd']
Tg = core.default_unit_database.known_units['Tg']
Tl = core.default_unit_database.known_units['Tl']
Tlm = core.default_unit_database.known_units['Tlm']
Tlx = core.default_unit_database.known_units['Tlx']
Tm = core.default_unit_database.known_units['Tm']
Tmol = core.default_unit_database.known_units['Tmol']
Trad = core.default_unit_database.known_units['Trad']
Ts = core.default_unit_database.known_units['Ts']
Tsr = core.default_unit_database.known_units['Tsr']
V = core.default_unit_database.known_units['V']
W = core.default_unit_database.known_units['W']
Wb = core.default_unit_database.known_units['Wb']
YA = core.default_unit_database.known_units['YA']
YBq = core.default_unit_database.known_units['YBq']
YC = core.default_unit_database.known_units['YC']
YF = core.default_unit_database.known_units['YF']
YGauss = core.default_unit_database.known_units['YGauss']
YH = core.default_unit_database.known_units['YH']
YHz = core.default_unit_database.known_units['YHz']
YJ = core.default_unit_database.known_units['YJ']
YK = core.default_unit_database.known_units['YK']
YN = core.default_unit_database.known_units['YN']
YOhm = core.default_unit_database.known_units['YOhm']
YPa = core.default_unit_database.known_units['YPa']
YS = core.default_unit_database.known_units['YS']
YT = core.default_unit_database.known_units['YT']
YV = core.default_unit_database.known_units['YV']
YW = core.default_unit_database.known_units['YW']
YWb = core.default_unit_database.known_units['YWb']
Ycd = core.default_unit_database.known_units['Ycd']
Yg = core.default_unit_database.known_units['Yg']
Yl = core.default_unit_database.known_units['Yl']
Ylm = core.default_unit_database.known_units['Ylm']
Ylx = core.default_unit_database.known_units['Ylx']
Ym = core.default_unit_database.known_units['Ym']
Ymol = core.default_unit_database.known_units['Ymol']
Yrad = core.default_unit_database.known_units['Yrad']
Ys = core.default_unit_database.known_units['Ys']
Ysr = core.default_unit_database.known_units['Ysr']
ZA = core.default_unit_database.known_units['ZA']
ZBq = core.default_unit_database.known_units['ZBq']
ZC = core.default_unit_database.known_units['ZC']
ZF = core.default_unit_database.known_units['ZF']
ZGauss = core.default_unit_database.known_units['ZGauss']
ZH = core.default_unit_database.known_units['ZH']
ZHz = core.default_unit_database.known_units['ZHz']
ZJ = core.default_unit_database.known_units['ZJ']
ZK = core.default_unit_database.known_units['ZK']
ZN = core.default_unit_database.known_units['ZN']
ZOhm = core.default_unit_database.known_units['ZOhm']
ZPa = core.default_unit_database.known_units['ZPa']
ZS = core.default_unit_database.known_units['ZS']
ZT = core.default_unit_database.known_units['ZT']
ZV = core.default_unit_database.known_units['ZV']
ZW = core.default_unit_database.known_units['ZW']
ZWb = core.default_unit_database.known_units['ZWb']
Zcd = core.default_unit_database.known_units['Zcd']
Zg = core.default_unit_database.known_units['Zg']
Zl = core.default_unit_database.known_units['Zl']
Zlm = core.default_unit_database.known_units['Zlm']
Zlx = core.default_unit_database.known_units['Zlx']
Zm = core.default_unit_database.known_units['Zm']
Zmol = core.default_unit_database.known_units['Zmol']
Zrad = core.default_unit_database.known_units['Zrad']
Zs = core.default_unit_database.known_units['Zs']
Zsr = core.default_unit_database.known_units['Zsr']
aA = core.default_unit_database.known_units['aA']
aBq = core.default_unit_database.known_units['aBq']
aC = core.default_unit_database.known_units['aC']
aF = core.default_unit_database.known_units['aF']
aGauss = core.default_unit_database.known_units['aGauss']
aH = core.default_unit_database.known_units['aH']
aHz = core.default_unit_database.known_units['aHz']
aJ = core.default_unit_database.known_units['aJ']
aK = core.default_unit_database.known_units['aK']
aN = core.default_unit_database.known_units['aN']
aOhm = core.default_unit_database.known_units['aOhm']
aPa = core.default_unit_database.known_units['aPa']
aS = core.default_unit_database.known_units['aS']
aT = core.default_unit_database.known_units['aT']
aV = core.default_unit_database.known_units['aV']
aW = core.default_unit_database.known_units['aW']
aWb = core.default_unit_database.known_units['aWb']
acd = core.default_unit_database.known_units['acd']
ag = core.default_unit_database.known_units['ag']
al = core.default_unit_database.known_units['al']
alm = core.default_unit_database.known_units['alm']
alx = core.default_unit_database.known_units['alx']
am = core.default_unit_database.known_units['am']
amol = core.default_unit_database.known_units['amol']
ampere = core.default_unit_database.known_units['ampere']
amu = core.default_unit_database.known_units['amu']
angstrom = core.default_unit_database.known_units['angstrom']
arad = core.default_unit_database.known_units['arad']
asr = core.default_unit_database.known_units['asr']
atomic_mass_unit = core.default_unit_database.known_units['atomic_mass_unit']
attoampere = core.default_unit_database.known_units['attoampere']
attobecquerel = core.default_unit_database.known_units['attobecquerel']
attocandela = core.default_unit_database.known_units['attocandela']
attocoulomb = core.default_unit_database.known_units['attocoulomb']
attofarad = core.default_unit_database.known_units['attofarad']
attogauss = core.default_unit_database.known_units['attogauss']
attogram = core.default_unit_database.known_units['attogram']
attohenry = core.default_unit_database.known_units['attohenry']
attohertz = core.default_unit_database.known_units['attohertz']
attojoule = core.default_unit_database.known_units['attojoule']
attokelvin = core.default_unit_database.known_units['attokelvin']
attoliter = core.default_unit_database.known_units['attoliter']
attolumen = core.default_unit_database.known_units['attolumen']
attolux = core.default_unit_database.known_units['attolux']
attometer = core.default_unit_database.known_units['attometer']
attomole = core.default_unit_database.known_units['attomole']
attonewton = core.default_unit_database.known_units['attonewton']
attoohm = core.default_unit_database.known_units['attoohm']
attopascal = core.default_unit_database.known_units['attopascal']
attoradian = core.default_unit_database.known_units['attoradian']
attosecond = core.default_unit_database.known_units['attosecond']
attosiemens = core.default_unit_database.known_units['attosiemens']
attosteradian = core.default_unit_database.known_units['attosteradian']
attotesla = core.default_unit_database.known_units['attotesla']
attovolt = core.default_unit_database.known_units['attovolt']
attowatt = core.default_unit_database.known_units['attowatt']
attoweber = core.default_unit_database.known_units['attoweber']
avogadro_constant = core.default_unit_database.known_units['avogadro_constant']
b = core.default_unit_database.known_units['b']
bar = core.default_unit_database.known_units['bar']
barn = core.default_unit_database.known_units['barn']
becquerel = core.default_unit_database.known_units['becquerel']
bohr_magneton = core.default_unit_database.known_units['bohr_magneton']
bohr_radius = core.default_unit_database.known_units['bohr_radius']
boltzmann_constant = core.default_unit_database.known_units['boltzmann_constant']
british_gallon = core.default_unit_database.known_units['british_gallon']
c = core.default_unit_database.known_units['c']
cA = core.default_unit_database.known_units['cA']
cBq = core.default_unit_database.known_units['cBq']
cC = core.default_unit_database.known_units['cC']
cF = core.default_unit_database.known_units['cF']
cGauss = core.default_unit_database.known_units['cGauss']
cH = core.default_unit_database.known_units['cH']
cHz = core.default_unit_database.known_units['cHz']
cJ = core.default_unit_database.known_units['cJ']
cK = core.default_unit_database.known_units['cK']
cN = core.default_unit_database.known_units['cN']
cOhm = core.default_unit_database.known_units['cOhm']
cPa = core.default_unit_database.known_units['cPa']
cS = core.default_unit_database.known_units['cS']
cT = core.default_unit_database.known_units['cT']
cV = core.default_unit_database.known_units['cV']
cW = core.default_unit_database.known_units['cW']
cWb = core.default_unit_database.known_units['cWb']
candela = core.default_unit_database.known_units['candela']
ccd = core.default_unit_database.known_units['ccd']
cd = core.default_unit_database.known_units['cd']
centiampere = core.default_unit_database.known_units['centiampere']
centibecquerel = core.default_unit_database.known_units['centibecquerel']
centicandela = core.default_unit_database.known_units['centicandela']
centicoulomb = core.default_unit_database.known_units['centicoulomb']
centifarad = core.default_unit_database.known_units['centifarad']
centigauss = core.default_unit_database.known_units['centigauss']
centigram = core.default_unit_database.known_units['centigram']
centihenry = core.default_unit_database.known_units['centihenry']
centihertz = core.default_unit_database.known_units['centihertz']
centijoule = core.default_unit_database.known_units['centijoule']
centikelvin = core.default_unit_database.known_units['centikelvin']
centiliter = core.default_unit_database.known_units['centiliter']
centilumen = core.default_unit_database.known_units['centilumen']
centilux = core.default_unit_database.known_units['centilux']
centimeter = core.default_unit_database.known_units['centimeter']
centimole = core.default_unit_database.known_units['centimole']
centinewton = core.default_unit_database.known_units['centinewton']
centiohm = core.default_unit_database.known_units['centiohm']
centipascal = core.default_unit_database.known_units['centipascal']
centiradian = core.default_unit_database.known_units['centiradian']
centisecond = core.default_unit_database.known_units['centisecond']
centisiemens = core.default_unit_database.known_units['centisiemens']
centisteradian = core.default_unit_database.known_units['centisteradian']
centitesla = core.default_unit_database.known_units['centitesla']
centivolt = core.default_unit_database.known_units['centivolt']
centiwatt = core.default_unit_database.known_units['centiwatt']
centiweber = core.default_unit_database.known_units['centiweber']
cg = core.default_unit_database.known_units['cg']
cl = core.default_unit_database.known_units['cl']
clm = core.default_unit_database.known_units['clm']
clx = core.default_unit_database.known_units['clx']
cm = core.default_unit_database.known_units['cm']
cmol = core.default_unit_database.known_units['cmol']
coulomb = core.default_unit_database.known_units['coulomb']
crad = core.default_unit_database.known_units['crad']
cs = core.default_unit_database.known_units['cs']
csr = core.default_unit_database.known_units['csr']
cup = core.default_unit_database.known_units['cup']
cyc = core.default_unit_database.known_units['cyc']
cycle = core.default_unit_database.known_units['cycle']
d = core.default_unit_database.known_units['d']
dA = core.default_unit_database.known_units['dA']
dB = core.default_unit_database.known_units['dB']
dBm = core.default_unit_database.known_units['dBm']
dBq = core.default_unit_database.known_units['dBq']
dC = core.default_unit_database.known_units['dC']
dF = core.default_unit_database.known_units['dF']
dGauss = core.default_unit_database.known_units['dGauss']
dH = core.default_unit_database.known_units['dH']
dHz = core.default_unit_database.known_units['dHz']
dJ = core.default_unit_database.known_units['dJ']
dK = core.default_unit_database.known_units['dK']
dN = core.default_unit_database.known_units['dN']
dOhm = core.default_unit_database.known_units['dOhm']
dPa = core.default_unit_database.known_units['dPa']
dS = core.default_unit_database.known_units['dS']
dT = core.default_unit_database.known_units['dT']
dV = core.default_unit_database.known_units['dV']
dW = core.default_unit_database.known_units['dW']
dWb = core.default_unit_database.known_units['dWb']
daA = core.default_unit_database.known_units['daA']
daBq = core.default_unit_database.known_units['daBq']
daC = core.default_unit_database.known_units['daC']
daF = core.default_unit_database.known_units['daF']
daGauss = core.default_unit_database.known_units['daGauss']
daH = core.default_unit_database.known_units['daH']
daHz = core.default_unit_database.known_units['daHz']
daJ = core.default_unit_database.known_units['daJ']
daK = core.default_unit_database.known_units['daK']
daN = core.default_unit_database.known_units['daN']
daOhm = core.default_unit_database.known_units['daOhm']
daPa = core.default_unit_database.known_units['daPa']
daS = core.default_unit_database.known_units['daS']
daT = core.default_unit_database.known_units['daT']
daV = core.default_unit_database.known_units['daV']
daW = core.default_unit_database.known_units['daW']
daWb = core.default_unit_database.known_units['daWb']
dacd = core.default_unit_database.known_units['dacd']
dag = core.default_unit_database.known_units['dag']
dal = core.default_unit_database.known_units['dal']
dalm = core.default_unit_database.known_units['dalm']
dalx = core.default_unit_database.known_units['dalx']
dam = core.default_unit_database.known_units['dam']
damol = core.default_unit_database.known_units['damol']
darad = core.default_unit_database.known_units['darad']
das = core.default_unit_database.known_units['das']
dasr = core.default_unit_database.known_units['dasr']
day = core.default_unit_database.known_units['day']
dcd = core.default_unit_database.known_units['dcd']
deciampere = core.default_unit_database.known_units['deciampere']
decibecquerel = core.default_unit_database.known_units['decibecquerel']
decibel = core.default_unit_database.known_units['decibel']
decicandela = core.default_unit_database.known_units['decicandela']
decicoulomb = core.default_unit_database.known_units['decicoulomb']
decifarad = core.default_unit_database.known_units['decifarad']
decigauss = core.default_unit_database.known_units['decigauss']
decigram = core.default_unit_database.known_units['decigram']
decihenry = core.default_unit_database.known_units['decihenry']
decihertz = core.default_unit_database.known_units['decihertz']
decijoule = core.default_unit_database.known_units['decijoule']
decikelvin = core.default_unit_database.known_units['decikelvin']
deciliter = core.default_unit_database.known_units['deciliter']
decilumen = core.default_unit_database.known_units['decilumen']
decilux = core.default_unit_database.known_units['decilux']
decimeter = core.default_unit_database.known_units['decimeter']
decimole = core.default_unit_database.known_units['decimole']
decinewton = core.default_unit_database.known_units['decinewton']
deciohm = core.default_unit_database.known_units['deciohm']
decipascal = core.default_unit_database.known_units['decipascal']
deciradian = core.default_unit_database.known_units['deciradian']
decisecond = core.default_unit_database.known_units['decisecond']
decisiemens = core.default_unit_database.known_units['decisiemens']
decisteradian = core.default_unit_database.known_units['decisteradian']
decitesla = core.default_unit_database.known_units['decitesla']
decivolt = core.default_unit_database.known_units['decivolt']
deciwatt = core.default_unit_database.known_units['deciwatt']
deciweber = core.default_unit_database.known_units['deciweber']
deg = core.default_unit_database.known_units['deg']
degC = core.default_unit_database.known_units['degC']
degF = core.default_unit_database.known_units['degF']
degR = core.default_unit_database.known_units['degR']
dekaampere = core.default_unit_database.known_units['dekaampere']
dekabecquerel = core.default_unit_database.known_units['dekabecquerel']
dekacandela = core.default_unit_database.known_units['dekacandela']
dekacoulomb = core.default_unit_database.known_units['dekacoulomb']
dekafarad = core.default_unit_database.known_units['dekafarad']
dekagauss = core.default_unit_database.known_units['dekagauss']
dekagram = core.default_unit_database.known_units['dekagram']
dekahenry = core.default_unit_database.known_units['dekahenry']
dekahertz = core.default_unit_database.known_units['dekahertz']
dekajoule = core.default_unit_database.known_units['dekajoule']
dekakelvin = core.default_unit_database.known_units['dekakelvin']
dekaliter = core.default_unit_database.known_units['dekaliter']
dekalumen = core.default_unit_database.known_units['dekalumen']
dekalux = core.default_unit_database.known_units['dekalux']
dekameter = core.default_unit_database.known_units['dekameter']
dekamole = core.default_unit_database.known_units['dekamole']
dekanewton = core.default_unit_database.known_units['dekanewton']
dekaohm = core.default_unit_database.known_units['dekaohm']
dekapascal = core.default_unit_database.known_units['dekapascal']
dekaradian = core.default_unit_database.known_units['dekaradian']
dekasecond = core.default_unit_database.known_units['dekasecond']
dekasiemens = core.default_unit_database.known_units['dekasiemens']
dekasteradian = core.default_unit_database.known_units['dekasteradian']
dekatesla = core.default_unit_database.known_units['dekatesla']
dekavolt = core.default_unit_database.known_units['dekavolt']
dekawatt = core.default_unit_database.known_units['dekawatt']
dekaweber = core.default_unit_database.known_units['dekaweber']
dg = core.default_unit_database.known_units['dg']
dl = core.default_unit_database.known_units['dl']
dlm = core.default_unit_database.known_units['dlm']
dlx = core.default_unit_database.known_units['dlx']
dm = core.default_unit_database.known_units['dm']
dmol = core.default_unit_database.known_units['dmol']
drad = core.default_unit_database.known_units['drad']
ds = core.default_unit_database.known_units['ds']
dsr = core.default_unit_database.known_units['dsr']
e = core.default_unit_database.known_units['e']
electron_mass = core.default_unit_database.known_units['electron_mass']
elementary_charge = core.default_unit_database.known_units['elementary_charge']
eV = core.default_unit_database.known_units['eV']
eps0 = core.default_unit_database.known_units['eps0']
exaampere = core.default_unit_database.known_units['exaampere']
exabecquerel = core.default_unit_database.known_units['exabecquerel']
exacandela = core.default_unit_database.known_units['exacandela']
exacoulomb = core.default_unit_database.known_units['exacoulomb']
exafarad = core.default_unit_database.known_units['exafarad']
exagauss = core.default_unit_database.known_units['exagauss']
exagram = core.default_unit_database.known_units['exagram']
exahenry = core.default_unit_database.known_units['exahenry']
exahertz = core.default_unit_database.known_units['exahertz']
exajoule = core.default_unit_database.known_units['exajoule']
exakelvin = core.default_unit_database.known_units['exakelvin']
exaliter = core.default_unit_database.known_units['exaliter']
exalumen = core.default_unit_database.known_units['exalumen']
exalux = core.default_unit_database.known_units['exalux']
exameter = core.default_unit_database.known_units['exameter']
examole = core.default_unit_database.known_units['examole']
exanewton = core.default_unit_database.known_units['exanewton']
exaohm = core.default_unit_database.known_units['exaohm']
exapascal = core.default_unit_database.known_units['exapascal']
exaradian = core.default_unit_database.known_units['exaradian']
exasecond = core.default_unit_database.known_units['exasecond']
exasiemens = core.default_unit_database.known_units['exasiemens']
exasteradian = core.default_unit_database.known_units['exasteradian']
exatesla = core.default_unit_database.known_units['exatesla']
exavolt = core.default_unit_database.known_units['exavolt']
exawatt = core.default_unit_database.known_units['exawatt']
exaweber = core.default_unit_database.known_units['exaweber']
fA = core.default_unit_database.known_units['fA']
fBq = core.default_unit_database.known_units['fBq']
fC = core.default_unit_database.known_units['fC']
fF = core.default_unit_database.known_units['fF']
fGauss = core.default_unit_database.known_units['fGauss']
fH = core.default_unit_database.known_units['fH']
fHz = core.default_unit_database.known_units['fHz']
fJ = core.default_unit_database.known_units['fJ']
fK = core.default_unit_database.known_units['fK']
fN = core.default_unit_database.known_units['fN']
fOhm = core.default_unit_database.known_units['fOhm']
fPa = core.default_unit_database.known_units['fPa']
fS = core.default_unit_database.known_units['fS']
fT = core.default_unit_database.known_units['fT']
fV = core.default_unit_database.known_units['fV']
fW = core.default_unit_database.known_units['fW']
fWb = core.default_unit_database.known_units['fWb']
farad = core.default_unit_database.known_units['farad']
fcd = core.default_unit_database.known_units['fcd']
femtoampere = core.default_unit_database.known_units['femtoampere']
femtobecquerel = core.default_unit_database.known_units['femtobecquerel']
femtocandela = core.default_unit_database.known_units['femtocandela']
femtocoulomb = core.default_unit_database.known_units['femtocoulomb']
femtofarad = core.default_unit_database.known_units['femtofarad']
femtogauss = core.default_unit_database.known_units['femtogauss']
femtogram = core.default_unit_database.known_units['femtogram']
femtohenry = core.default_unit_database.known_units['femtohenry']
femtohertz = core.default_unit_database.known_units['femtohertz']
femtojoule = core.default_unit_database.known_units['femtojoule']
femtokelvin = core.default_unit_database.known_units['femtokelvin']
femtoliter = core.default_unit_database.known_units['femtoliter']
femtolumen = core.default_unit_database.known_units['femtolumen']
femtolux = core.default_unit_database.known_units['femtolux']
femtometer = core.default_unit_database.known_units['femtometer']
femtomole = core.default_unit_database.known_units['femtomole']
femtonewton = core.default_unit_database.known_units['femtonewton']
femtoohm = core.default_unit_database.known_units['femtoohm']
femtopascal = core.default_unit_database.known_units['femtopascal']
femtoradian = core.default_unit_database.known_units['femtoradian']
femtosecond = core.default_unit_database.known_units['femtosecond']
femtosiemens = core.default_unit_database.known_units['femtosiemens']
femtosteradian = core.default_unit_database.known_units['femtosteradian']
femtotesla = core.default_unit_database.known_units['femtotesla']
femtovolt = core.default_unit_database.known_units['femtovolt']
femtowatt = core.default_unit_database.known_units['femtowatt']
femtoweber = core.default_unit_database.known_units['femtoweber']
fg = core.default_unit_database.known_units['fg']
fl = core.default_unit_database.known_units['fl']
flm = core.default_unit_database.known_units['flm']
floz = core.default_unit_database.known_units['floz']
fluid_ounce = core.default_unit_database.known_units['fluid_ounce']
flx = core.default_unit_database.known_units['flx']
fm = core.default_unit_database.known_units['fm']
fmol = core.default_unit_database.known_units['fmol']
foot = core.default_unit_database.known_units['foot']
frad = core.default_unit_database.known_units['frad']
fs = core.default_unit_database.known_units['fs']
fsr = core.default_unit_database.known_units['fsr']
ft = core.default_unit_database.known_units['ft']
g = core.default_unit_database.known_units['g']
galUK = core.default_unit_database.known_units['galUK']
galUS = core.default_unit_database.known_units['galUS']
gauss = core.default_unit_database.known_units['gauss']
gigaampere = core.default_unit_database.known_units['gigaampere']
gigabecquerel = core.default_unit_database.known_units['gigabecquerel']
gigacandela = core.default_unit_database.known_units['gigacandela']
gigacoulomb = core.default_unit_database.known_units['gigacoulomb']
gigafarad = core.default_unit_database.known_units['gigafarad']
gigagauss = core.default_unit_database.known_units['gigagauss']
gigagram = core.default_unit_database.known_units['gigagram']
gigahenry = core.default_unit_database.known_units['gigahenry']
gigahertz = core.default_unit_database.known_units['gigahertz']
gigajoule = core.default_unit_database.known_units['gigajoule']
gigakelvin = core.default_unit_database.known_units['gigakelvin']
gigaliter = core.default_unit_database.known_units['gigaliter']
gigalumen = core.default_unit_database.known_units['gigalumen']
gigalux = core.default_unit_database.known_units['gigalux']
gigameter = core.default_unit_database.known_units['gigameter']
gigamole = core.default_unit_database.known_units['gigamole']
giganewton = core.default_unit_database.known_units['giganewton']
gigaohm = core.default_unit_database.known_units['gigaohm']
gigapascal = core.default_unit_database.known_units['gigapascal']
gigaradian = core.default_unit_database.known_units['gigaradian']
gigasecond = core.default_unit_database.known_units['gigasecond']
gigasiemens = core.default_unit_database.known_units['gigasiemens']
gigasteradian = core.default_unit_database.known_units['gigasteradian']
gigatesla = core.default_unit_database.known_units['gigatesla']
gigavolt = core.default_unit_database.known_units['gigavolt']
gigawatt = core.default_unit_database.known_units['gigawatt']
gigaweber = core.default_unit_database.known_units['gigaweber']
gram = core.default_unit_database.known_units['gram']
gravitational_constant = core.default_unit_database.known_units['gravitational_constant']
h = core.default_unit_database.known_units['h']
hA = core.default_unit_database.known_units['hA']
hBq = core.default_unit_database.known_units['hBq']
hC = core.default_unit_database.known_units['hC']
hF = core.default_unit_database.known_units['hF']
hGauss = core.default_unit_database.known_units['hGauss']
hH = core.default_unit_database.known_units['hH']
hHz = core.default_unit_database.known_units['hHz']
hJ = core.default_unit_database.known_units['hJ']
hK = core.default_unit_database.known_units['hK']
hN = core.default_unit_database.known_units['hN']
hOhm = core.default_unit_database.known_units['hOhm']
hPa = core.default_unit_database.known_units['hPa']
hS = core.default_unit_database.known_units['hS']
hT = core.default_unit_database.known_units['hT']
hV = core.default_unit_database.known_units['hV']
hW = core.default_unit_database.known_units['hW']
hWb = core.default_unit_database.known_units['hWb']
ha = core.default_unit_database.known_units['ha']
hbar = core.default_unit_database.known_units['hbar']
hcd = core.default_unit_database.known_units['hcd']
hectare = core.default_unit_database.known_units['hectare']
hectoampere = core.default_unit_database.known_units['hectoampere']
hectobecquerel = core.default_unit_database.known_units['hectobecquerel']
hectocandela = core.default_unit_database.known_units['hectocandela']
hectocoulomb = core.default_unit_database.known_units['hectocoulomb']
hectofarad = core.default_unit_database.known_units['hectofarad']
hectogauss = core.default_unit_database.known_units['hectogauss']
hectogram = core.default_unit_database.known_units['hectogram']
hectohenry = core.default_unit_database.known_units['hectohenry']
hectohertz = core.default_unit_database.known_units['hectohertz']
hectojoule = core.default_unit_database.known_units['hectojoule']
hectokelvin = core.default_unit_database.known_units['hectokelvin']
hectoliter = core.default_unit_database.known_units['hectoliter']
hectolumen = core.default_unit_database.known_units['hectolumen']
hectolux = core.default_unit_database.known_units['hectolux']
hectometer = core.default_unit_database.known_units['hectometer']
hectomole = core.default_unit_database.known_units['hectomole']
hectonewton = core.default_unit_database.known_units['hectonewton']
hectoohm = core.default_unit_database.known_units['hectoohm']
hectopascal = core.default_unit_database.known_units['hectopascal']
hectoradian = core.default_unit_database.known_units['hectoradian']
hectosecond = core.default_unit_database.known_units['hectosecond']
hectosiemens = core.default_unit_database.known_units['hectosiemens']
hectosteradian = core.default_unit_database.known_units['hectosteradian']
hectotesla = core.default_unit_database.known_units['hectotesla']
hectovolt = core.default_unit_database.known_units['hectovolt']
hectowatt = core.default_unit_database.known_units['hectowatt']
hectoweber = core.default_unit_database.known_units['hectoweber']
henry = core.default_unit_database.known_units['henry']
hertz = core.default_unit_database.known_units['hertz']
hg = core.default_unit_database.known_units['hg']
hl = core.default_unit_database.known_units['hl']
hlm = core.default_unit_database.known_units['hlm']
hlx = core.default_unit_database.known_units['hlx']
hm = core.default_unit_database.known_units['hm']
hmol = core.default_unit_database.known_units['hmol']
hour = core.default_unit_database.known_units['hour']
hplanck = core.default_unit_database.known_units['hplanck']
hrad = core.default_unit_database.known_units['hrad']
hs = core.default_unit_database.known_units['hs']
hsr = core.default_unit_database.known_units['hsr']
inch = core.default_unit_database.known_units['inch']
joule = core.default_unit_database.known_units['joule']
k = core.default_unit_database.known_units['k']
kA = core.default_unit_database.known_units['kA']
kBq = core.default_unit_database.known_units['kBq']
kC = core.default_unit_database.known_units['kC']
kF = core.default_unit_database.known_units['kF']
kGauss = core.default_unit_database.known_units['kGauss']
kH = core.default_unit_database.known_units['kH']
kHz = core.default_unit_database.known_units['kHz']
kJ = core.default_unit_database.known_units['kJ']
kK = core.default_unit_database.known_units['kK']
kN = core.default_unit_database.known_units['kN']
kOhm = core.default_unit_database.known_units['kOhm']
kPa = core.default_unit_database.known_units['kPa']
kS = core.default_unit_database.known_units['kS']
kT = core.default_unit_database.known_units['kT']
kV = core.default_unit_database.known_units['kV']
kW = core.default_unit_database.known_units['kW']
kWb = core.default_unit_database.known_units['kWb']
kcd = core.default_unit_database.known_units['kcd']
kelvin = core.default_unit_database.known_units['kelvin']
kg = core.default_unit_database.known_units['kg']
kiloampere = core.default_unit_database.known_units['kiloampere']
kilobecquerel = core.default_unit_database.known_units['kilobecquerel']
kilocandela = core.default_unit_database.known_units['kilocandela']
kilocoulomb = core.default_unit_database.known_units['kilocoulomb']
kilofarad = core.default_unit_database.known_units['kilofarad']
kilogauss = core.default_unit_database.known_units['kilogauss']
kilogram = core.default_unit_database.known_units['kilogram']
kilohenry = core.default_unit_database.known_units['kilohenry']
kilohertz = core.default_unit_database.known_units['kilohertz']
kilojoule = core.default_unit_database.known_units['kilojoule']
kilokelvin = core.default_unit_database.known_units['kilokelvin']
kiloliter = core.default_unit_database.known_units['kiloliter']
kilolumen = core.default_unit_database.known_units['kilolumen']
kilolux = core.default_unit_database.known_units['kilolux']
kilometer = core.default_unit_database.known_units['kilometer']
kilomole = core.default_unit_database.known_units['kilomole']
kilonewton = core.default_unit_database.known_units['kilonewton']
kiloohm = core.default_unit_database.known_units['kiloohm']
kilopascal = core.default_unit_database.known_units['kilopascal']
kiloradian = core.default_unit_database.known_units['kiloradian']
kilosecond = core.default_unit_database.known_units['kilosecond']
kilosiemens = core.default_unit_database.known_units['kilosiemens']
kilosteradian = core.default_unit_database.known_units['kilosteradian']
kilotesla = core.default_unit_database.known_units['kilotesla']
kilovolt = core.default_unit_database.known_units['kilovolt']
kilowatt = core.default_unit_database.known_units['kilowatt']
kiloweber = core.default_unit_database.known_units['kiloweber']
kl = core.default_unit_database.known_units['kl']
klm = core.default_unit_database.known_units['klm']
klx = core.default_unit_database.known_units['klx']
km = core.default_unit_database.known_units['km']
kmol = core.default_unit_database.known_units['kmol']
krad = core.default_unit_database.known_units['krad']
ks = core.default_unit_database.known_units['ks']
ksr = core.default_unit_database.known_units['ksr']
l = core.default_unit_database.known_units['l']
lb = core.default_unit_database.known_units['lb']
light_year = core.default_unit_database.known_units['light_year']
liter = core.default_unit_database.known_units['liter']
lm = core.default_unit_database.known_units['lm']
lumen = core.default_unit_database.known_units['lumen']
lux = core.default_unit_database.known_units['lux']
lx = core.default_unit_database.known_units['lx']
ly = core.default_unit_database.known_units['ly']
lyr = core.default_unit_database.known_units['lyr']
m = core.default_unit_database.known_units['m']
mA = core.default_unit_database.known_units['mA']
mBq = core.default_unit_database.known_units['mBq']
mC = core.default_unit_database.known_units['mC']
mF = core.default_unit_database.known_units['mF']
mGauss = core.default_unit_database.known_units['mGauss']
mH = core.default_unit_database.known_units['mH']
mHz = core.default_unit_database.known_units['mHz']
mJ = core.default_unit_database.known_units['mJ']
mK = core.default_unit_database.known_units['mK']
mN = core.default_unit_database.known_units['mN']
mOhm = core.default_unit_database.known_units['mOhm']
mPa = core.default_unit_database.known_units['mPa']
mS = core.default_unit_database.known_units['mS']
mT = core.default_unit_database.known_units['mT']
mV = core.default_unit_database.known_units['mV']
mW = core.default_unit_database.known_units['mW']
mWb = core.default_unit_database.known_units['mWb']
mcd = core.default_unit_database.known_units['mcd']
me = core.default_unit_database.known_units['me']
megaampere = core.default_unit_database.known_units['megaampere']
megabecquerel = core.default_unit_database.known_units['megabecquerel']
megacandela = core.default_unit_database.known_units['megacandela']
megacoulomb = core.default_unit_database.known_units['megacoulomb']
megafarad = core.default_unit_database.known_units['megafarad']
megagauss = core.default_unit_database.known_units['megagauss']
megagram = core.default_unit_database.known_units['megagram']
megahenry = core.default_unit_database.known_units['megahenry']
megahertz = core.default_unit_database.known_units['megahertz']
megajoule = core.default_unit_database.known_units['megajoule']
megakelvin = core.default_unit_database.known_units['megakelvin']
megaliter = core.default_unit_database.known_units['megaliter']
megalumen = core.default_unit_database.known_units['megalumen']
megalux = core.default_unit_database.known_units['megalux']
megameter = core.default_unit_database.known_units['megameter']
megamole = core.default_unit_database.known_units['megamole']
meganewton = core.default_unit_database.known_units['meganewton']
megaohm = core.default_unit_database.known_units['megaohm']
megapascal = core.default_unit_database.known_units['megapascal']
megaradian = core.default_unit_database.known_units['megaradian']
megasecond = core.default_unit_database.known_units['megasecond']
megasiemens = core.default_unit_database.known_units['megasiemens']
megasteradian = core.default_unit_database.known_units['megasteradian']
megatesla = core.default_unit_database.known_units['megatesla']
megavolt = core.default_unit_database.known_units['megavolt']
megawatt = core.default_unit_database.known_units['megawatt']
megaweber = core.default_unit_database.known_units['megaweber']
meter = core.default_unit_database.known_units['meter']
mg = core.default_unit_database.known_units['mg']
microampere = core.default_unit_database.known_units['microampere']
microbecquerel = core.default_unit_database.known_units['microbecquerel']
microcandela = core.default_unit_database.known_units['microcandela']
microcoulomb = core.default_unit_database.known_units['microcoulomb']
microfarad = core.default_unit_database.known_units['microfarad']
microgauss = core.default_unit_database.known_units['microgauss']
microgram = core.default_unit_database.known_units['microgram']
microhenry = core.default_unit_database.known_units['microhenry']
microhertz = core.default_unit_database.known_units['microhertz']
microjoule = core.default_unit_database.known_units['microjoule']
microkelvin = core.default_unit_database.known_units['microkelvin']
microliter = core.default_unit_database.known_units['microliter']
microlumen = core.default_unit_database.known_units['microlumen']
microlux = core.default_unit_database.known_units['microlux']
micrometer = core.default_unit_database.known_units['micrometer']
micromole = core.default_unit_database.known_units['micromole']
micronewton = core.default_unit_database.known_units['micronewton']
microohm = core.default_unit_database.known_units['microohm']
micropascal = core.default_unit_database.known_units['micropascal']
microradian = core.default_unit_database.known_units['microradian']
microsecond = core.default_unit_database.known_units['microsecond']
microsiemens = core.default_unit_database.known_units['microsiemens']
microsteradian = core.default_unit_database.known_units['microsteradian']
microtesla = core.default_unit_database.known_units['microtesla']
microvolt = core.default_unit_database.known_units['microvolt']
microwatt = core.default_unit_database.known_units['microwatt']
microweber = core.default_unit_database.known_units['microweber']
milliampere = core.default_unit_database.known_units['milliampere']
millibecquerel = core.default_unit_database.known_units['millibecquerel']
millicandela = core.default_unit_database.known_units['millicandela']
millicoulomb = core.default_unit_database.known_units['millicoulomb']
millifarad = core.default_unit_database.known_units['millifarad']
milligauss = core.default_unit_database.known_units['milligauss']
milligram = core.default_unit_database.known_units['milligram']
millihenry = core.default_unit_database.known_units['millihenry']
millihertz = core.default_unit_database.known_units['millihertz']
millijoule = core.default_unit_database.known_units['millijoule']
millikelvin = core.default_unit_database.known_units['millikelvin']
milliliter = core.default_unit_database.known_units['milliliter']
millilumen = core.default_unit_database.known_units['millilumen']
millilux = core.default_unit_database.known_units['millilux']
millimeter = core.default_unit_database.known_units['millimeter']
millimole = core.default_unit_database.known_units['millimole']
millinewton = core.default_unit_database.known_units['millinewton']
milliohm = core.default_unit_database.known_units['milliohm']
millipascal = core.default_unit_database.known_units['millipascal']
milliradian = core.default_unit_database.known_units['milliradian']
millisecond = core.default_unit_database.known_units['millisecond']
millisiemens = core.default_unit_database.known_units['millisiemens']
millisteradian = core.default_unit_database.known_units['millisteradian']
millitesla = core.default_unit_database.known_units['millitesla']
millivolt = core.default_unit_database.known_units['millivolt']
milliwatt = core.default_unit_database.known_units['milliwatt']
milliweber = core.default_unit_database.known_units['milliweber']
minute = core.default_unit_database.known_units['minute']
ml = core.default_unit_database.known_units['ml']
mlm = core.default_unit_database.known_units['mlm']
mlx = core.default_unit_database.known_units['mlx']
mm = core.default_unit_database.known_units['mm']
mmol = core.default_unit_database.known_units['mmol']
mol = core.default_unit_database.known_units['mol']
mole = core.default_unit_database.known_units['mole']
mp = core.default_unit_database.known_units['mp']
mrad = core.default_unit_database.known_units['mrad']
ms = core.default_unit_database.known_units['ms']
msr = core.default_unit_database.known_units['msr']
mu0 = core.default_unit_database.known_units['mu0']
nA = core.default_unit_database.known_units['nA']
nBq = core.default_unit_database.known_units['nBq']
nC = core.default_unit_database.known_units['nC']
nF = core.default_unit_database.known_units['nF']
nGauss = core.default_unit_database.known_units['nGauss']
nH = core.default_unit_database.known_units['nH']
nHz = core.default_unit_database.known_units['nHz']
nJ = core.default_unit_database.known_units['nJ']
nK = core.default_unit_database.known_units['nK']
nN = core.default_unit_database.known_units['nN']
nOhm = core.default_unit_database.known_units['nOhm']
nPa = core.default_unit_database.known_units['nPa']
nS = core.default_unit_database.known_units['nS']
nT = core.default_unit_database.known_units['nT']
nV = core.default_unit_database.known_units['nV']
nW = core.default_unit_database.known_units['nW']
nWb = core.default_unit_database.known_units['nWb']
nanoampere = core.default_unit_database.known_units['nanoampere']
nanobecquerel = core.default_unit_database.known_units['nanobecquerel']
nanocandela = core.default_unit_database.known_units['nanocandela']
nanocoulomb = core.default_unit_database.known_units['nanocoulomb']
nanofarad = core.default_unit_database.known_units['nanofarad']
nanogauss = core.default_unit_database.known_units['nanogauss']
nanogram = core.default_unit_database.known_units['nanogram']
nanohenry = core.default_unit_database.known_units['nanohenry']
nanohertz = core.default_unit_database.known_units['nanohertz']
nanojoule = core.default_unit_database.known_units['nanojoule']
nanokelvin = core.default_unit_database.known_units['nanokelvin']
nanoliter = core.default_unit_database.known_units['nanoliter']
nanolumen = core.default_unit_database.known_units['nanolumen']
nanolux = core.default_unit_database.known_units['nanolux']
nanometer = core.default_unit_database.known_units['nanometer']
nanomole = core.default_unit_database.known_units['nanomole']
nanonewton = core.default_unit_database.known_units['nanonewton']
nanoohm = core.default_unit_database.known_units['nanoohm']
nanopascal = core.default_unit_database.known_units['nanopascal']
nanoradian = core.default_unit_database.known_units['nanoradian']
nanosecond = core.default_unit_database.known_units['nanosecond']
nanosiemens = core.default_unit_database.known_units['nanosiemens']
nanosteradian = core.default_unit_database.known_units['nanosteradian']
nanotesla = core.default_unit_database.known_units['nanotesla']
nanovolt = core.default_unit_database.known_units['nanovolt']
nanowatt = core.default_unit_database.known_units['nanowatt']
nanoweber = core.default_unit_database.known_units['nanoweber']
nautical_mile = core.default_unit_database.known_units['nautical_mile']
ncd = core.default_unit_database.known_units['ncd']
newton = core.default_unit_database.known_units['newton']
ng = core.default_unit_database.known_units['ng']
nl = core.default_unit_database.known_units['nl']
nlm = core.default_unit_database.known_units['nlm']
nlx = core.default_unit_database.known_units['nlx']
nm = core.default_unit_database.known_units['nm']
nmi = core.default_unit_database.known_units['nmi']
nmol = core.default_unit_database.known_units['nmol']
nrad = core.default_unit_database.known_units['nrad']
ns = core.default_unit_database.known_units['ns']
nsr = core.default_unit_database.known_units['nsr']
ohm = core.default_unit_database.known_units['ohm']
ounce = core.default_unit_database.known_units['ounce']
oz = core.default_unit_database.known_units['oz']
pA = core.default_unit_database.known_units['pA']
pBq = core.default_unit_database.known_units['pBq']
pC = core.default_unit_database.known_units['pC']
pF = core.default_unit_database.known_units['pF']
pGauss = core.default_unit_database.known_units['pGauss']
pH = core.default_unit_database.known_units['pH']
pHz = core.default_unit_database.known_units['pHz']
pJ = core.default_unit_database.known_units['pJ']
pK = core.default_unit_database.known_units['pK']
pN = core.default_unit_database.known_units['pN']
pOhm = core.default_unit_database.known_units['pOhm']
pPa = core.default_unit_database.known_units['pPa']
pS = core.default_unit_database.known_units['pS']
pT = core.default_unit_database.known_units['pT']
pV = core.default_unit_database.known_units['pV']
pW = core.default_unit_database.known_units['pW']
pWb = core.default_unit_database.known_units['pWb']
pascal = core.default_unit_database.known_units['pascal']
pcd = core.default_unit_database.known_units['pcd']
petaampere = core.default_unit_database.known_units['petaampere']
petabecquerel = core.default_unit_database.known_units['petabecquerel']
petacandela = core.default_unit_database.known_units['petacandela']
petacoulomb = core.default_unit_database.known_units['petacoulomb']
petafarad = core.default_unit_database.known_units['petafarad']
petagauss = core.default_unit_database.known_units['petagauss']
petagram = core.default_unit_database.known_units['petagram']
petahenry = core.default_unit_database.known_units['petahenry']
petahertz = core.default_unit_database.known_units['petahertz']
petajoule = core.default_unit_database.known_units['petajoule']
petakelvin = core.default_unit_database.known_units['petakelvin']
petaliter = core.default_unit_database.known_units['petaliter']
petalumen = core.default_unit_database.known_units['petalumen']
petalux = core.default_unit_database.known_units['petalux']
petameter = core.default_unit_database.known_units['petameter']
petamole = core.default_unit_database.known_units['petamole']
petanewton = core.default_unit_database.known_units['petanewton']
petaohm = core.default_unit_database.known_units['petaohm']
petapascal = core.default_unit_database.known_units['petapascal']
petaradian = core.default_unit_database.known_units['petaradian']
petasecond = core.default_unit_database.known_units['petasecond']
petasiemens = core.default_unit_database.known_units['petasiemens']
petasteradian = core.default_unit_database.known_units['petasteradian']
petatesla = core.default_unit_database.known_units['petatesla']
petavolt = core.default_unit_database.known_units['petavolt']
petawatt = core.default_unit_database.known_units['petawatt']
petaweber = core.default_unit_database.known_units['petaweber']
pg = core.default_unit_database.known_units['pg']
phi0 = core.default_unit_database.known_units['phi0']
picoampere = core.default_unit_database.known_units['picoampere']
picobecquerel = core.default_unit_database.known_units['picobecquerel']
picocandela = core.default_unit_database.known_units['picocandela']
picocoulomb = core.default_unit_database.known_units['picocoulomb']
picofarad = core.default_unit_database.known_units['picofarad']
picogauss = core.default_unit_database.known_units['picogauss']
picogram = core.default_unit_database.known_units['picogram']
picohenry = core.default_unit_database.known_units['picohenry']
picohertz = core.default_unit_database.known_units['picohertz']
picojoule = core.default_unit_database.known_units['picojoule']
picokelvin = core.default_unit_database.known_units['picokelvin']
picoliter = core.default_unit_database.known_units['picoliter']
picolumen = core.default_unit_database.known_units['picolumen']
picolux = core.default_unit_database.known_units['picolux']
picometer = core.default_unit_database.known_units['picometer']
picomole = core.default_unit_database.known_units['picomole']
piconewton = core.default_unit_database.known_units['piconewton']
picoohm = core.default_unit_database.known_units['picoohm']
picopascal = core.default_unit_database.known_units['picopascal']
picoradian = core.default_unit_database.known_units['picoradian']
picosecond = core.default_unit_database.known_units['picosecond']
picosiemens = core.default_unit_database.known_units['picosiemens']
picosteradian = core.default_unit_database.known_units['picosteradian']
picotesla = core.default_unit_database.known_units['picotesla']
picovolt = core.default_unit_database.known_units['picovolt']
picowatt = core.default_unit_database.known_units['picowatt']
picoweber = core.default_unit_database.known_units['picoweber']
pint = core.default_unit_database.known_units['pint']
pl = core.default_unit_database.known_units['pl']
planck_constant = core.default_unit_database.known_units['planck_constant']
plm = core.default_unit_database.known_units['plm']
plx = core.default_unit_database.known_units['plx']
pm = core.default_unit_database.known_units['pm']
pmol = core.default_unit_database.known_units['pmol']
pound = core.default_unit_database.known_units['pound']
pounds_per_square_inch = core.default_unit_database.known_units['pounds_per_square_inch']
prad = core.default_unit_database.known_units['prad']
proton_mass = core.default_unit_database.known_units['proton_mass']
ps = core.default_unit_database.known_units['ps']
psi = core.default_unit_database.known_units['psi']
psr = core.default_unit_database.known_units['psr']
qt = core.default_unit_database.known_units['qt']
quart = core.default_unit_database.known_units['quart']
rad = core.default_unit_database.known_units['rad']
radian = core.default_unit_database.known_units['radian']
reduced_planck_constant = core.default_unit_database.known_units['reduced_planck_constant']
rootHz = core.default_unit_database.known_units['rootHz']
s = core.default_unit_database.known_units['s']
second = core.default_unit_database.known_units['second']
siemens = core.default_unit_database.known_units['siemens']
speed_of_light = core.default_unit_database.known_units['speed_of_light']
sqrtHz = core.default_unit_database.known_units['sqrtHz']
sr = core.default_unit_database.known_units['sr']
steradian = core.default_unit_database.known_units['steradian']
tablespoon = core.default_unit_database.known_units['tablespoon']
tbsp = core.default_unit_database.known_units['tbsp']
teaspoon = core.default_unit_database.known_units['teaspoon']
teraampere = core.default_unit_database.known_units['teraampere']
terabecquerel = core.default_unit_database.known_units['terabecquerel']
teracandela = core.default_unit_database.known_units['teracandela']
teracoulomb = core.default_unit_database.known_units['teracoulomb']
terafarad = core.default_unit_database.known_units['terafarad']
teragauss = core.default_unit_database.known_units['teragauss']
teragram = core.default_unit_database.known_units['teragram']
terahenry = core.default_unit_database.known_units['terahenry']
terahertz = core.default_unit_database.known_units['terahertz']
terajoule = core.default_unit_database.known_units['terajoule']
terakelvin = core.default_unit_database.known_units['terakelvin']
teraliter = core.default_unit_database.known_units['teraliter']
teralumen = core.default_unit_database.known_units['teralumen']
teralux = core.default_unit_database.known_units['teralux']
terameter = core.default_unit_database.known_units['terameter']
teramole = core.default_unit_database.known_units['teramole']
teranewton = core.default_unit_database.known_units['teranewton']
teraohm = core.default_unit_database.known_units['teraohm']
terapascal = core.default_unit_database.known_units['terapascal']
teraradian = core.default_unit_database.known_units['teraradian']
terasecond = core.default_unit_database.known_units['terasecond']
terasiemens = core.default_unit_database.known_units['terasiemens']
terasteradian = core.default_unit_database.known_units['terasteradian']
teratesla = core.default_unit_database.known_units['teratesla']
teravolt = core.default_unit_database.known_units['teravolt']
terawatt = core.default_unit_database.known_units['terawatt']
teraweber = core.default_unit_database.known_units['teraweber']
tesla = core.default_unit_database.known_units['tesla']
ton = core.default_unit_database.known_units['ton']
tsp = core.default_unit_database.known_units['tsp']
uA = core.default_unit_database.known_units['uA']
uBq = core.default_unit_database.known_units['uBq']
uC = core.default_unit_database.known_units['uC']
uF = core.default_unit_database.known_units['uF']
uGauss = core.default_unit_database.known_units['uGauss']
uH = core.default_unit_database.known_units['uH']
uHz = core.default_unit_database.known_units['uHz']
uJ = core.default_unit_database.known_units['uJ']
uK = core.default_unit_database.known_units['uK']
uN = core.default_unit_database.known_units['uN']
uOhm = core.default_unit_database.known_units['uOhm']
uPa = core.default_unit_database.known_units['uPa']
uS = core.default_unit_database.known_units['uS']
uT = core.default_unit_database.known_units['uT']
uV = core.default_unit_database.known_units['uV']
uW = core.default_unit_database.known_units['uW']
uWb = core.default_unit_database.known_units['uWb']
ucd = core.default_unit_database.known_units['ucd']
ug = core.default_unit_database.known_units['ug']
ul = core.default_unit_database.known_units['ul']
ulm = core.default_unit_database.known_units['ulm']
ulx = core.default_unit_database.known_units['ulx']
um = core.default_unit_database.known_units['um']
umol = core.default_unit_database.known_units['umol']
urad = core.default_unit_database.known_units['urad']
us = core.default_unit_database.known_units['us']
us_gallon = core.default_unit_database.known_units['us_gallon']
usr = core.default_unit_database.known_units['usr']
vacuum_permeability = core.default_unit_database.known_units['vacuum_permeability']
vacuum_permittivity = core.default_unit_database.known_units['vacuum_permittivity']
volt = core.default_unit_database.known_units['volt']
watt = core.default_unit_database.known_units['watt']
weber = core.default_unit_database.known_units['weber']
week = core.default_unit_database.known_units['week']
wk = core.default_unit_database.known_units['wk']
yA = core.default_unit_database.known_units['yA']
yBq = core.default_unit_database.known_units['yBq']
yC = core.default_unit_database.known_units['yC']
yF = core.default_unit_database.known_units['yF']
yGauss = core.default_unit_database.known_units['yGauss']
yH = core.default_unit_database.known_units['yH']
yHz = core.default_unit_database.known_units['yHz']
yJ = core.default_unit_database.known_units['yJ']
yK = core.default_unit_database.known_units['yK']
yN = core.default_unit_database.known_units['yN']
yOhm = core.default_unit_database.known_units['yOhm']
yPa = core.default_unit_database.known_units['yPa']
yS = core.default_unit_database.known_units['yS']
yT = core.default_unit_database.known_units['yT']
yV = core.default_unit_database.known_units['yV']
yW = core.default_unit_database.known_units['yW']
yWb = core.default_unit_database.known_units['yWb']
yard = core.default_unit_database.known_units['yard']
ycd = core.default_unit_database.known_units['ycd']
yd = core.default_unit_database.known_units['yd']
year = core.default_unit_database.known_units['year']
yg = core.default_unit_database.known_units['yg']
yl = core.default_unit_database.known_units['yl']
ylm = core.default_unit_database.known_units['ylm']
ylx = core.default_unit_database.known_units['ylx']
ym = core.default_unit_database.known_units['ym']
ymol = core.default_unit_database.known_units['ymol']
yoctoampere = core.default_unit_database.known_units['yoctoampere']
yoctobecquerel = core.default_unit_database.known_units['yoctobecquerel']
yoctocandela = core.default_unit_database.known_units['yoctocandela']
yoctocoulomb = core.default_unit_database.known_units['yoctocoulomb']
yoctofarad = core.default_unit_database.known_units['yoctofarad']
yoctogauss = core.default_unit_database.known_units['yoctogauss']
yoctogram = core.default_unit_database.known_units['yoctogram']
yoctohenry = core.default_unit_database.known_units['yoctohenry']
yoctohertz = core.default_unit_database.known_units['yoctohertz']
yoctojoule = core.default_unit_database.known_units['yoctojoule']
yoctokelvin = core.default_unit_database.known_units['yoctokelvin']
yoctoliter = core.default_unit_database.known_units['yoctoliter']
yoctolumen = core.default_unit_database.known_units['yoctolumen']
yoctolux = core.default_unit_database.known_units['yoctolux']
yoctometer = core.default_unit_database.known_units['yoctometer']
yoctomole = core.default_unit_database.known_units['yoctomole']
yoctonewton = core.default_unit_database.known_units['yoctonewton']
yoctoohm = core.default_unit_database.known_units['yoctoohm']
yoctopascal = core.default_unit_database.known_units['yoctopascal']
yoctoradian = core.default_unit_database.known_units['yoctoradian']
yoctosecond = core.default_unit_database.known_units['yoctosecond']
yoctosiemens = core.default_unit_database.known_units['yoctosiemens']
yoctosteradian = core.default_unit_database.known_units['yoctosteradian']
yoctotesla = core.default_unit_database.known_units['yoctotesla']
yoctovolt = core.default_unit_database.known_units['yoctovolt']
yoctowatt = core.default_unit_database.known_units['yoctowatt']
yoctoweber = core.default_unit_database.known_units['yoctoweber']
yottaampere = core.default_unit_database.known_units['yottaampere']
yottabecquerel = core.default_unit_database.known_units['yottabecquerel']
yottacandela = core.default_unit_database.known_units['yottacandela']
yottacoulomb = core.default_unit_database.known_units['yottacoulomb']
yottafarad = core.default_unit_database.known_units['yottafarad']
yottagauss = core.default_unit_database.known_units['yottagauss']
yottagram = core.default_unit_database.known_units['yottagram']
yottahenry = core.default_unit_database.known_units['yottahenry']
yottahertz = core.default_unit_database.known_units['yottahertz']
yottajoule = core.default_unit_database.known_units['yottajoule']
yottakelvin = core.default_unit_database.known_units['yottakelvin']
yottaliter = core.default_unit_database.known_units['yottaliter']
yottalumen = core.default_unit_database.known_units['yottalumen']
yottalux = core.default_unit_database.known_units['yottalux']
yottameter = core.default_unit_database.known_units['yottameter']
yottamole = core.default_unit_database.known_units['yottamole']
yottanewton = core.default_unit_database.known_units['yottanewton']
yottaohm = core.default_unit_database.known_units['yottaohm']
yottapascal = core.default_unit_database.known_units['yottapascal']
yottaradian = core.default_unit_database.known_units['yottaradian']
yottasecond = core.default_unit_database.known_units['yottasecond']
yottasiemens = core.default_unit_database.known_units['yottasiemens']
yottasteradian = core.default_unit_database.known_units['yottasteradian']
yottatesla = core.default_unit_database.known_units['yottatesla']
yottavolt = core.default_unit_database.known_units['yottavolt']
yottawatt = core.default_unit_database.known_units['yottawatt']
yottaweber = core.default_unit_database.known_units['yottaweber']
yr = core.default_unit_database.known_units['yr']
yrad = core.default_unit_database.known_units['yrad']
ys = core.default_unit_database.known_units['ys']
ysr = core.default_unit_database.known_units['ysr']
zA = core.default_unit_database.known_units['zA']
zBq = core.default_unit_database.known_units['zBq']
zC = core.default_unit_database.known_units['zC']
zF = core.default_unit_database.known_units['zF']
zGauss = core.default_unit_database.known_units['zGauss']
zH = core.default_unit_database.known_units['zH']
zHz = core.default_unit_database.known_units['zHz']
zJ = core.default_unit_database.known_units['zJ']
zK = core.default_unit_database.known_units['zK']
zN = core.default_unit_database.known_units['zN']
zOhm = core.default_unit_database.known_units['zOhm']
zPa = core.default_unit_database.known_units['zPa']
zS = core.default_unit_database.known_units['zS']
zT = core.default_unit_database.known_units['zT']
zV = core.default_unit_database.known_units['zV']
zW = core.default_unit_database.known_units['zW']
zWb = core.default_unit_database.known_units['zWb']
zcd = core.default_unit_database.known_units['zcd']
zeptoampere = core.default_unit_database.known_units['zeptoampere']
zeptobecquerel = core.default_unit_database.known_units['zeptobecquerel']
zeptocandela = core.default_unit_database.known_units['zeptocandela']
zeptocoulomb = core.default_unit_database.known_units['zeptocoulomb']
zeptofarad = core.default_unit_database.known_units['zeptofarad']
zeptogauss = core.default_unit_database.known_units['zeptogauss']
zeptogram = core.default_unit_database.known_units['zeptogram']
zeptohenry = core.default_unit_database.known_units['zeptohenry']
zeptohertz = core.default_unit_database.known_units['zeptohertz']
zeptojoule = core.default_unit_database.known_units['zeptojoule']
zeptokelvin = core.default_unit_database.known_units['zeptokelvin']
zeptoliter = core.default_unit_database.known_units['zeptoliter']
zeptolumen = core.default_unit_database.known_units['zeptolumen']
zeptolux = core.default_unit_database.known_units['zeptolux']
zeptometer = core.default_unit_database.known_units['zeptometer']
zeptomole = core.default_unit_database.known_units['zeptomole']
zeptonewton = core.default_unit_database.known_units['zeptonewton']
zeptoohm = core.default_unit_database.known_units['zeptoohm']
zeptopascal = core.default_unit_database.known_units['zeptopascal']
zeptoradian = core.default_unit_database.known_units['zeptoradian']
zeptosecond = core.default_unit_database.known_units['zeptosecond']
zeptosiemens = core.default_unit_database.known_units['zeptosiemens']
zeptosteradian = core.default_unit_database.known_units['zeptosteradian']
zeptotesla = core.default_unit_database.known_units['zeptotesla']
zeptovolt = core.default_unit_database.known_units['zeptovolt']
zeptowatt = core.default_unit_database.known_units['zeptowatt']
zeptoweber = core.default_unit_database.known_units['zeptoweber']
zettaampere = core.default_unit_database.known_units['zettaampere']
zettabecquerel = core.default_unit_database.known_units['zettabecquerel']
zettacandela = core.default_unit_database.known_units['zettacandela']
zettacoulomb = core.default_unit_database.known_units['zettacoulomb']
zettafarad = core.default_unit_database.known_units['zettafarad']
zettagauss = core.default_unit_database.known_units['zettagauss']
zettagram = core.default_unit_database.known_units['zettagram']
zettahenry = core.default_unit_database.known_units['zettahenry']
zettahertz = core.default_unit_database.known_units['zettahertz']
zettajoule = core.default_unit_database.known_units['zettajoule']
zettakelvin = core.default_unit_database.known_units['zettakelvin']
zettaliter = core.default_unit_database.known_units['zettaliter']
zettalumen = core.default_unit_database.known_units['zettalumen']
zettalux = core.default_unit_database.known_units['zettalux']
zettameter = core.default_unit_database.known_units['zettameter']
zettamole = core.default_unit_database.known_units['zettamole']
zettanewton = core.default_unit_database.known_units['zettanewton']
zettaohm = core.default_unit_database.known_units['zettaohm']
zettapascal = core.default_unit_database.known_units['zettapascal']
zettaradian = core.default_unit_database.known_units['zettaradian']
zettasecond = core.default_unit_database.known_units['zettasecond']
zettasiemens = core.default_unit_database.known_units['zettasiemens']
zettasteradian = core.default_unit_database.known_units['zettasteradian']
zettatesla = core.default_unit_database.known_units['zettatesla']
zettavolt = core.default_unit_database.known_units['zettavolt']
zettawatt = core.default_unit_database.known_units['zettawatt']
zettaweber = core.default_unit_database.known_units['zettaweber']
zg = core.default_unit_database.known_units['zg']
zl = core.default_unit_database.known_units['zl']
zlm = core.default_unit_database.known_units['zlm']
zlx = core.default_unit_database.known_units['zlx']
zm = core.default_unit_database.known_units['zm']
zmol = core.default_unit_database.known_units['zmol']
zrad = core.default_unit_database.known_units['zrad']
zs = core.default_unit_database.known_units['zs']
zsr = core.default_unit_database.known_units['zsr']

__all__ = [
    "A",
    "Ang",
    "Bohr",
    "Bq",
    "C",
    "EA",
    "EBq",
    "EC",
    "EF",
    "EGauss",
    "EH",
    "EHz",
    "EJ",
    "EK",
    "EN",
    "EOhm",
    "EPa",
    "ES",
    "ET",
    "EV",
    "EW",
    "EWb",
    "Ecd",
    "Eg",
    "El",
    "Elm",
    "Elx",
    "Em",
    "Emol",
    "Erad",
    "Es",
    "Esr",
    "F",
    "G",
    "GA",
    "GBq",
    "GC",
    "GF",
    "GGauss",
    "GH",
    "GHz",
    "GJ",
    "GK",
    "GN",
    "GOhm",
    "GPa",
    "GS",
    "GT",
    "GV",
    "GW",
    "GWb",
    "Gauss",
    "Gcd",
    "Gg",
    "Gl",
    "Glm",
    "Glx",
    "Gm",
    "Gmol",
    "Grad",
    "Gs",
    "Gsr",
    "H",
    "Hartree",
    "Hz",
    "J",
    "K",
    "MA",
    "MBq",
    "MC",
    "MF",
    "MGauss",
    "MH",
    "MHz",
    "MJ",
    "MK",
    "MN",
    "MOhm",
    "MPa",
    "MS",
    "MT",
    "MV",
    "MW",
    "MWb",
    "Mcd",
    "Mg",
    "Ml",
    "Mlm",
    "Mlx",
    "Mm",
    "Mmol",
    "Mrad",
    "Ms",
    "Msr",
    "N",
    "Nav",
    "Ohm",
    "PA",
    "PBq",
    "PC",
    "PF",
    "PGauss",
    "PH",
    "PHz",
    "PJ",
    "PK",
    "PN",
    "POhm",
    "PPa",
    "PS",
    "PT",
    "PV",
    "PW",
    "PWb",
    "Pa",
    "Pcd",
    "Pg",
    "Pl",
    "Plm",
    "Plx",
    "Pm",
    "Pmol",
    "Prad",
    "Ps",
    "Psr",
    "R_k",
    "S",
    "T",
    "TA",
    "TBq",
    "TC",
    "TF",
    "TGauss",
    "TH",
    "THz",
    "TJ",
    "TK",
    "TN",
    "TOhm",
    "TPa",
    "TS",
    "TT",
    "TV",
    "TW",
    "TWb",
    "Tcd",
    "Tg",
    "Tl",
    "Tlm",
    "Tlx",
    "Tm",
    "Tmol",
    "Trad",
    "Ts",
    "Tsr",
    "V",
    "W",
    "Wb",
    "YA",
    "YBq",
    "YC",
    "YF",
    "YGauss",
    "YH",
    "YHz",
    "YJ",
    "YK",
    "YN",
    "YOhm",
    "YPa",
    "YS",
    "YT",
    "YV",
    "YW",
    "YWb",
    "Ycd",
    "Yg",
    "Yl",
    "Ylm",
    "Ylx",
    "Ym",
    "Ymol",
    "Yrad",
    "Ys",
    "Ysr",
    "ZA",
    "ZBq",
    "ZC",
    "ZF",
    "ZGauss",
    "ZH",
    "ZHz",
    "ZJ",
    "ZK",
    "ZN",
    "ZOhm",
    "ZPa",
    "ZS",
    "ZT",
    "ZV",
    "ZW",
    "ZWb",
    "Zcd",
    "Zg",
    "Zl",
    "Zlm",
    "Zlx",
    "Zm",
    "Zmol",
    "Zrad",
    "Zs",
    "Zsr",
    "aA",
    "aBq",
    "aC",
    "aF",
    "aGauss",
    "aH",
    "aHz",
    "aJ",
    "aK",
    "aN",
    "aOhm",
    "aPa",
    "aS",
    "aT",
    "aV",
    "aW",
    "aWb",
    "acd",
    "ag",
    "al",
    "alm",
    "alx",
    "am",
    "amol",
    "ampere",
    "amu",
    "angstrom",
    "arad",
    "asr",
    "atomic_mass_unit",
    "attoampere",
    "attobecquerel",
    "attocandela",
    "attocoulomb",
    "attofarad",
    "attogauss",
    "attogram",
    "attohenry",
    "attohertz",
    "attojoule",
    "attokelvin",
    "attoliter",
    "attolumen",
    "attolux",
    "attometer",
    "attomole",
    "attonewton",
    "attoohm",
    "attopascal",
    "attoradian",
    "attosecond",
    "attosiemens",
    "attosteradian",
    "attotesla",
    "attovolt",
    "attowatt",
    "attoweber",
    "avogadro_constant",
    "b",
    "bar",
    "barn",
    "becquerel",
    "bohr_magneton",
    "bohr_radius",
    "boltzmann_constant",
    "british_gallon",
    "c",
    "cA",
    "cBq",
    "cC",
    "cF",
    "cGauss",
    "cH",
    "cHz",
    "cJ",
    "cK",
    "cN",
    "cOhm",
    "cPa",
    "cS",
    "cT",
    "cV",
    "cW",
    "cWb",
    "candela",
    "ccd",
    "cd",
    "centiampere",
    "centibecquerel",
    "centicandela",
    "centicoulomb",
    "centifarad",
    "centigauss",
    "centigram",
    "centihenry",
    "centihertz",
    "centijoule",
    "centikelvin",
    "centiliter",
    "centilumen",
    "centilux",
    "centimeter",
    "centimole",
    "centinewton",
    "centiohm",
    "centipascal",
    "centiradian",
    "centisecond",
    "centisiemens",
    "centisteradian",
    "centitesla",
    "centivolt",
    "centiwatt",
    "centiweber",
    "cg",
    "cl",
    "clm",
    "clx",
    "cm",
    "cmol",
    "coulomb",
    "crad",
    "cs",
    "csr",
    "cup",
    "cyc",
    "cycle",
    "d",
    "dA",
    "dB",
    "dBm",
    "dBq",
    "dC",
    "dF",
    "dGauss",
    "dH",
    "dHz",
    "dJ",
    "dK",
    "dN",
    "dOhm",
    "dPa",
    "dS",
    "dT",
    "dV",
    "dW",
    "dWb",
    "daA",
    "daBq",
    "daC",
    "daF",
    "daGauss",
    "daH",
    "daHz",
    "daJ",
    "daK",
    "daN",
    "daOhm",
    "daPa",
    "daS",
    "daT",
    "daV",
    "daW",
    "daWb",
    "dacd",
    "dag",
    "dal",
    "dalm",
    "dalx",
    "dam",
    "damol",
    "darad",
    "das",
    "dasr",
    "day",
    "dcd",
    "deciampere",
    "decibecquerel",
    "decibel",
    "decicandela",
    "decicoulomb",
    "decifarad",
    "decigauss",
    "decigram",
    "decihenry",
    "decihertz",
    "decijoule",
    "decikelvin",
    "deciliter",
    "decilumen",
    "decilux",
    "decimeter",
    "decimole",
    "decinewton",
    "deciohm",
    "decipascal",
    "deciradian",
    "decisecond",
    "decisiemens",
    "decisteradian",
    "decitesla",
    "decivolt",
    "deciwatt",
    "deciweber",
    "deg",
    "degC",
    "degF",
    "degR",
    "dekaampere",
    "dekabecquerel",
    "dekacandela",
    "dekacoulomb",
    "dekafarad",
    "dekagauss",
    "dekagram",
    "dekahenry",
    "dekahertz",
    "dekajoule",
    "dekakelvin",
    "dekaliter",
    "dekalumen",
    "dekalux",
    "dekameter",
    "dekamole",
    "dekanewton",
    "dekaohm",
    "dekapascal",
    "dekaradian",
    "dekasecond",
    "dekasiemens",
    "dekasteradian",
    "dekatesla",
    "dekavolt",
    "dekawatt",
    "dekaweber",
    "dg",
    "dl",
    "dlm",
    "dlx",
    "dm",
    "dmol",
    "drad",
    "ds",
    "dsr",
    "e",
    "electron_mass",
    "elementary_charge",
    "eV",
    "eps0",
    "exaampere",
    "exabecquerel",
    "exacandela",
    "exacoulomb",
    "exafarad",
    "exagauss",
    "exagram",
    "exahenry",
    "exahertz",
    "exajoule",
    "exakelvin",
    "exaliter",
    "exalumen",
    "exalux",
    "exameter",
    "examole",
    "exanewton",
    "exaohm",
    "exapascal",
    "exaradian",
    "exasecond",
    "exasiemens",
    "exasteradian",
    "exatesla",
    "exavolt",
    "exawatt",
    "exaweber",
    "fA",
    "fBq",
    "fC",
    "fF",
    "fGauss",
    "fH",
    "fHz",
    "fJ",
    "fK",
    "fN",
    "fOhm",
    "fPa",
    "fS",
    "fT",
    "fV",
    "fW",
    "fWb",
    "farad",
    "fcd",
    "femtoampere",
    "femtobecquerel",
    "femtocandela",
    "femtocoulomb",
    "femtofarad",
    "femtogauss",
    "femtogram",
    "femtohenry",
    "femtohertz",
    "femtojoule",
    "femtokelvin",
    "femtoliter",
    "femtolumen",
    "femtolux",
    "femtometer",
    "femtomole",
    "femtonewton",
    "femtoohm",
    "femtopascal",
    "femtoradian",
    "femtosecond",
    "femtosiemens",
    "femtosteradian",
    "femtotesla",
    "femtovolt",
    "femtowatt",
    "femtoweber",
    "fg",
    "fl",
    "flm",
    "floz",
    "fluid_ounce",
    "flx",
    "fm",
    "fmol",
    "foot",
    "frad",
    "fs",
    "fsr",
    "ft",
    "g",
    "galUK",
    "galUS",
    "gauss",
    "gigaampere",
    "gigabecquerel",
    "gigacandela",
    "gigacoulomb",
    "gigafarad",
    "gigagauss",
    "gigagram",
    "gigahenry",
    "gigahertz",
    "gigajoule",
    "gigakelvin",
    "gigaliter",
    "gigalumen",
    "gigalux",
    "gigameter",
    "gigamole",
    "giganewton",
    "gigaohm",
    "gigapascal",
    "gigaradian",
    "gigasecond",
    "gigasiemens",
    "gigasteradian",
    "gigatesla",
    "gigavolt",
    "gigawatt",
    "gigaweber",
    "gram",
    "gravitational_constant",
    "h",
    "hA",
    "hBq",
    "hC",
    "hF",
    "hGauss",
    "hH",
    "hHz",
    "hJ",
    "hK",
    "hN",
    "hOhm",
    "hPa",
    "hS",
    "hT",
    "hV",
    "hW",
    "hWb",
    "ha",
    "hbar",
    "hcd",
    "hectare",
    "hectoampere",
    "hectobecquerel",
    "hectocandela",
    "hectocoulomb",
    "hectofarad",
    "hectogauss",
    "hectogram",
    "hectohenry",
    "hectohertz",
    "hectojoule",
    "hectokelvin",
    "hectoliter",
    "hectolumen",
    "hectolux",
    "hectometer",
    "hectomole",
    "hectonewton",
    "hectoohm",
    "hectopascal",
    "hectoradian",
    "hectosecond",
    "hectosiemens",
    "hectosteradian",
    "hectotesla",
    "hectovolt",
    "hectowatt",
    "hectoweber",
    "henry",
    "hertz",
    "hg",
    "hl",
    "hlm",
    "hlx",
    "hm",
    "hmol",
    "hour",
    "hplanck",
    "hrad",
    "hs",
    "hsr",
    "inch",
    "joule",
    "k",
    "kA",
    "kBq",
    "kC",
    "kF",
    "kGauss",
    "kH",
    "kHz",
    "kJ",
    "kK",
    "kN",
    "kOhm",
    "kPa",
    "kS",
    "kT",
    "kV",
    "kW",
    "kWb",
    "kcd",
    "kelvin",
    "kg",
    "kiloampere",
    "kilobecquerel",
    "kilocandela",
    "kilocoulomb",
    "kilofarad",
    "kilogauss",
    "kilogram",
    "kilohenry",
    "kilohertz",
    "kilojoule",
    "kilokelvin",
    "kiloliter",
    "kilolumen",
    "kilolux",
    "kilometer",
    "kilomole",
    "kilonewton",
    "kiloohm",
    "kilopascal",
    "kiloradian",
    "kilosecond",
    "kilosiemens",
    "kilosteradian",
    "kilotesla",
    "kilovolt",
    "kilowatt",
    "kiloweber",
    "kl",
    "klm",
    "klx",
    "km",
    "kmol",
    "krad",
    "ks",
    "ksr",
    "l",
    "lb",
    "light_year",
    "liter",
    "lm",
    "lumen",
    "lux",
    "lx",
    "ly",
    "lyr",
    "m",
    "mA",
    "mBq",
    "mC",
    "mF",
    "mGauss",
    "mH",
    "mHz",
    "mJ",
    "mK",
    "mN",
    "mOhm",
    "mPa",
    "mS",
    "mT",
    "mV",
    "mW",
    "mWb",
    "mcd",
    "me",
    "megaampere",
    "megabecquerel",
    "megacandela",
    "megacoulomb",
    "megafarad",
    "megagauss",
    "megagram",
    "megahenry",
    "megahertz",
    "megajoule",
    "megakelvin",
    "megaliter",
    "megalumen",
    "megalux",
    "megameter",
    "megamole",
    "meganewton",
    "megaohm",
    "megapascal",
    "megaradian",
    "megasecond",
    "megasiemens",
    "megasteradian",
    "megatesla",
    "megavolt",
    "megawatt",
    "megaweber",
    "meter",
    "mg",
    "microampere",
    "microbecquerel",
    "microcandela",
    "microcoulomb",
    "microfarad",
    "microgauss",
    "microgram",
    "microhenry",
    "microhertz",
    "microjoule",
    "microkelvin",
    "microliter",
    "microlumen",
    "microlux",
    "micrometer",
    "micromole",
    "micronewton",
    "microohm",
    "micropascal",
    "microradian",
    "microsecond",
    "microsiemens",
    "microsteradian",
    "microtesla",
    "microvolt",
    "microwatt",
    "microweber",
    "milliampere",
    "millibecquerel",
    "millicandela",
    "millicoulomb",
    "millifarad",
    "milligauss",
    "milligram",
    "millihenry",
    "millihertz",
    "millijoule",
    "millikelvin",
    "milliliter",
    "millilumen",
    "millilux",
    "millimeter",
    "millimole",
    "millinewton",
    "milliohm",
    "millipascal",
    "milliradian",
    "millisecond",
    "millisiemens",
    "millisteradian",
    "millitesla",
    "millivolt",
    "milliwatt",
    "milliweber",
    "minute",
    "ml",
    "mlm",
    "mlx",
    "mm",
    "mmol",
    "mol",
    "mole",
    "mp",
    "mrad",
    "ms",
    "msr",
    "mu0",
    "nA",
    "nBq",
    "nC",
    "nF",
    "nGauss",
    "nH",
    "nHz",
    "nJ",
    "nK",
    "nN",
    "nOhm",
    "nPa",
    "nS",
    "nT",
    "nV",
    "nW",
    "nWb",
    "nanoampere",
    "nanobecquerel",
    "nanocandela",
    "nanocoulomb",
    "nanofarad",
    "nanogauss",
    "nanogram",
    "nanohenry",
    "nanohertz",
    "nanojoule",
    "nanokelvin",
    "nanoliter",
    "nanolumen",
    "nanolux",
    "nanometer",
    "nanomole",
    "nanonewton",
    "nanoohm",
    "nanopascal",
    "nanoradian",
    "nanosecond",
    "nanosiemens",
    "nanosteradian",
    "nanotesla",
    "nanovolt",
    "nanowatt",
    "nanoweber",
    "nautical_mile",
    "ncd",
    "newton",
    "ng",
    "nl",
    "nlm",
    "nlx",
    "nm",
    "nmi",
    "nmol",
    "nrad",
    "ns",
    "nsr",
    "ohm",
    "ounce",
    "oz",
    "pA",
    "pBq",
    "pC",
    "pF",
    "pGauss",
    "pH",
    "pHz",
    "pJ",
    "pK",
    "pN",
    "pOhm",
    "pPa",
    "pS",
    "pT",
    "pV",
    "pW",
    "pWb",
    "pascal",
    "pcd",
    "petaampere",
    "petabecquerel",
    "petacandela",
    "petacoulomb",
    "petafarad",
    "petagauss",
    "petagram",
    "petahenry",
    "petahertz",
    "petajoule",
    "petakelvin",
    "petaliter",
    "petalumen",
    "petalux",
    "petameter",
    "petamole",
    "petanewton",
    "petaohm",
    "petapascal",
    "petaradian",
    "petasecond",
    "petasiemens",
    "petasteradian",
    "petatesla",
    "petavolt",
    "petawatt",
    "petaweber",
    "pg",
    "phi0",
    "picoampere",
    "picobecquerel",
    "picocandela",
    "picocoulomb",
    "picofarad",
    "picogauss",
    "picogram",
    "picohenry",
    "picohertz",
    "picojoule",
    "picokelvin",
    "picoliter",
    "picolumen",
    "picolux",
    "picometer",
    "picomole",
    "piconewton",
    "picoohm",
    "picopascal",
    "picoradian",
    "picosecond",
    "picosiemens",
    "picosteradian",
    "picotesla",
    "picovolt",
    "picowatt",
    "picoweber",
    "pint",
    "pl",
    "planck_constant",
    "plm",
    "plx",
    "pm",
    "pmol",
    "pound",
    "pounds_per_square_inch",
    "prad",
    "proton_mass",
    "ps",
    "psi",
    "psr",
    "qt",
    "quart",
    "rad",
    "radian",
    "reduced_planck_constant",
    "rootHz",
    "s",
    "second",
    "siemens",
    "speed_of_light",
    "sqrtHz",
    "sr",
    "steradian",
    "tablespoon",
    "tbsp",
    "teaspoon",
    "teraampere",
    "terabecquerel",
    "teracandela",
    "teracoulomb",
    "terafarad",
    "teragauss",
    "teragram",
    "terahenry",
    "terahertz",
    "terajoule",
    "terakelvin",
    "teraliter",
    "teralumen",
    "teralux",
    "terameter",
    "teramole",
    "teranewton",
    "teraohm",
    "terapascal",
    "teraradian",
    "terasecond",
    "terasiemens",
    "terasteradian",
    "teratesla",
    "teravolt",
    "terawatt",
    "teraweber",
    "tesla",
    "ton",
    "tsp",
    "uA",
    "uBq",
    "uC",
    "uF",
    "uGauss",
    "uH",
    "uHz",
    "uJ",
    "uK",
    "uN",
    "uOhm",
    "uPa",
    "uS",
    "uT",
    "uV",
    "uW",
    "uWb",
    "ucd",
    "ug",
    "ul",
    "ulm",
    "ulx",
    "um",
    "umol",
    "urad",
    "us",
    "us_gallon",
    "usr",
    "vacuum_permeability",
    "vacuum_permittivity",
    "volt",
    "watt",
    "weber",
    "week",
    "wk",
    "yA",
    "yBq",
    "yC",
    "yF",
    "yGauss",
    "yH",
    "yHz",
    "yJ",
    "yK",
    "yN",
    "yOhm",
    "yPa",
    "yS",
    "yT",
    "yV",
    "yW",
    "yWb",
    "yard",
    "ycd",
    "yd",
    "year",
    "yg",
    "yl",
    "ylm",
    "ylx",
    "ym",
    "ymol",
    "yoctoampere",
    "yoctobecquerel",
    "yoctocandela",
    "yoctocoulomb",
    "yoctofarad",
    "yoctogauss",
    "yoctogram",
    "yoctohenry",
    "yoctohertz",
    "yoctojoule",
    "yoctokelvin",
    "yoctoliter",
    "yoctolumen",
    "yoctolux",
    "yoctometer",
    "yoctomole",
    "yoctonewton",
    "yoctoohm",
    "yoctopascal",
    "yoctoradian",
    "yoctosecond",
    "yoctosiemens",
    "yoctosteradian",
    "yoctotesla",
    "yoctovolt",
    "yoctowatt",
    "yoctoweber",
    "yottaampere",
    "yottabecquerel",
    "yottacandela",
    "yottacoulomb",
    "yottafarad",
    "yottagauss",
    "yottagram",
    "yottahenry",
    "yottahertz",
    "yottajoule",
    "yottakelvin",
    "yottaliter",
    "yottalumen",
    "yottalux",
    "yottameter",
    "yottamole",
    "yottanewton",
    "yottaohm",
    "yottapascal",
    "yottaradian",
    "yottasecond",
    "yottasiemens",
    "yottasteradian",
    "yottatesla",
    "yottavolt",
    "yottawatt",
    "yottaweber",
    "yr",
    "yrad",
    "ys",
    "ysr",
    "zA",
    "zBq",
    "zC",
    "zF",
    "zGauss",
    "zH",
    "zHz",
    "zJ",
    "zK",
    "zN",
    "zOhm",
    "zPa",
    "zS",
    "zT",
    "zV",
    "zW",
    "zWb",
    "zcd",
    "zeptoampere",
    "zeptobecquerel",
    "zeptocandela",
    "zeptocoulomb",
    "zeptofarad",
    "zeptogauss",
    "zeptogram",
    "zeptohenry",
    "zeptohertz",
    "zeptojoule",
    "zeptokelvin",
    "zeptoliter",
    "zeptolumen",
    "zeptolux",
    "zeptometer",
    "zeptomole",
    "zeptonewton",
    "zeptoohm",
    "zeptopascal",
    "zeptoradian",
    "zeptosecond",
    "zeptosiemens",
    "zeptosteradian",
    "zeptotesla",
    "zeptovolt",
    "zeptowatt",
    "zeptoweber",
    "zettaampere",
    "zettabecquerel",
    "zettacandela",
    "zettacoulomb",
    "zettafarad",
    "zettagauss",
    "zettagram",
    "zettahenry",
    "zettahertz",
    "zettajoule",
    "zettakelvin",
    "zettaliter",
    "zettalumen",
    "zettalux",
    "zettameter",
    "zettamole",
    "zettanewton",
    "zettaohm",
    "zettapascal",
    "zettaradian",
    "zettasecond",
    "zettasiemens",
    "zettasteradian",
    "zettatesla",
    "zettavolt",
    "zettawatt",
    "zettaweber",
    "zg",
    "zl",
    "zlm",
    "zlx",
    "zm",
    "zmol",
    "zrad",
    "zs",
    "zsr",
]
