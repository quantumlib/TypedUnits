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
A = core.ElectricCurrent(core.default_unit_database.known_units['A'])
Ang = core.Length(core.default_unit_database.known_units['Ang'])
Bohr = core.Length(core.default_unit_database.known_units['Bohr'])
Bq = core.Frequency(core.default_unit_database.known_units['Bq'])
C = core.Charge(core.default_unit_database.known_units['C'])
EA = core.ElectricCurrent(core.default_unit_database.known_units['EA'])
EBq = core.Frequency(core.default_unit_database.known_units['EBq'])
EC = core.Charge(core.default_unit_database.known_units['EC'])
EF = core.Capacitance(core.default_unit_database.known_units['EF'])
EGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['EGauss'])
EH = core.Inductance(core.default_unit_database.known_units['EH'])
EHz = core.Frequency(core.default_unit_database.known_units['EHz'])
EJ = core.Energy(core.default_unit_database.known_units['EJ'])
EK = core.Temperature(core.default_unit_database.known_units['EK'])
EN = core.Force(core.default_unit_database.known_units['EN'])
EOhm = core.Resistance(core.default_unit_database.known_units['EOhm'])
EPa = core.Pressure(core.default_unit_database.known_units['EPa'])
ES = core.ElectricalConductance(core.default_unit_database.known_units['ES'])
ET = core.MagneticFluxDensity(core.default_unit_database.known_units['ET'])
EV = core.ElectricPotential(core.default_unit_database.known_units['EV'])
EW = core.Power(core.default_unit_database.known_units['EW'])
EWb = core.MagneticFlux(core.default_unit_database.known_units['EWb'])
Ecd = core.LuminousIntensity(core.default_unit_database.known_units['Ecd'])
Eg = core.Mass(core.default_unit_database.known_units['Eg'])
El = core.Volume(core.default_unit_database.known_units['El'])
Elm = core.LuminousFlux(core.default_unit_database.known_units['Elm'])
Elx = core.Illuminance(core.default_unit_database.known_units['Elx'])
Em = core.Length(core.default_unit_database.known_units['Em'])
Emol = core.Quantity(core.default_unit_database.known_units['Emol'])
Erad = core.Angle(core.default_unit_database.known_units['Erad'])
Es = core.Time(core.default_unit_database.known_units['Es'])
Esr = core.Angle(core.default_unit_database.known_units['Esr'])
F = core.Capacitance(core.default_unit_database.known_units['F'])
G = core.default_unit_database.known_units['G']
GA = core.ElectricCurrent(core.default_unit_database.known_units['GA'])
GBq = core.Frequency(core.default_unit_database.known_units['GBq'])
GC = core.Charge(core.default_unit_database.known_units['GC'])
GF = core.Capacitance(core.default_unit_database.known_units['GF'])
GGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['GGauss'])
GH = core.Inductance(core.default_unit_database.known_units['GH'])
GHz = core.Frequency(core.default_unit_database.known_units['GHz'])
GJ = core.Energy(core.default_unit_database.known_units['GJ'])
GK = core.Temperature(core.default_unit_database.known_units['GK'])
GN = core.Force(core.default_unit_database.known_units['GN'])
GOhm = core.Resistance(core.default_unit_database.known_units['GOhm'])
GPa = core.Pressure(core.default_unit_database.known_units['GPa'])
GS = core.ElectricalConductance(core.default_unit_database.known_units['GS'])
GT = core.MagneticFluxDensity(core.default_unit_database.known_units['GT'])
GV = core.ElectricPotential(core.default_unit_database.known_units['GV'])
GW = core.Power(core.default_unit_database.known_units['GW'])
GWb = core.MagneticFlux(core.default_unit_database.known_units['GWb'])
Gauss = core.MagneticFluxDensity(core.default_unit_database.known_units['Gauss'])
Gcd = core.LuminousIntensity(core.default_unit_database.known_units['Gcd'])
Gg = core.Mass(core.default_unit_database.known_units['Gg'])
Gl = core.Volume(core.default_unit_database.known_units['Gl'])
Glm = core.LuminousFlux(core.default_unit_database.known_units['Glm'])
Glx = core.Illuminance(core.default_unit_database.known_units['Glx'])
Gm = core.Length(core.default_unit_database.known_units['Gm'])
Gmol = core.Quantity(core.default_unit_database.known_units['Gmol'])
Grad = core.Angle(core.default_unit_database.known_units['Grad'])
Gs = core.Time(core.default_unit_database.known_units['Gs'])
Gsr = core.Angle(core.default_unit_database.known_units['Gsr'])
H = core.Inductance(core.default_unit_database.known_units['H'])
Hartree = core.Energy(core.default_unit_database.known_units['Hartree'])
Hz = core.Frequency(core.default_unit_database.known_units['Hz'])
J = core.Energy(core.default_unit_database.known_units['J'])
K = core.Temperature(core.default_unit_database.known_units['K'])
MA = core.ElectricCurrent(core.default_unit_database.known_units['MA'])
MBq = core.Frequency(core.default_unit_database.known_units['MBq'])
MC = core.Charge(core.default_unit_database.known_units['MC'])
MF = core.Capacitance(core.default_unit_database.known_units['MF'])
MGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['MGauss'])
MH = core.Inductance(core.default_unit_database.known_units['MH'])
MHz = core.Frequency(core.default_unit_database.known_units['MHz'])
MJ = core.Energy(core.default_unit_database.known_units['MJ'])
MK = core.Temperature(core.default_unit_database.known_units['MK'])
MN = core.Force(core.default_unit_database.known_units['MN'])
MOhm = core.Resistance(core.default_unit_database.known_units['MOhm'])
MPa = core.Pressure(core.default_unit_database.known_units['MPa'])
MS = core.ElectricalConductance(core.default_unit_database.known_units['MS'])
MT = core.MagneticFluxDensity(core.default_unit_database.known_units['MT'])
MV = core.ElectricPotential(core.default_unit_database.known_units['MV'])
MW = core.Power(core.default_unit_database.known_units['MW'])
MWb = core.MagneticFlux(core.default_unit_database.known_units['MWb'])
Mcd = core.LuminousIntensity(core.default_unit_database.known_units['Mcd'])
Mg = core.Mass(core.default_unit_database.known_units['Mg'])
Ml = core.Volume(core.default_unit_database.known_units['Ml'])
Mlm = core.LuminousFlux(core.default_unit_database.known_units['Mlm'])
Mlx = core.Illuminance(core.default_unit_database.known_units['Mlx'])
Mm = core.Length(core.default_unit_database.known_units['Mm'])
Mmol = core.Quantity(core.default_unit_database.known_units['Mmol'])
Mrad = core.Angle(core.default_unit_database.known_units['Mrad'])
Ms = core.Time(core.default_unit_database.known_units['Ms'])
Msr = core.Angle(core.default_unit_database.known_units['Msr'])
N = core.Force(core.default_unit_database.known_units['N'])
Nav = core.default_unit_database.known_units['Nav']
Ohm = core.Resistance(core.default_unit_database.known_units['Ohm'])
PA = core.ElectricCurrent(core.default_unit_database.known_units['PA'])
PBq = core.Frequency(core.default_unit_database.known_units['PBq'])
PC = core.Charge(core.default_unit_database.known_units['PC'])
PF = core.Capacitance(core.default_unit_database.known_units['PF'])
PGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['PGauss'])
PH = core.Inductance(core.default_unit_database.known_units['PH'])
PHz = core.Frequency(core.default_unit_database.known_units['PHz'])
PJ = core.Energy(core.default_unit_database.known_units['PJ'])
PK = core.Temperature(core.default_unit_database.known_units['PK'])
PN = core.Force(core.default_unit_database.known_units['PN'])
POhm = core.Resistance(core.default_unit_database.known_units['POhm'])
PPa = core.Pressure(core.default_unit_database.known_units['PPa'])
PS = core.ElectricalConductance(core.default_unit_database.known_units['PS'])
PT = core.MagneticFluxDensity(core.default_unit_database.known_units['PT'])
PV = core.ElectricPotential(core.default_unit_database.known_units['PV'])
PW = core.Power(core.default_unit_database.known_units['PW'])
PWb = core.MagneticFlux(core.default_unit_database.known_units['PWb'])
Pa = core.Pressure(core.default_unit_database.known_units['Pa'])
Pcd = core.LuminousIntensity(core.default_unit_database.known_units['Pcd'])
Pg = core.Mass(core.default_unit_database.known_units['Pg'])
Pl = core.Volume(core.default_unit_database.known_units['Pl'])
Plm = core.LuminousFlux(core.default_unit_database.known_units['Plm'])
Plx = core.Illuminance(core.default_unit_database.known_units['Plx'])
Pm = core.Length(core.default_unit_database.known_units['Pm'])
Pmol = core.Quantity(core.default_unit_database.known_units['Pmol'])
Prad = core.Angle(core.default_unit_database.known_units['Prad'])
Ps = core.Time(core.default_unit_database.known_units['Ps'])
Psr = core.Angle(core.default_unit_database.known_units['Psr'])
R_k = core.Resistance(core.default_unit_database.known_units['R_k'])
S = core.ElectricalConductance(core.default_unit_database.known_units['S'])
T = core.MagneticFluxDensity(core.default_unit_database.known_units['T'])
TA = core.ElectricCurrent(core.default_unit_database.known_units['TA'])
TBq = core.Frequency(core.default_unit_database.known_units['TBq'])
TC = core.Charge(core.default_unit_database.known_units['TC'])
TF = core.Capacitance(core.default_unit_database.known_units['TF'])
TGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['TGauss'])
TH = core.Inductance(core.default_unit_database.known_units['TH'])
THz = core.Frequency(core.default_unit_database.known_units['THz'])
TJ = core.Energy(core.default_unit_database.known_units['TJ'])
TK = core.Temperature(core.default_unit_database.known_units['TK'])
TN = core.Force(core.default_unit_database.known_units['TN'])
TOhm = core.Resistance(core.default_unit_database.known_units['TOhm'])
TPa = core.Pressure(core.default_unit_database.known_units['TPa'])
TS = core.ElectricalConductance(core.default_unit_database.known_units['TS'])
TT = core.MagneticFluxDensity(core.default_unit_database.known_units['TT'])
TV = core.ElectricPotential(core.default_unit_database.known_units['TV'])
TW = core.Power(core.default_unit_database.known_units['TW'])
TWb = core.MagneticFlux(core.default_unit_database.known_units['TWb'])
Tcd = core.LuminousIntensity(core.default_unit_database.known_units['Tcd'])
Tg = core.Mass(core.default_unit_database.known_units['Tg'])
Tl = core.Volume(core.default_unit_database.known_units['Tl'])
Tlm = core.LuminousFlux(core.default_unit_database.known_units['Tlm'])
Tlx = core.Illuminance(core.default_unit_database.known_units['Tlx'])
Tm = core.Length(core.default_unit_database.known_units['Tm'])
Tmol = core.Quantity(core.default_unit_database.known_units['Tmol'])
Trad = core.Angle(core.default_unit_database.known_units['Trad'])
Ts = core.Time(core.default_unit_database.known_units['Ts'])
Tsr = core.Angle(core.default_unit_database.known_units['Tsr'])
V = core.ElectricPotential(core.default_unit_database.known_units['V'])
W = core.Power(core.default_unit_database.known_units['W'])
Wb = core.MagneticFlux(core.default_unit_database.known_units['Wb'])
YA = core.ElectricCurrent(core.default_unit_database.known_units['YA'])
YBq = core.Frequency(core.default_unit_database.known_units['YBq'])
YC = core.Charge(core.default_unit_database.known_units['YC'])
YF = core.Capacitance(core.default_unit_database.known_units['YF'])
YGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['YGauss'])
YH = core.Inductance(core.default_unit_database.known_units['YH'])
YHz = core.Frequency(core.default_unit_database.known_units['YHz'])
YJ = core.Energy(core.default_unit_database.known_units['YJ'])
YK = core.Temperature(core.default_unit_database.known_units['YK'])
YN = core.Force(core.default_unit_database.known_units['YN'])
YOhm = core.Resistance(core.default_unit_database.known_units['YOhm'])
YPa = core.Pressure(core.default_unit_database.known_units['YPa'])
YS = core.ElectricalConductance(core.default_unit_database.known_units['YS'])
YT = core.MagneticFluxDensity(core.default_unit_database.known_units['YT'])
YV = core.ElectricPotential(core.default_unit_database.known_units['YV'])
YW = core.Power(core.default_unit_database.known_units['YW'])
YWb = core.MagneticFlux(core.default_unit_database.known_units['YWb'])
Ycd = core.LuminousIntensity(core.default_unit_database.known_units['Ycd'])
Yg = core.Mass(core.default_unit_database.known_units['Yg'])
Yl = core.Volume(core.default_unit_database.known_units['Yl'])
Ylm = core.LuminousFlux(core.default_unit_database.known_units['Ylm'])
Ylx = core.Illuminance(core.default_unit_database.known_units['Ylx'])
Ym = core.Length(core.default_unit_database.known_units['Ym'])
Ymol = core.Quantity(core.default_unit_database.known_units['Ymol'])
Yrad = core.Angle(core.default_unit_database.known_units['Yrad'])
Ys = core.Time(core.default_unit_database.known_units['Ys'])
Ysr = core.Angle(core.default_unit_database.known_units['Ysr'])
ZA = core.ElectricCurrent(core.default_unit_database.known_units['ZA'])
ZBq = core.Frequency(core.default_unit_database.known_units['ZBq'])
ZC = core.Charge(core.default_unit_database.known_units['ZC'])
ZF = core.Capacitance(core.default_unit_database.known_units['ZF'])
ZGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['ZGauss'])
ZH = core.Inductance(core.default_unit_database.known_units['ZH'])
ZHz = core.Frequency(core.default_unit_database.known_units['ZHz'])
ZJ = core.Energy(core.default_unit_database.known_units['ZJ'])
ZK = core.Temperature(core.default_unit_database.known_units['ZK'])
ZN = core.Force(core.default_unit_database.known_units['ZN'])
ZOhm = core.Resistance(core.default_unit_database.known_units['ZOhm'])
ZPa = core.Pressure(core.default_unit_database.known_units['ZPa'])
ZS = core.ElectricalConductance(core.default_unit_database.known_units['ZS'])
ZT = core.MagneticFluxDensity(core.default_unit_database.known_units['ZT'])
ZV = core.ElectricPotential(core.default_unit_database.known_units['ZV'])
ZW = core.Power(core.default_unit_database.known_units['ZW'])
ZWb = core.MagneticFlux(core.default_unit_database.known_units['ZWb'])
Zcd = core.LuminousIntensity(core.default_unit_database.known_units['Zcd'])
Zg = core.Mass(core.default_unit_database.known_units['Zg'])
Zl = core.Volume(core.default_unit_database.known_units['Zl'])
Zlm = core.LuminousFlux(core.default_unit_database.known_units['Zlm'])
Zlx = core.Illuminance(core.default_unit_database.known_units['Zlx'])
Zm = core.Length(core.default_unit_database.known_units['Zm'])
Zmol = core.Quantity(core.default_unit_database.known_units['Zmol'])
Zrad = core.Angle(core.default_unit_database.known_units['Zrad'])
Zs = core.Time(core.default_unit_database.known_units['Zs'])
Zsr = core.Angle(core.default_unit_database.known_units['Zsr'])
aA = core.ElectricCurrent(core.default_unit_database.known_units['aA'])
aBq = core.Frequency(core.default_unit_database.known_units['aBq'])
aC = core.Charge(core.default_unit_database.known_units['aC'])
aF = core.Capacitance(core.default_unit_database.known_units['aF'])
aGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['aGauss'])
aH = core.Inductance(core.default_unit_database.known_units['aH'])
aHz = core.Frequency(core.default_unit_database.known_units['aHz'])
aJ = core.Energy(core.default_unit_database.known_units['aJ'])
aK = core.Temperature(core.default_unit_database.known_units['aK'])
aN = core.Force(core.default_unit_database.known_units['aN'])
aOhm = core.Resistance(core.default_unit_database.known_units['aOhm'])
aPa = core.Pressure(core.default_unit_database.known_units['aPa'])
aS = core.ElectricalConductance(core.default_unit_database.known_units['aS'])
aT = core.MagneticFluxDensity(core.default_unit_database.known_units['aT'])
aV = core.ElectricPotential(core.default_unit_database.known_units['aV'])
aW = core.Power(core.default_unit_database.known_units['aW'])
aWb = core.MagneticFlux(core.default_unit_database.known_units['aWb'])
acd = core.LuminousIntensity(core.default_unit_database.known_units['acd'])
ag = core.Mass(core.default_unit_database.known_units['ag'])
al = core.Volume(core.default_unit_database.known_units['al'])
alm = core.LuminousFlux(core.default_unit_database.known_units['alm'])
alx = core.Illuminance(core.default_unit_database.known_units['alx'])
am = core.Length(core.default_unit_database.known_units['am'])
amol = core.Quantity(core.default_unit_database.known_units['amol'])
ampere = core.ElectricCurrent(core.default_unit_database.known_units['ampere'])
amu = core.Mass(core.default_unit_database.known_units['amu'])
angstrom = core.Length(core.default_unit_database.known_units['angstrom'])
arad = core.Angle(core.default_unit_database.known_units['arad'])
asr = core.Angle(core.default_unit_database.known_units['asr'])
atomic_mass_unit = core.Mass(core.default_unit_database.known_units['atomic_mass_unit'])
attoampere = core.ElectricCurrent(core.default_unit_database.known_units['attoampere'])
attobecquerel = core.Frequency(core.default_unit_database.known_units['attobecquerel'])
attocandela = core.LuminousIntensity(core.default_unit_database.known_units['attocandela'])
attocoulomb = core.Charge(core.default_unit_database.known_units['attocoulomb'])
attofarad = core.Capacitance(core.default_unit_database.known_units['attofarad'])
attogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['attogauss'])
attogram = core.Mass(core.default_unit_database.known_units['attogram'])
attohenry = core.Inductance(core.default_unit_database.known_units['attohenry'])
attohertz = core.Frequency(core.default_unit_database.known_units['attohertz'])
attojoule = core.Energy(core.default_unit_database.known_units['attojoule'])
attokelvin = core.Temperature(core.default_unit_database.known_units['attokelvin'])
attoliter = core.Volume(core.default_unit_database.known_units['attoliter'])
attolumen = core.LuminousFlux(core.default_unit_database.known_units['attolumen'])
attolux = core.Illuminance(core.default_unit_database.known_units['attolux'])
attometer = core.Length(core.default_unit_database.known_units['attometer'])
attomole = core.Quantity(core.default_unit_database.known_units['attomole'])
attonewton = core.Force(core.default_unit_database.known_units['attonewton'])
attoohm = core.Resistance(core.default_unit_database.known_units['attoohm'])
attopascal = core.Pressure(core.default_unit_database.known_units['attopascal'])
attoradian = core.Angle(core.default_unit_database.known_units['attoradian'])
attosecond = core.Time(core.default_unit_database.known_units['attosecond'])
attosiemens = core.ElectricalConductance(core.default_unit_database.known_units['attosiemens'])
attosteradian = core.Angle(core.default_unit_database.known_units['attosteradian'])
attotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['attotesla'])
attovolt = core.ElectricPotential(core.default_unit_database.known_units['attovolt'])
attowatt = core.Power(core.default_unit_database.known_units['attowatt'])
attoweber = core.MagneticFlux(core.default_unit_database.known_units['attoweber'])
avogadro_constant = core.default_unit_database.known_units['avogadro_constant']
b = core.Area(core.default_unit_database.known_units['b'])
bar = core.Pressure(core.default_unit_database.known_units['bar'])
barn = core.Area(core.default_unit_database.known_units['barn'])
becquerel = core.Frequency(core.default_unit_database.known_units['becquerel'])
bohr_magneton = core.default_unit_database.known_units['bohr_magneton']
bohr_radius = core.Length(core.default_unit_database.known_units['bohr_radius'])
boltzmann_constant = core.default_unit_database.known_units['boltzmann_constant']
british_gallon = core.Volume(core.default_unit_database.known_units['british_gallon'])
c = core.Speed(core.default_unit_database.known_units['c'])
cA = core.ElectricCurrent(core.default_unit_database.known_units['cA'])
cBq = core.Frequency(core.default_unit_database.known_units['cBq'])
cC = core.Charge(core.default_unit_database.known_units['cC'])
cF = core.Capacitance(core.default_unit_database.known_units['cF'])
cGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['cGauss'])
cH = core.Inductance(core.default_unit_database.known_units['cH'])
cHz = core.Frequency(core.default_unit_database.known_units['cHz'])
cJ = core.Energy(core.default_unit_database.known_units['cJ'])
cK = core.Temperature(core.default_unit_database.known_units['cK'])
cN = core.Force(core.default_unit_database.known_units['cN'])
cOhm = core.Resistance(core.default_unit_database.known_units['cOhm'])
cPa = core.Pressure(core.default_unit_database.known_units['cPa'])
cS = core.ElectricalConductance(core.default_unit_database.known_units['cS'])
cT = core.MagneticFluxDensity(core.default_unit_database.known_units['cT'])
cV = core.ElectricPotential(core.default_unit_database.known_units['cV'])
cW = core.Power(core.default_unit_database.known_units['cW'])
cWb = core.MagneticFlux(core.default_unit_database.known_units['cWb'])
candela = core.LuminousIntensity(core.default_unit_database.known_units['candela'])
ccd = core.LuminousIntensity(core.default_unit_database.known_units['ccd'])
cd = core.LuminousIntensity(core.default_unit_database.known_units['cd'])
centiampere = core.ElectricCurrent(core.default_unit_database.known_units['centiampere'])
centibecquerel = core.Frequency(core.default_unit_database.known_units['centibecquerel'])
centicandela = core.LuminousIntensity(core.default_unit_database.known_units['centicandela'])
centicoulomb = core.Charge(core.default_unit_database.known_units['centicoulomb'])
centifarad = core.Capacitance(core.default_unit_database.known_units['centifarad'])
centigauss = core.MagneticFluxDensity(core.default_unit_database.known_units['centigauss'])
centigram = core.Mass(core.default_unit_database.known_units['centigram'])
centihenry = core.Inductance(core.default_unit_database.known_units['centihenry'])
centihertz = core.Frequency(core.default_unit_database.known_units['centihertz'])
centijoule = core.Energy(core.default_unit_database.known_units['centijoule'])
centikelvin = core.Temperature(core.default_unit_database.known_units['centikelvin'])
centiliter = core.Volume(core.default_unit_database.known_units['centiliter'])
centilumen = core.LuminousFlux(core.default_unit_database.known_units['centilumen'])
centilux = core.Illuminance(core.default_unit_database.known_units['centilux'])
centimeter = core.Length(core.default_unit_database.known_units['centimeter'])
centimole = core.Quantity(core.default_unit_database.known_units['centimole'])
centinewton = core.Force(core.default_unit_database.known_units['centinewton'])
centiohm = core.Resistance(core.default_unit_database.known_units['centiohm'])
centipascal = core.Pressure(core.default_unit_database.known_units['centipascal'])
centiradian = core.Angle(core.default_unit_database.known_units['centiradian'])
centisecond = core.Time(core.default_unit_database.known_units['centisecond'])
centisiemens = core.ElectricalConductance(core.default_unit_database.known_units['centisiemens'])
centisteradian = core.Angle(core.default_unit_database.known_units['centisteradian'])
centitesla = core.MagneticFluxDensity(core.default_unit_database.known_units['centitesla'])
centivolt = core.ElectricPotential(core.default_unit_database.known_units['centivolt'])
centiwatt = core.Power(core.default_unit_database.known_units['centiwatt'])
centiweber = core.MagneticFlux(core.default_unit_database.known_units['centiweber'])
cg = core.Mass(core.default_unit_database.known_units['cg'])
cl = core.Volume(core.default_unit_database.known_units['cl'])
clm = core.LuminousFlux(core.default_unit_database.known_units['clm'])
clx = core.Illuminance(core.default_unit_database.known_units['clx'])
cm = core.Length(core.default_unit_database.known_units['cm'])
cmol = core.Quantity(core.default_unit_database.known_units['cmol'])
coulomb = core.Charge(core.default_unit_database.known_units['coulomb'])
crad = core.Angle(core.default_unit_database.known_units['crad'])
cs = core.Time(core.default_unit_database.known_units['cs'])
csr = core.Angle(core.default_unit_database.known_units['csr'])
cup = core.Volume(core.default_unit_database.known_units['cup'])
cyc = core.Angle(core.default_unit_database.known_units['cyc'])
cycle = core.Angle(core.default_unit_database.known_units['cycle'])
d = core.Time(core.default_unit_database.known_units['d'])
dA = core.ElectricCurrent(core.default_unit_database.known_units['dA'])
dB = core.LogPower(core.default_unit_database.known_units['dB'])
dBm = core.LogPower(core.default_unit_database.known_units['dBm'])
dBq = core.Frequency(core.default_unit_database.known_units['dBq'])
dC = core.Charge(core.default_unit_database.known_units['dC'])
dF = core.Capacitance(core.default_unit_database.known_units['dF'])
dGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['dGauss'])
dH = core.Inductance(core.default_unit_database.known_units['dH'])
dHz = core.Frequency(core.default_unit_database.known_units['dHz'])
dJ = core.Energy(core.default_unit_database.known_units['dJ'])
dK = core.Temperature(core.default_unit_database.known_units['dK'])
dN = core.Force(core.default_unit_database.known_units['dN'])
dOhm = core.Resistance(core.default_unit_database.known_units['dOhm'])
dPa = core.Pressure(core.default_unit_database.known_units['dPa'])
dS = core.ElectricalConductance(core.default_unit_database.known_units['dS'])
dT = core.MagneticFluxDensity(core.default_unit_database.known_units['dT'])
dV = core.ElectricPotential(core.default_unit_database.known_units['dV'])
dW = core.Power(core.default_unit_database.known_units['dW'])
dWb = core.MagneticFlux(core.default_unit_database.known_units['dWb'])
daA = core.ElectricCurrent(core.default_unit_database.known_units['daA'])
daBq = core.Frequency(core.default_unit_database.known_units['daBq'])
daC = core.Charge(core.default_unit_database.known_units['daC'])
daF = core.Capacitance(core.default_unit_database.known_units['daF'])
daGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['daGauss'])
daH = core.Inductance(core.default_unit_database.known_units['daH'])
daHz = core.Frequency(core.default_unit_database.known_units['daHz'])
daJ = core.Energy(core.default_unit_database.known_units['daJ'])
daK = core.Temperature(core.default_unit_database.known_units['daK'])
daN = core.Force(core.default_unit_database.known_units['daN'])
daOhm = core.Resistance(core.default_unit_database.known_units['daOhm'])
daPa = core.Pressure(core.default_unit_database.known_units['daPa'])
daS = core.ElectricalConductance(core.default_unit_database.known_units['daS'])
daT = core.MagneticFluxDensity(core.default_unit_database.known_units['daT'])
daV = core.ElectricPotential(core.default_unit_database.known_units['daV'])
daW = core.Power(core.default_unit_database.known_units['daW'])
daWb = core.MagneticFlux(core.default_unit_database.known_units['daWb'])
dacd = core.LuminousIntensity(core.default_unit_database.known_units['dacd'])
dag = core.Mass(core.default_unit_database.known_units['dag'])
dal = core.Volume(core.default_unit_database.known_units['dal'])
dalm = core.LuminousFlux(core.default_unit_database.known_units['dalm'])
dalx = core.Illuminance(core.default_unit_database.known_units['dalx'])
dam = core.Length(core.default_unit_database.known_units['dam'])
damol = core.Quantity(core.default_unit_database.known_units['damol'])
darad = core.Angle(core.default_unit_database.known_units['darad'])
das = core.Time(core.default_unit_database.known_units['das'])
dasr = core.Angle(core.default_unit_database.known_units['dasr'])
day = core.Time(core.default_unit_database.known_units['day'])
dcd = core.LuminousIntensity(core.default_unit_database.known_units['dcd'])
deciampere = core.ElectricCurrent(core.default_unit_database.known_units['deciampere'])
decibecquerel = core.Frequency(core.default_unit_database.known_units['decibecquerel'])
decibel = core.LogPower(core.default_unit_database.known_units['decibel'])
decicandela = core.LuminousIntensity(core.default_unit_database.known_units['decicandela'])
decicoulomb = core.Charge(core.default_unit_database.known_units['decicoulomb'])
decifarad = core.Capacitance(core.default_unit_database.known_units['decifarad'])
decigauss = core.MagneticFluxDensity(core.default_unit_database.known_units['decigauss'])
decigram = core.Mass(core.default_unit_database.known_units['decigram'])
decihenry = core.Inductance(core.default_unit_database.known_units['decihenry'])
decihertz = core.Frequency(core.default_unit_database.known_units['decihertz'])
decijoule = core.Energy(core.default_unit_database.known_units['decijoule'])
decikelvin = core.Temperature(core.default_unit_database.known_units['decikelvin'])
deciliter = core.Volume(core.default_unit_database.known_units['deciliter'])
decilumen = core.LuminousFlux(core.default_unit_database.known_units['decilumen'])
decilux = core.Illuminance(core.default_unit_database.known_units['decilux'])
decimeter = core.Length(core.default_unit_database.known_units['decimeter'])
decimole = core.Quantity(core.default_unit_database.known_units['decimole'])
decinewton = core.Force(core.default_unit_database.known_units['decinewton'])
deciohm = core.Resistance(core.default_unit_database.known_units['deciohm'])
decipascal = core.Pressure(core.default_unit_database.known_units['decipascal'])
deciradian = core.Angle(core.default_unit_database.known_units['deciradian'])
decisecond = core.Time(core.default_unit_database.known_units['decisecond'])
decisiemens = core.ElectricalConductance(core.default_unit_database.known_units['decisiemens'])
decisteradian = core.Angle(core.default_unit_database.known_units['decisteradian'])
decitesla = core.MagneticFluxDensity(core.default_unit_database.known_units['decitesla'])
decivolt = core.ElectricPotential(core.default_unit_database.known_units['decivolt'])
deciwatt = core.Power(core.default_unit_database.known_units['deciwatt'])
deciweber = core.MagneticFlux(core.default_unit_database.known_units['deciweber'])
deg = core.Angle(core.default_unit_database.known_units['deg'])
degC = core.Temperature(core.default_unit_database.known_units['degC'])
degF = core.Temperature(core.default_unit_database.known_units['degF'])
degR = core.Temperature(core.default_unit_database.known_units['degR'])
dekaampere = core.ElectricCurrent(core.default_unit_database.known_units['dekaampere'])
dekabecquerel = core.Frequency(core.default_unit_database.known_units['dekabecquerel'])
dekacandela = core.LuminousIntensity(core.default_unit_database.known_units['dekacandela'])
dekacoulomb = core.Charge(core.default_unit_database.known_units['dekacoulomb'])
dekafarad = core.Capacitance(core.default_unit_database.known_units['dekafarad'])
dekagauss = core.MagneticFluxDensity(core.default_unit_database.known_units['dekagauss'])
dekagram = core.Mass(core.default_unit_database.known_units['dekagram'])
dekahenry = core.Inductance(core.default_unit_database.known_units['dekahenry'])
dekahertz = core.Frequency(core.default_unit_database.known_units['dekahertz'])
dekajoule = core.Energy(core.default_unit_database.known_units['dekajoule'])
dekakelvin = core.Temperature(core.default_unit_database.known_units['dekakelvin'])
dekaliter = core.Volume(core.default_unit_database.known_units['dekaliter'])
dekalumen = core.LuminousFlux(core.default_unit_database.known_units['dekalumen'])
dekalux = core.Illuminance(core.default_unit_database.known_units['dekalux'])
dekameter = core.Length(core.default_unit_database.known_units['dekameter'])
dekamole = core.Quantity(core.default_unit_database.known_units['dekamole'])
dekanewton = core.Force(core.default_unit_database.known_units['dekanewton'])
dekaohm = core.Resistance(core.default_unit_database.known_units['dekaohm'])
dekapascal = core.Pressure(core.default_unit_database.known_units['dekapascal'])
dekaradian = core.Angle(core.default_unit_database.known_units['dekaradian'])
dekasecond = core.Time(core.default_unit_database.known_units['dekasecond'])
dekasiemens = core.ElectricalConductance(core.default_unit_database.known_units['dekasiemens'])
dekasteradian = core.Angle(core.default_unit_database.known_units['dekasteradian'])
dekatesla = core.MagneticFluxDensity(core.default_unit_database.known_units['dekatesla'])
dekavolt = core.ElectricPotential(core.default_unit_database.known_units['dekavolt'])
dekawatt = core.Power(core.default_unit_database.known_units['dekawatt'])
dekaweber = core.MagneticFlux(core.default_unit_database.known_units['dekaweber'])
dg = core.Mass(core.default_unit_database.known_units['dg'])
dl = core.Volume(core.default_unit_database.known_units['dl'])
dlm = core.LuminousFlux(core.default_unit_database.known_units['dlm'])
dlx = core.Illuminance(core.default_unit_database.known_units['dlx'])
dm = core.Length(core.default_unit_database.known_units['dm'])
dmol = core.Quantity(core.default_unit_database.known_units['dmol'])
drad = core.Angle(core.default_unit_database.known_units['drad'])
ds = core.Time(core.default_unit_database.known_units['ds'])
dsr = core.Angle(core.default_unit_database.known_units['dsr'])
e = core.Charge(core.default_unit_database.known_units['e'])
electron_mass = core.Mass(core.default_unit_database.known_units['electron_mass'])
elementary_charge = core.Charge(core.default_unit_database.known_units['elementary_charge'])
eV = core.Energy(core.default_unit_database.known_units['eV'])
eps0 = core.default_unit_database.known_units['eps0']
exaampere = core.ElectricCurrent(core.default_unit_database.known_units['exaampere'])
exabecquerel = core.Frequency(core.default_unit_database.known_units['exabecquerel'])
exacandela = core.LuminousIntensity(core.default_unit_database.known_units['exacandela'])
exacoulomb = core.Charge(core.default_unit_database.known_units['exacoulomb'])
exafarad = core.Capacitance(core.default_unit_database.known_units['exafarad'])
exagauss = core.MagneticFluxDensity(core.default_unit_database.known_units['exagauss'])
exagram = core.Mass(core.default_unit_database.known_units['exagram'])
exahenry = core.Inductance(core.default_unit_database.known_units['exahenry'])
exahertz = core.Frequency(core.default_unit_database.known_units['exahertz'])
exajoule = core.Energy(core.default_unit_database.known_units['exajoule'])
exakelvin = core.Temperature(core.default_unit_database.known_units['exakelvin'])
exaliter = core.Volume(core.default_unit_database.known_units['exaliter'])
exalumen = core.LuminousFlux(core.default_unit_database.known_units['exalumen'])
exalux = core.Illuminance(core.default_unit_database.known_units['exalux'])
exameter = core.Length(core.default_unit_database.known_units['exameter'])
examole = core.Quantity(core.default_unit_database.known_units['examole'])
exanewton = core.Force(core.default_unit_database.known_units['exanewton'])
exaohm = core.Resistance(core.default_unit_database.known_units['exaohm'])
exapascal = core.Pressure(core.default_unit_database.known_units['exapascal'])
exaradian = core.Angle(core.default_unit_database.known_units['exaradian'])
exasecond = core.Time(core.default_unit_database.known_units['exasecond'])
exasiemens = core.ElectricalConductance(core.default_unit_database.known_units['exasiemens'])
exasteradian = core.Angle(core.default_unit_database.known_units['exasteradian'])
exatesla = core.MagneticFluxDensity(core.default_unit_database.known_units['exatesla'])
exavolt = core.ElectricPotential(core.default_unit_database.known_units['exavolt'])
exawatt = core.Power(core.default_unit_database.known_units['exawatt'])
exaweber = core.MagneticFlux(core.default_unit_database.known_units['exaweber'])
fA = core.ElectricCurrent(core.default_unit_database.known_units['fA'])
fBq = core.Frequency(core.default_unit_database.known_units['fBq'])
fC = core.Charge(core.default_unit_database.known_units['fC'])
fF = core.Capacitance(core.default_unit_database.known_units['fF'])
fGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['fGauss'])
fH = core.Inductance(core.default_unit_database.known_units['fH'])
fHz = core.Frequency(core.default_unit_database.known_units['fHz'])
fJ = core.Energy(core.default_unit_database.known_units['fJ'])
fK = core.Temperature(core.default_unit_database.known_units['fK'])
fN = core.Force(core.default_unit_database.known_units['fN'])
fOhm = core.Resistance(core.default_unit_database.known_units['fOhm'])
fPa = core.Pressure(core.default_unit_database.known_units['fPa'])
fS = core.ElectricalConductance(core.default_unit_database.known_units['fS'])
fT = core.MagneticFluxDensity(core.default_unit_database.known_units['fT'])
fV = core.ElectricPotential(core.default_unit_database.known_units['fV'])
fW = core.Power(core.default_unit_database.known_units['fW'])
fWb = core.MagneticFlux(core.default_unit_database.known_units['fWb'])
farad = core.Capacitance(core.default_unit_database.known_units['farad'])
fcd = core.LuminousIntensity(core.default_unit_database.known_units['fcd'])
femtoampere = core.ElectricCurrent(core.default_unit_database.known_units['femtoampere'])
femtobecquerel = core.Frequency(core.default_unit_database.known_units['femtobecquerel'])
femtocandela = core.LuminousIntensity(core.default_unit_database.known_units['femtocandela'])
femtocoulomb = core.Charge(core.default_unit_database.known_units['femtocoulomb'])
femtofarad = core.Capacitance(core.default_unit_database.known_units['femtofarad'])
femtogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['femtogauss'])
femtogram = core.Mass(core.default_unit_database.known_units['femtogram'])
femtohenry = core.Inductance(core.default_unit_database.known_units['femtohenry'])
femtohertz = core.Frequency(core.default_unit_database.known_units['femtohertz'])
femtojoule = core.Energy(core.default_unit_database.known_units['femtojoule'])
femtokelvin = core.Temperature(core.default_unit_database.known_units['femtokelvin'])
femtoliter = core.Volume(core.default_unit_database.known_units['femtoliter'])
femtolumen = core.LuminousFlux(core.default_unit_database.known_units['femtolumen'])
femtolux = core.Illuminance(core.default_unit_database.known_units['femtolux'])
femtometer = core.Length(core.default_unit_database.known_units['femtometer'])
femtomole = core.Quantity(core.default_unit_database.known_units['femtomole'])
femtonewton = core.Force(core.default_unit_database.known_units['femtonewton'])
femtoohm = core.Resistance(core.default_unit_database.known_units['femtoohm'])
femtopascal = core.Pressure(core.default_unit_database.known_units['femtopascal'])
femtoradian = core.Angle(core.default_unit_database.known_units['femtoradian'])
femtosecond = core.Time(core.default_unit_database.known_units['femtosecond'])
femtosiemens = core.ElectricalConductance(core.default_unit_database.known_units['femtosiemens'])
femtosteradian = core.Angle(core.default_unit_database.known_units['femtosteradian'])
femtotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['femtotesla'])
femtovolt = core.ElectricPotential(core.default_unit_database.known_units['femtovolt'])
femtowatt = core.Power(core.default_unit_database.known_units['femtowatt'])
femtoweber = core.MagneticFlux(core.default_unit_database.known_units['femtoweber'])
fg = core.Mass(core.default_unit_database.known_units['fg'])
fl = core.Volume(core.default_unit_database.known_units['fl'])
flm = core.LuminousFlux(core.default_unit_database.known_units['flm'])
floz = core.Volume(core.default_unit_database.known_units['floz'])
fluid_ounce = core.Volume(core.default_unit_database.known_units['fluid_ounce'])
flx = core.Illuminance(core.default_unit_database.known_units['flx'])
fm = core.Length(core.default_unit_database.known_units['fm'])
fmol = core.Quantity(core.default_unit_database.known_units['fmol'])
foot = core.Length(core.default_unit_database.known_units['foot'])
frad = core.Angle(core.default_unit_database.known_units['frad'])
fs = core.Time(core.default_unit_database.known_units['fs'])
fsr = core.Angle(core.default_unit_database.known_units['fsr'])
ft = core.Length(core.default_unit_database.known_units['ft'])
g = core.Mass(core.default_unit_database.known_units['g'])
galUK = core.Volume(core.default_unit_database.known_units['galUK'])
galUS = core.Volume(core.default_unit_database.known_units['galUS'])
gauss = core.MagneticFluxDensity(core.default_unit_database.known_units['gauss'])
gigaampere = core.ElectricCurrent(core.default_unit_database.known_units['gigaampere'])
gigabecquerel = core.Frequency(core.default_unit_database.known_units['gigabecquerel'])
gigacandela = core.LuminousIntensity(core.default_unit_database.known_units['gigacandela'])
gigacoulomb = core.Charge(core.default_unit_database.known_units['gigacoulomb'])
gigafarad = core.Capacitance(core.default_unit_database.known_units['gigafarad'])
gigagauss = core.MagneticFluxDensity(core.default_unit_database.known_units['gigagauss'])
gigagram = core.Mass(core.default_unit_database.known_units['gigagram'])
gigahenry = core.Inductance(core.default_unit_database.known_units['gigahenry'])
gigahertz = core.Frequency(core.default_unit_database.known_units['gigahertz'])
gigajoule = core.Energy(core.default_unit_database.known_units['gigajoule'])
gigakelvin = core.Temperature(core.default_unit_database.known_units['gigakelvin'])
gigaliter = core.Volume(core.default_unit_database.known_units['gigaliter'])
gigalumen = core.LuminousFlux(core.default_unit_database.known_units['gigalumen'])
gigalux = core.Illuminance(core.default_unit_database.known_units['gigalux'])
gigameter = core.Length(core.default_unit_database.known_units['gigameter'])
gigamole = core.Quantity(core.default_unit_database.known_units['gigamole'])
giganewton = core.Force(core.default_unit_database.known_units['giganewton'])
gigaohm = core.Resistance(core.default_unit_database.known_units['gigaohm'])
gigapascal = core.Pressure(core.default_unit_database.known_units['gigapascal'])
gigaradian = core.Angle(core.default_unit_database.known_units['gigaradian'])
gigasecond = core.Time(core.default_unit_database.known_units['gigasecond'])
gigasiemens = core.ElectricalConductance(core.default_unit_database.known_units['gigasiemens'])
gigasteradian = core.Angle(core.default_unit_database.known_units['gigasteradian'])
gigatesla = core.MagneticFluxDensity(core.default_unit_database.known_units['gigatesla'])
gigavolt = core.ElectricPotential(core.default_unit_database.known_units['gigavolt'])
gigawatt = core.Power(core.default_unit_database.known_units['gigawatt'])
gigaweber = core.MagneticFlux(core.default_unit_database.known_units['gigaweber'])
gram = core.Mass(core.default_unit_database.known_units['gram'])
gravitational_constant = core.default_unit_database.known_units['gravitational_constant']
h = core.Time(core.default_unit_database.known_units['h'])
hA = core.ElectricCurrent(core.default_unit_database.known_units['hA'])
hBq = core.Frequency(core.default_unit_database.known_units['hBq'])
hC = core.Charge(core.default_unit_database.known_units['hC'])
hF = core.Capacitance(core.default_unit_database.known_units['hF'])
hGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['hGauss'])
hH = core.Inductance(core.default_unit_database.known_units['hH'])
hHz = core.Frequency(core.default_unit_database.known_units['hHz'])
hJ = core.Energy(core.default_unit_database.known_units['hJ'])
hK = core.Temperature(core.default_unit_database.known_units['hK'])
hN = core.Force(core.default_unit_database.known_units['hN'])
hOhm = core.Resistance(core.default_unit_database.known_units['hOhm'])
hPa = core.Pressure(core.default_unit_database.known_units['hPa'])
hS = core.ElectricalConductance(core.default_unit_database.known_units['hS'])
hT = core.MagneticFluxDensity(core.default_unit_database.known_units['hT'])
hV = core.ElectricPotential(core.default_unit_database.known_units['hV'])
hW = core.Power(core.default_unit_database.known_units['hW'])
hWb = core.MagneticFlux(core.default_unit_database.known_units['hWb'])
ha = core.Area(core.default_unit_database.known_units['ha'])
hbar = core.default_unit_database.known_units['hbar']
hcd = core.LuminousIntensity(core.default_unit_database.known_units['hcd'])
hectare = core.Area(core.default_unit_database.known_units['hectare'])
hectoampere = core.ElectricCurrent(core.default_unit_database.known_units['hectoampere'])
hectobecquerel = core.Frequency(core.default_unit_database.known_units['hectobecquerel'])
hectocandela = core.LuminousIntensity(core.default_unit_database.known_units['hectocandela'])
hectocoulomb = core.Charge(core.default_unit_database.known_units['hectocoulomb'])
hectofarad = core.Capacitance(core.default_unit_database.known_units['hectofarad'])
hectogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['hectogauss'])
hectogram = core.Mass(core.default_unit_database.known_units['hectogram'])
hectohenry = core.Inductance(core.default_unit_database.known_units['hectohenry'])
hectohertz = core.Frequency(core.default_unit_database.known_units['hectohertz'])
hectojoule = core.Energy(core.default_unit_database.known_units['hectojoule'])
hectokelvin = core.Temperature(core.default_unit_database.known_units['hectokelvin'])
hectoliter = core.Volume(core.default_unit_database.known_units['hectoliter'])
hectolumen = core.LuminousFlux(core.default_unit_database.known_units['hectolumen'])
hectolux = core.Illuminance(core.default_unit_database.known_units['hectolux'])
hectometer = core.Length(core.default_unit_database.known_units['hectometer'])
hectomole = core.Quantity(core.default_unit_database.known_units['hectomole'])
hectonewton = core.Force(core.default_unit_database.known_units['hectonewton'])
hectoohm = core.Resistance(core.default_unit_database.known_units['hectoohm'])
hectopascal = core.Pressure(core.default_unit_database.known_units['hectopascal'])
hectoradian = core.Angle(core.default_unit_database.known_units['hectoradian'])
hectosecond = core.Time(core.default_unit_database.known_units['hectosecond'])
hectosiemens = core.ElectricalConductance(core.default_unit_database.known_units['hectosiemens'])
hectosteradian = core.Angle(core.default_unit_database.known_units['hectosteradian'])
hectotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['hectotesla'])
hectovolt = core.ElectricPotential(core.default_unit_database.known_units['hectovolt'])
hectowatt = core.Power(core.default_unit_database.known_units['hectowatt'])
hectoweber = core.MagneticFlux(core.default_unit_database.known_units['hectoweber'])
henry = core.Inductance(core.default_unit_database.known_units['henry'])
hertz = core.Frequency(core.default_unit_database.known_units['hertz'])
hg = core.Mass(core.default_unit_database.known_units['hg'])
hl = core.Volume(core.default_unit_database.known_units['hl'])
hlm = core.LuminousFlux(core.default_unit_database.known_units['hlm'])
hlx = core.Illuminance(core.default_unit_database.known_units['hlx'])
hm = core.Length(core.default_unit_database.known_units['hm'])
hmol = core.Quantity(core.default_unit_database.known_units['hmol'])
hour = core.Time(core.default_unit_database.known_units['hour'])
hplanck = core.default_unit_database.known_units['hplanck']
hrad = core.Angle(core.default_unit_database.known_units['hrad'])
hs = core.Time(core.default_unit_database.known_units['hs'])
hsr = core.Angle(core.default_unit_database.known_units['hsr'])
inch = core.Length(core.default_unit_database.known_units['inch'])
joule = core.Energy(core.default_unit_database.known_units['joule'])
k = core.default_unit_database.known_units['k']
kA = core.ElectricCurrent(core.default_unit_database.known_units['kA'])
kBq = core.Frequency(core.default_unit_database.known_units['kBq'])
kC = core.Charge(core.default_unit_database.known_units['kC'])
kF = core.Capacitance(core.default_unit_database.known_units['kF'])
kGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['kGauss'])
kH = core.Inductance(core.default_unit_database.known_units['kH'])
kHz = core.Frequency(core.default_unit_database.known_units['kHz'])
kJ = core.Energy(core.default_unit_database.known_units['kJ'])
kK = core.Temperature(core.default_unit_database.known_units['kK'])
kN = core.Force(core.default_unit_database.known_units['kN'])
kOhm = core.Resistance(core.default_unit_database.known_units['kOhm'])
kPa = core.Pressure(core.default_unit_database.known_units['kPa'])
kS = core.ElectricalConductance(core.default_unit_database.known_units['kS'])
kT = core.MagneticFluxDensity(core.default_unit_database.known_units['kT'])
kV = core.ElectricPotential(core.default_unit_database.known_units['kV'])
kW = core.Power(core.default_unit_database.known_units['kW'])
kWb = core.MagneticFlux(core.default_unit_database.known_units['kWb'])
kcd = core.LuminousIntensity(core.default_unit_database.known_units['kcd'])
kelvin = core.Temperature(core.default_unit_database.known_units['kelvin'])
kg = core.Mass(core.default_unit_database.known_units['kg'])
kiloampere = core.ElectricCurrent(core.default_unit_database.known_units['kiloampere'])
kilobecquerel = core.Frequency(core.default_unit_database.known_units['kilobecquerel'])
kilocandela = core.LuminousIntensity(core.default_unit_database.known_units['kilocandela'])
kilocoulomb = core.Charge(core.default_unit_database.known_units['kilocoulomb'])
kilofarad = core.Capacitance(core.default_unit_database.known_units['kilofarad'])
kilogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['kilogauss'])
kilogram = core.Mass(core.default_unit_database.known_units['kilogram'])
kilohenry = core.Inductance(core.default_unit_database.known_units['kilohenry'])
kilohertz = core.Frequency(core.default_unit_database.known_units['kilohertz'])
kilojoule = core.Energy(core.default_unit_database.known_units['kilojoule'])
kilokelvin = core.Temperature(core.default_unit_database.known_units['kilokelvin'])
kiloliter = core.Volume(core.default_unit_database.known_units['kiloliter'])
kilolumen = core.LuminousFlux(core.default_unit_database.known_units['kilolumen'])
kilolux = core.Illuminance(core.default_unit_database.known_units['kilolux'])
kilometer = core.Length(core.default_unit_database.known_units['kilometer'])
kilomole = core.Quantity(core.default_unit_database.known_units['kilomole'])
kilonewton = core.Force(core.default_unit_database.known_units['kilonewton'])
kiloohm = core.Resistance(core.default_unit_database.known_units['kiloohm'])
kilopascal = core.Pressure(core.default_unit_database.known_units['kilopascal'])
kiloradian = core.Angle(core.default_unit_database.known_units['kiloradian'])
kilosecond = core.Time(core.default_unit_database.known_units['kilosecond'])
kilosiemens = core.ElectricalConductance(core.default_unit_database.known_units['kilosiemens'])
kilosteradian = core.Angle(core.default_unit_database.known_units['kilosteradian'])
kilotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['kilotesla'])
kilovolt = core.ElectricPotential(core.default_unit_database.known_units['kilovolt'])
kilowatt = core.Power(core.default_unit_database.known_units['kilowatt'])
kiloweber = core.MagneticFlux(core.default_unit_database.known_units['kiloweber'])
kl = core.Volume(core.default_unit_database.known_units['kl'])
klm = core.LuminousFlux(core.default_unit_database.known_units['klm'])
klx = core.Illuminance(core.default_unit_database.known_units['klx'])
km = core.Length(core.default_unit_database.known_units['km'])
kmol = core.Quantity(core.default_unit_database.known_units['kmol'])
krad = core.Angle(core.default_unit_database.known_units['krad'])
ks = core.Time(core.default_unit_database.known_units['ks'])
ksr = core.Angle(core.default_unit_database.known_units['ksr'])
l = core.Volume(core.default_unit_database.known_units['l'])
lb = core.Mass(core.default_unit_database.known_units['lb'])
light_year = core.Length(core.default_unit_database.known_units['light_year'])
liter = core.Volume(core.default_unit_database.known_units['liter'])
lm = core.LuminousFlux(core.default_unit_database.known_units['lm'])
lumen = core.LuminousFlux(core.default_unit_database.known_units['lumen'])
lux = core.Illuminance(core.default_unit_database.known_units['lux'])
lx = core.Illuminance(core.default_unit_database.known_units['lx'])
ly = core.Length(core.default_unit_database.known_units['ly'])
lyr = core.Length(core.default_unit_database.known_units['lyr'])
m = core.Length(core.default_unit_database.known_units['m'])
mA = core.ElectricCurrent(core.default_unit_database.known_units['mA'])
mBq = core.Frequency(core.default_unit_database.known_units['mBq'])
mC = core.Charge(core.default_unit_database.known_units['mC'])
mF = core.Capacitance(core.default_unit_database.known_units['mF'])
mGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['mGauss'])
mH = core.Inductance(core.default_unit_database.known_units['mH'])
mHz = core.Frequency(core.default_unit_database.known_units['mHz'])
mJ = core.Energy(core.default_unit_database.known_units['mJ'])
mK = core.Temperature(core.default_unit_database.known_units['mK'])
mN = core.Force(core.default_unit_database.known_units['mN'])
mOhm = core.Resistance(core.default_unit_database.known_units['mOhm'])
mPa = core.Pressure(core.default_unit_database.known_units['mPa'])
mS = core.ElectricalConductance(core.default_unit_database.known_units['mS'])
mT = core.MagneticFluxDensity(core.default_unit_database.known_units['mT'])
mV = core.ElectricPotential(core.default_unit_database.known_units['mV'])
mW = core.Power(core.default_unit_database.known_units['mW'])
mWb = core.MagneticFlux(core.default_unit_database.known_units['mWb'])
mcd = core.LuminousIntensity(core.default_unit_database.known_units['mcd'])
me = core.Mass(core.default_unit_database.known_units['me'])
megaampere = core.ElectricCurrent(core.default_unit_database.known_units['megaampere'])
megabecquerel = core.Frequency(core.default_unit_database.known_units['megabecquerel'])
megacandela = core.LuminousIntensity(core.default_unit_database.known_units['megacandela'])
megacoulomb = core.Charge(core.default_unit_database.known_units['megacoulomb'])
megafarad = core.Capacitance(core.default_unit_database.known_units['megafarad'])
megagauss = core.MagneticFluxDensity(core.default_unit_database.known_units['megagauss'])
megagram = core.Mass(core.default_unit_database.known_units['megagram'])
megahenry = core.Inductance(core.default_unit_database.known_units['megahenry'])
megahertz = core.Frequency(core.default_unit_database.known_units['megahertz'])
megajoule = core.Energy(core.default_unit_database.known_units['megajoule'])
megakelvin = core.Temperature(core.default_unit_database.known_units['megakelvin'])
megaliter = core.Volume(core.default_unit_database.known_units['megaliter'])
megalumen = core.LuminousFlux(core.default_unit_database.known_units['megalumen'])
megalux = core.Illuminance(core.default_unit_database.known_units['megalux'])
megameter = core.Length(core.default_unit_database.known_units['megameter'])
megamole = core.Quantity(core.default_unit_database.known_units['megamole'])
meganewton = core.Force(core.default_unit_database.known_units['meganewton'])
megaohm = core.Resistance(core.default_unit_database.known_units['megaohm'])
megapascal = core.Pressure(core.default_unit_database.known_units['megapascal'])
megaradian = core.Angle(core.default_unit_database.known_units['megaradian'])
megasecond = core.Time(core.default_unit_database.known_units['megasecond'])
megasiemens = core.ElectricalConductance(core.default_unit_database.known_units['megasiemens'])
megasteradian = core.Angle(core.default_unit_database.known_units['megasteradian'])
megatesla = core.MagneticFluxDensity(core.default_unit_database.known_units['megatesla'])
megavolt = core.ElectricPotential(core.default_unit_database.known_units['megavolt'])
megawatt = core.Power(core.default_unit_database.known_units['megawatt'])
megaweber = core.MagneticFlux(core.default_unit_database.known_units['megaweber'])
meter = core.Length(core.default_unit_database.known_units['meter'])
mg = core.Mass(core.default_unit_database.known_units['mg'])
microampere = core.ElectricCurrent(core.default_unit_database.known_units['microampere'])
microbecquerel = core.Frequency(core.default_unit_database.known_units['microbecquerel'])
microcandela = core.LuminousIntensity(core.default_unit_database.known_units['microcandela'])
microcoulomb = core.Charge(core.default_unit_database.known_units['microcoulomb'])
microfarad = core.Capacitance(core.default_unit_database.known_units['microfarad'])
microgauss = core.MagneticFluxDensity(core.default_unit_database.known_units['microgauss'])
microgram = core.Mass(core.default_unit_database.known_units['microgram'])
microhenry = core.Inductance(core.default_unit_database.known_units['microhenry'])
microhertz = core.Frequency(core.default_unit_database.known_units['microhertz'])
microjoule = core.Energy(core.default_unit_database.known_units['microjoule'])
microkelvin = core.Temperature(core.default_unit_database.known_units['microkelvin'])
microliter = core.Volume(core.default_unit_database.known_units['microliter'])
microlumen = core.LuminousFlux(core.default_unit_database.known_units['microlumen'])
microlux = core.Illuminance(core.default_unit_database.known_units['microlux'])
micrometer = core.Length(core.default_unit_database.known_units['micrometer'])
micromole = core.Quantity(core.default_unit_database.known_units['micromole'])
micronewton = core.Force(core.default_unit_database.known_units['micronewton'])
microohm = core.Resistance(core.default_unit_database.known_units['microohm'])
micropascal = core.Pressure(core.default_unit_database.known_units['micropascal'])
microradian = core.Angle(core.default_unit_database.known_units['microradian'])
microsecond = core.Time(core.default_unit_database.known_units['microsecond'])
microsiemens = core.ElectricalConductance(core.default_unit_database.known_units['microsiemens'])
microsteradian = core.Angle(core.default_unit_database.known_units['microsteradian'])
microtesla = core.MagneticFluxDensity(core.default_unit_database.known_units['microtesla'])
microvolt = core.ElectricPotential(core.default_unit_database.known_units['microvolt'])
microwatt = core.Power(core.default_unit_database.known_units['microwatt'])
microweber = core.MagneticFlux(core.default_unit_database.known_units['microweber'])
milliampere = core.ElectricCurrent(core.default_unit_database.known_units['milliampere'])
millibecquerel = core.Frequency(core.default_unit_database.known_units['millibecquerel'])
millicandela = core.LuminousIntensity(core.default_unit_database.known_units['millicandela'])
millicoulomb = core.Charge(core.default_unit_database.known_units['millicoulomb'])
millifarad = core.Capacitance(core.default_unit_database.known_units['millifarad'])
milligauss = core.MagneticFluxDensity(core.default_unit_database.known_units['milligauss'])
milligram = core.Mass(core.default_unit_database.known_units['milligram'])
millihenry = core.Inductance(core.default_unit_database.known_units['millihenry'])
millihertz = core.Frequency(core.default_unit_database.known_units['millihertz'])
millijoule = core.Energy(core.default_unit_database.known_units['millijoule'])
millikelvin = core.Temperature(core.default_unit_database.known_units['millikelvin'])
milliliter = core.Volume(core.default_unit_database.known_units['milliliter'])
millilumen = core.LuminousFlux(core.default_unit_database.known_units['millilumen'])
millilux = core.Illuminance(core.default_unit_database.known_units['millilux'])
millimeter = core.Length(core.default_unit_database.known_units['millimeter'])
millimole = core.Quantity(core.default_unit_database.known_units['millimole'])
millinewton = core.Force(core.default_unit_database.known_units['millinewton'])
milliohm = core.Resistance(core.default_unit_database.known_units['milliohm'])
millipascal = core.Pressure(core.default_unit_database.known_units['millipascal'])
milliradian = core.Angle(core.default_unit_database.known_units['milliradian'])
millisecond = core.Time(core.default_unit_database.known_units['millisecond'])
millisiemens = core.ElectricalConductance(core.default_unit_database.known_units['millisiemens'])
millisteradian = core.Angle(core.default_unit_database.known_units['millisteradian'])
millitesla = core.MagneticFluxDensity(core.default_unit_database.known_units['millitesla'])
millivolt = core.ElectricPotential(core.default_unit_database.known_units['millivolt'])
milliwatt = core.Power(core.default_unit_database.known_units['milliwatt'])
milliweber = core.MagneticFlux(core.default_unit_database.known_units['milliweber'])
minute = core.Time(core.default_unit_database.known_units['minute'])
ml = core.Volume(core.default_unit_database.known_units['ml'])
mlm = core.LuminousFlux(core.default_unit_database.known_units['mlm'])
mlx = core.Illuminance(core.default_unit_database.known_units['mlx'])
mm = core.Length(core.default_unit_database.known_units['mm'])
mmol = core.Quantity(core.default_unit_database.known_units['mmol'])
mol = core.Quantity(core.default_unit_database.known_units['mol'])
mole = core.Quantity(core.default_unit_database.known_units['mole'])
mp = core.Mass(core.default_unit_database.known_units['mp'])
mrad = core.Angle(core.default_unit_database.known_units['mrad'])
ms = core.Time(core.default_unit_database.known_units['ms'])
msr = core.Angle(core.default_unit_database.known_units['msr'])
mu0 = core.default_unit_database.known_units['mu0']
nA = core.ElectricCurrent(core.default_unit_database.known_units['nA'])
nBq = core.Frequency(core.default_unit_database.known_units['nBq'])
nC = core.Charge(core.default_unit_database.known_units['nC'])
nF = core.Capacitance(core.default_unit_database.known_units['nF'])
nGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['nGauss'])
nH = core.Inductance(core.default_unit_database.known_units['nH'])
nHz = core.Frequency(core.default_unit_database.known_units['nHz'])
nJ = core.Energy(core.default_unit_database.known_units['nJ'])
nK = core.Temperature(core.default_unit_database.known_units['nK'])
nN = core.Force(core.default_unit_database.known_units['nN'])
nOhm = core.Resistance(core.default_unit_database.known_units['nOhm'])
nPa = core.Pressure(core.default_unit_database.known_units['nPa'])
nS = core.ElectricalConductance(core.default_unit_database.known_units['nS'])
nT = core.MagneticFluxDensity(core.default_unit_database.known_units['nT'])
nV = core.ElectricPotential(core.default_unit_database.known_units['nV'])
nW = core.Power(core.default_unit_database.known_units['nW'])
nWb = core.MagneticFlux(core.default_unit_database.known_units['nWb'])
nanoampere = core.ElectricCurrent(core.default_unit_database.known_units['nanoampere'])
nanobecquerel = core.Frequency(core.default_unit_database.known_units['nanobecquerel'])
nanocandela = core.LuminousIntensity(core.default_unit_database.known_units['nanocandela'])
nanocoulomb = core.Charge(core.default_unit_database.known_units['nanocoulomb'])
nanofarad = core.Capacitance(core.default_unit_database.known_units['nanofarad'])
nanogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['nanogauss'])
nanogram = core.Mass(core.default_unit_database.known_units['nanogram'])
nanohenry = core.Inductance(core.default_unit_database.known_units['nanohenry'])
nanohertz = core.Frequency(core.default_unit_database.known_units['nanohertz'])
nanojoule = core.Energy(core.default_unit_database.known_units['nanojoule'])
nanokelvin = core.Temperature(core.default_unit_database.known_units['nanokelvin'])
nanoliter = core.Volume(core.default_unit_database.known_units['nanoliter'])
nanolumen = core.LuminousFlux(core.default_unit_database.known_units['nanolumen'])
nanolux = core.Illuminance(core.default_unit_database.known_units['nanolux'])
nanometer = core.Length(core.default_unit_database.known_units['nanometer'])
nanomole = core.Quantity(core.default_unit_database.known_units['nanomole'])
nanonewton = core.Force(core.default_unit_database.known_units['nanonewton'])
nanoohm = core.Resistance(core.default_unit_database.known_units['nanoohm'])
nanopascal = core.Pressure(core.default_unit_database.known_units['nanopascal'])
nanoradian = core.Angle(core.default_unit_database.known_units['nanoradian'])
nanosecond = core.Time(core.default_unit_database.known_units['nanosecond'])
nanosiemens = core.ElectricalConductance(core.default_unit_database.known_units['nanosiemens'])
nanosteradian = core.Angle(core.default_unit_database.known_units['nanosteradian'])
nanotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['nanotesla'])
nanovolt = core.ElectricPotential(core.default_unit_database.known_units['nanovolt'])
nanowatt = core.Power(core.default_unit_database.known_units['nanowatt'])
nanoweber = core.MagneticFlux(core.default_unit_database.known_units['nanoweber'])
nautical_mile = core.Length(core.default_unit_database.known_units['nautical_mile'])
ncd = core.LuminousIntensity(core.default_unit_database.known_units['ncd'])
newton = core.Force(core.default_unit_database.known_units['newton'])
ng = core.Mass(core.default_unit_database.known_units['ng'])
nl = core.Volume(core.default_unit_database.known_units['nl'])
nlm = core.LuminousFlux(core.default_unit_database.known_units['nlm'])
nlx = core.Illuminance(core.default_unit_database.known_units['nlx'])
nm = core.Length(core.default_unit_database.known_units['nm'])
nmi = core.Length(core.default_unit_database.known_units['nmi'])
nmol = core.Quantity(core.default_unit_database.known_units['nmol'])
nrad = core.Angle(core.default_unit_database.known_units['nrad'])
ns = core.Time(core.default_unit_database.known_units['ns'])
nsr = core.Angle(core.default_unit_database.known_units['nsr'])
ohm = core.Resistance(core.default_unit_database.known_units['ohm'])
ounce = core.Mass(core.default_unit_database.known_units['ounce'])
oz = core.Mass(core.default_unit_database.known_units['oz'])
pA = core.ElectricCurrent(core.default_unit_database.known_units['pA'])
pBq = core.Frequency(core.default_unit_database.known_units['pBq'])
pC = core.Charge(core.default_unit_database.known_units['pC'])
pF = core.Capacitance(core.default_unit_database.known_units['pF'])
pGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['pGauss'])
pH = core.Inductance(core.default_unit_database.known_units['pH'])
pHz = core.Frequency(core.default_unit_database.known_units['pHz'])
pJ = core.Energy(core.default_unit_database.known_units['pJ'])
pK = core.Temperature(core.default_unit_database.known_units['pK'])
pN = core.Force(core.default_unit_database.known_units['pN'])
pOhm = core.Resistance(core.default_unit_database.known_units['pOhm'])
pPa = core.Pressure(core.default_unit_database.known_units['pPa'])
pS = core.ElectricalConductance(core.default_unit_database.known_units['pS'])
pT = core.MagneticFluxDensity(core.default_unit_database.known_units['pT'])
pV = core.ElectricPotential(core.default_unit_database.known_units['pV'])
pW = core.Power(core.default_unit_database.known_units['pW'])
pWb = core.MagneticFlux(core.default_unit_database.known_units['pWb'])
pascal = core.Pressure(core.default_unit_database.known_units['pascal'])
pcd = core.LuminousIntensity(core.default_unit_database.known_units['pcd'])
petaampere = core.ElectricCurrent(core.default_unit_database.known_units['petaampere'])
petabecquerel = core.Frequency(core.default_unit_database.known_units['petabecquerel'])
petacandela = core.LuminousIntensity(core.default_unit_database.known_units['petacandela'])
petacoulomb = core.Charge(core.default_unit_database.known_units['petacoulomb'])
petafarad = core.Capacitance(core.default_unit_database.known_units['petafarad'])
petagauss = core.MagneticFluxDensity(core.default_unit_database.known_units['petagauss'])
petagram = core.Mass(core.default_unit_database.known_units['petagram'])
petahenry = core.Inductance(core.default_unit_database.known_units['petahenry'])
petahertz = core.Frequency(core.default_unit_database.known_units['petahertz'])
petajoule = core.Energy(core.default_unit_database.known_units['petajoule'])
petakelvin = core.Temperature(core.default_unit_database.known_units['petakelvin'])
petaliter = core.Volume(core.default_unit_database.known_units['petaliter'])
petalumen = core.LuminousFlux(core.default_unit_database.known_units['petalumen'])
petalux = core.Illuminance(core.default_unit_database.known_units['petalux'])
petameter = core.Length(core.default_unit_database.known_units['petameter'])
petamole = core.Quantity(core.default_unit_database.known_units['petamole'])
petanewton = core.Force(core.default_unit_database.known_units['petanewton'])
petaohm = core.Resistance(core.default_unit_database.known_units['petaohm'])
petapascal = core.Pressure(core.default_unit_database.known_units['petapascal'])
petaradian = core.Angle(core.default_unit_database.known_units['petaradian'])
petasecond = core.Time(core.default_unit_database.known_units['petasecond'])
petasiemens = core.ElectricalConductance(core.default_unit_database.known_units['petasiemens'])
petasteradian = core.Angle(core.default_unit_database.known_units['petasteradian'])
petatesla = core.MagneticFluxDensity(core.default_unit_database.known_units['petatesla'])
petavolt = core.ElectricPotential(core.default_unit_database.known_units['petavolt'])
petawatt = core.Power(core.default_unit_database.known_units['petawatt'])
petaweber = core.MagneticFlux(core.default_unit_database.known_units['petaweber'])
pg = core.Mass(core.default_unit_database.known_units['pg'])
phi0 = core.MagneticFlux(core.default_unit_database.known_units['phi0'])
picoampere = core.ElectricCurrent(core.default_unit_database.known_units['picoampere'])
picobecquerel = core.Frequency(core.default_unit_database.known_units['picobecquerel'])
picocandela = core.LuminousIntensity(core.default_unit_database.known_units['picocandela'])
picocoulomb = core.Charge(core.default_unit_database.known_units['picocoulomb'])
picofarad = core.Capacitance(core.default_unit_database.known_units['picofarad'])
picogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['picogauss'])
picogram = core.Mass(core.default_unit_database.known_units['picogram'])
picohenry = core.Inductance(core.default_unit_database.known_units['picohenry'])
picohertz = core.Frequency(core.default_unit_database.known_units['picohertz'])
picojoule = core.Energy(core.default_unit_database.known_units['picojoule'])
picokelvin = core.Temperature(core.default_unit_database.known_units['picokelvin'])
picoliter = core.Volume(core.default_unit_database.known_units['picoliter'])
picolumen = core.LuminousFlux(core.default_unit_database.known_units['picolumen'])
picolux = core.Illuminance(core.default_unit_database.known_units['picolux'])
picometer = core.Length(core.default_unit_database.known_units['picometer'])
picomole = core.Quantity(core.default_unit_database.known_units['picomole'])
piconewton = core.Force(core.default_unit_database.known_units['piconewton'])
picoohm = core.Resistance(core.default_unit_database.known_units['picoohm'])
picopascal = core.Pressure(core.default_unit_database.known_units['picopascal'])
picoradian = core.Angle(core.default_unit_database.known_units['picoradian'])
picosecond = core.Time(core.default_unit_database.known_units['picosecond'])
picosiemens = core.ElectricalConductance(core.default_unit_database.known_units['picosiemens'])
picosteradian = core.Angle(core.default_unit_database.known_units['picosteradian'])
picotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['picotesla'])
picovolt = core.ElectricPotential(core.default_unit_database.known_units['picovolt'])
picowatt = core.Power(core.default_unit_database.known_units['picowatt'])
picoweber = core.MagneticFlux(core.default_unit_database.known_units['picoweber'])
pint = core.Volume(core.default_unit_database.known_units['pint'])
pl = core.Volume(core.default_unit_database.known_units['pl'])
planck_constant = core.default_unit_database.known_units['planck_constant']
plm = core.LuminousFlux(core.default_unit_database.known_units['plm'])
plx = core.Illuminance(core.default_unit_database.known_units['plx'])
pm = core.Length(core.default_unit_database.known_units['pm'])
pmol = core.Quantity(core.default_unit_database.known_units['pmol'])
pound = core.Mass(core.default_unit_database.known_units['pound'])
pounds_per_square_inch = core.Pressure(
    core.default_unit_database.known_units['pounds_per_square_inch']
)
prad = core.Angle(core.default_unit_database.known_units['prad'])
proton_mass = core.Mass(core.default_unit_database.known_units['proton_mass'])
ps = core.Time(core.default_unit_database.known_units['ps'])
psi = core.Pressure(core.default_unit_database.known_units['psi'])
psr = core.Angle(core.default_unit_database.known_units['psr'])
qt = core.Volume(core.default_unit_database.known_units['qt'])
quart = core.Volume(core.default_unit_database.known_units['quart'])
rad = core.Angle(core.default_unit_database.known_units['rad'])
radian = core.Angle(core.default_unit_database.known_units['radian'])
reduced_planck_constant = core.default_unit_database.known_units['reduced_planck_constant']
rootHz = core.default_unit_database.known_units['rootHz']
s = core.Time(core.default_unit_database.known_units['s'])
second = core.Time(core.default_unit_database.known_units['second'])
siemens = core.ElectricalConductance(core.default_unit_database.known_units['siemens'])
speed_of_light = core.Speed(core.default_unit_database.known_units['speed_of_light'])
sqrtHz = core.default_unit_database.known_units['sqrtHz']
sr = core.Angle(core.default_unit_database.known_units['sr'])
steradian = core.Angle(core.default_unit_database.known_units['steradian'])
tablespoon = core.Volume(core.default_unit_database.known_units['tablespoon'])
tbsp = core.Volume(core.default_unit_database.known_units['tbsp'])
teaspoon = core.Volume(core.default_unit_database.known_units['teaspoon'])
teraampere = core.ElectricCurrent(core.default_unit_database.known_units['teraampere'])
terabecquerel = core.Frequency(core.default_unit_database.known_units['terabecquerel'])
teracandela = core.LuminousIntensity(core.default_unit_database.known_units['teracandela'])
teracoulomb = core.Charge(core.default_unit_database.known_units['teracoulomb'])
terafarad = core.Capacitance(core.default_unit_database.known_units['terafarad'])
teragauss = core.MagneticFluxDensity(core.default_unit_database.known_units['teragauss'])
teragram = core.Mass(core.default_unit_database.known_units['teragram'])
terahenry = core.Inductance(core.default_unit_database.known_units['terahenry'])
terahertz = core.Frequency(core.default_unit_database.known_units['terahertz'])
terajoule = core.Energy(core.default_unit_database.known_units['terajoule'])
terakelvin = core.Temperature(core.default_unit_database.known_units['terakelvin'])
teraliter = core.Volume(core.default_unit_database.known_units['teraliter'])
teralumen = core.LuminousFlux(core.default_unit_database.known_units['teralumen'])
teralux = core.Illuminance(core.default_unit_database.known_units['teralux'])
terameter = core.Length(core.default_unit_database.known_units['terameter'])
teramole = core.Quantity(core.default_unit_database.known_units['teramole'])
teranewton = core.Force(core.default_unit_database.known_units['teranewton'])
teraohm = core.Resistance(core.default_unit_database.known_units['teraohm'])
terapascal = core.Pressure(core.default_unit_database.known_units['terapascal'])
teraradian = core.Angle(core.default_unit_database.known_units['teraradian'])
terasecond = core.Time(core.default_unit_database.known_units['terasecond'])
terasiemens = core.ElectricalConductance(core.default_unit_database.known_units['terasiemens'])
terasteradian = core.Angle(core.default_unit_database.known_units['terasteradian'])
teratesla = core.MagneticFluxDensity(core.default_unit_database.known_units['teratesla'])
teravolt = core.ElectricPotential(core.default_unit_database.known_units['teravolt'])
terawatt = core.Power(core.default_unit_database.known_units['terawatt'])
teraweber = core.MagneticFlux(core.default_unit_database.known_units['teraweber'])
tesla = core.MagneticFluxDensity(core.default_unit_database.known_units['tesla'])
ton = core.Mass(core.default_unit_database.known_units['ton'])
tsp = core.Volume(core.default_unit_database.known_units['tsp'])
uA = core.ElectricCurrent(core.default_unit_database.known_units['uA'])
uBq = core.Frequency(core.default_unit_database.known_units['uBq'])
uC = core.Charge(core.default_unit_database.known_units['uC'])
uF = core.Capacitance(core.default_unit_database.known_units['uF'])
uGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['uGauss'])
uH = core.Inductance(core.default_unit_database.known_units['uH'])
uHz = core.Frequency(core.default_unit_database.known_units['uHz'])
uJ = core.Energy(core.default_unit_database.known_units['uJ'])
uK = core.Temperature(core.default_unit_database.known_units['uK'])
uN = core.Force(core.default_unit_database.known_units['uN'])
uOhm = core.Resistance(core.default_unit_database.known_units['uOhm'])
uPa = core.Pressure(core.default_unit_database.known_units['uPa'])
uS = core.ElectricalConductance(core.default_unit_database.known_units['uS'])
uT = core.MagneticFluxDensity(core.default_unit_database.known_units['uT'])
uV = core.ElectricPotential(core.default_unit_database.known_units['uV'])
uW = core.Power(core.default_unit_database.known_units['uW'])
uWb = core.MagneticFlux(core.default_unit_database.known_units['uWb'])
ucd = core.LuminousIntensity(core.default_unit_database.known_units['ucd'])
ug = core.Mass(core.default_unit_database.known_units['ug'])
ul = core.Volume(core.default_unit_database.known_units['ul'])
ulm = core.LuminousFlux(core.default_unit_database.known_units['ulm'])
ulx = core.Illuminance(core.default_unit_database.known_units['ulx'])
um = core.Length(core.default_unit_database.known_units['um'])
umol = core.Quantity(core.default_unit_database.known_units['umol'])
urad = core.Angle(core.default_unit_database.known_units['urad'])
us = core.Time(core.default_unit_database.known_units['us'])
us_gallon = core.Volume(core.default_unit_database.known_units['us_gallon'])
usr = core.Angle(core.default_unit_database.known_units['usr'])
vacuum_permeability = core.default_unit_database.known_units['vacuum_permeability']
vacuum_permittivity = core.default_unit_database.known_units['vacuum_permittivity']
volt = core.ElectricPotential(core.default_unit_database.known_units['volt'])
watt = core.Power(core.default_unit_database.known_units['watt'])
weber = core.MagneticFlux(core.default_unit_database.known_units['weber'])
week = core.Time(core.default_unit_database.known_units['week'])
wk = core.Time(core.default_unit_database.known_units['wk'])
yA = core.ElectricCurrent(core.default_unit_database.known_units['yA'])
yBq = core.Frequency(core.default_unit_database.known_units['yBq'])
yC = core.Charge(core.default_unit_database.known_units['yC'])
yF = core.Capacitance(core.default_unit_database.known_units['yF'])
yGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['yGauss'])
yH = core.Inductance(core.default_unit_database.known_units['yH'])
yHz = core.Frequency(core.default_unit_database.known_units['yHz'])
yJ = core.Energy(core.default_unit_database.known_units['yJ'])
yK = core.Temperature(core.default_unit_database.known_units['yK'])
yN = core.Force(core.default_unit_database.known_units['yN'])
yOhm = core.Resistance(core.default_unit_database.known_units['yOhm'])
yPa = core.Pressure(core.default_unit_database.known_units['yPa'])
yS = core.ElectricalConductance(core.default_unit_database.known_units['yS'])
yT = core.MagneticFluxDensity(core.default_unit_database.known_units['yT'])
yV = core.ElectricPotential(core.default_unit_database.known_units['yV'])
yW = core.Power(core.default_unit_database.known_units['yW'])
yWb = core.MagneticFlux(core.default_unit_database.known_units['yWb'])
yard = core.Length(core.default_unit_database.known_units['yard'])
ycd = core.LuminousIntensity(core.default_unit_database.known_units['ycd'])
yd = core.Length(core.default_unit_database.known_units['yd'])
year = core.Time(core.default_unit_database.known_units['year'])
yg = core.Mass(core.default_unit_database.known_units['yg'])
yl = core.Volume(core.default_unit_database.known_units['yl'])
ylm = core.LuminousFlux(core.default_unit_database.known_units['ylm'])
ylx = core.Illuminance(core.default_unit_database.known_units['ylx'])
ym = core.Length(core.default_unit_database.known_units['ym'])
ymol = core.Quantity(core.default_unit_database.known_units['ymol'])
yoctoampere = core.ElectricCurrent(core.default_unit_database.known_units['yoctoampere'])
yoctobecquerel = core.Frequency(core.default_unit_database.known_units['yoctobecquerel'])
yoctocandela = core.LuminousIntensity(core.default_unit_database.known_units['yoctocandela'])
yoctocoulomb = core.Charge(core.default_unit_database.known_units['yoctocoulomb'])
yoctofarad = core.Capacitance(core.default_unit_database.known_units['yoctofarad'])
yoctogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['yoctogauss'])
yoctogram = core.Mass(core.default_unit_database.known_units['yoctogram'])
yoctohenry = core.Inductance(core.default_unit_database.known_units['yoctohenry'])
yoctohertz = core.Frequency(core.default_unit_database.known_units['yoctohertz'])
yoctojoule = core.Energy(core.default_unit_database.known_units['yoctojoule'])
yoctokelvin = core.Temperature(core.default_unit_database.known_units['yoctokelvin'])
yoctoliter = core.Volume(core.default_unit_database.known_units['yoctoliter'])
yoctolumen = core.LuminousFlux(core.default_unit_database.known_units['yoctolumen'])
yoctolux = core.Illuminance(core.default_unit_database.known_units['yoctolux'])
yoctometer = core.Length(core.default_unit_database.known_units['yoctometer'])
yoctomole = core.Quantity(core.default_unit_database.known_units['yoctomole'])
yoctonewton = core.Force(core.default_unit_database.known_units['yoctonewton'])
yoctoohm = core.Resistance(core.default_unit_database.known_units['yoctoohm'])
yoctopascal = core.Pressure(core.default_unit_database.known_units['yoctopascal'])
yoctoradian = core.Angle(core.default_unit_database.known_units['yoctoradian'])
yoctosecond = core.Time(core.default_unit_database.known_units['yoctosecond'])
yoctosiemens = core.ElectricalConductance(core.default_unit_database.known_units['yoctosiemens'])
yoctosteradian = core.Angle(core.default_unit_database.known_units['yoctosteradian'])
yoctotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['yoctotesla'])
yoctovolt = core.ElectricPotential(core.default_unit_database.known_units['yoctovolt'])
yoctowatt = core.Power(core.default_unit_database.known_units['yoctowatt'])
yoctoweber = core.MagneticFlux(core.default_unit_database.known_units['yoctoweber'])
yottaampere = core.ElectricCurrent(core.default_unit_database.known_units['yottaampere'])
yottabecquerel = core.Frequency(core.default_unit_database.known_units['yottabecquerel'])
yottacandela = core.LuminousIntensity(core.default_unit_database.known_units['yottacandela'])
yottacoulomb = core.Charge(core.default_unit_database.known_units['yottacoulomb'])
yottafarad = core.Capacitance(core.default_unit_database.known_units['yottafarad'])
yottagauss = core.MagneticFluxDensity(core.default_unit_database.known_units['yottagauss'])
yottagram = core.Mass(core.default_unit_database.known_units['yottagram'])
yottahenry = core.Inductance(core.default_unit_database.known_units['yottahenry'])
yottahertz = core.Frequency(core.default_unit_database.known_units['yottahertz'])
yottajoule = core.Energy(core.default_unit_database.known_units['yottajoule'])
yottakelvin = core.Temperature(core.default_unit_database.known_units['yottakelvin'])
yottaliter = core.Volume(core.default_unit_database.known_units['yottaliter'])
yottalumen = core.LuminousFlux(core.default_unit_database.known_units['yottalumen'])
yottalux = core.Illuminance(core.default_unit_database.known_units['yottalux'])
yottameter = core.Length(core.default_unit_database.known_units['yottameter'])
yottamole = core.Quantity(core.default_unit_database.known_units['yottamole'])
yottanewton = core.Force(core.default_unit_database.known_units['yottanewton'])
yottaohm = core.Resistance(core.default_unit_database.known_units['yottaohm'])
yottapascal = core.Pressure(core.default_unit_database.known_units['yottapascal'])
yottaradian = core.Angle(core.default_unit_database.known_units['yottaradian'])
yottasecond = core.Time(core.default_unit_database.known_units['yottasecond'])
yottasiemens = core.ElectricalConductance(core.default_unit_database.known_units['yottasiemens'])
yottasteradian = core.Angle(core.default_unit_database.known_units['yottasteradian'])
yottatesla = core.MagneticFluxDensity(core.default_unit_database.known_units['yottatesla'])
yottavolt = core.ElectricPotential(core.default_unit_database.known_units['yottavolt'])
yottawatt = core.Power(core.default_unit_database.known_units['yottawatt'])
yottaweber = core.MagneticFlux(core.default_unit_database.known_units['yottaweber'])
yr = core.Time(core.default_unit_database.known_units['yr'])
yrad = core.Angle(core.default_unit_database.known_units['yrad'])
ys = core.Time(core.default_unit_database.known_units['ys'])
ysr = core.Angle(core.default_unit_database.known_units['ysr'])
zA = core.ElectricCurrent(core.default_unit_database.known_units['zA'])
zBq = core.Frequency(core.default_unit_database.known_units['zBq'])
zC = core.Charge(core.default_unit_database.known_units['zC'])
zF = core.Capacitance(core.default_unit_database.known_units['zF'])
zGauss = core.MagneticFluxDensity(core.default_unit_database.known_units['zGauss'])
zH = core.Inductance(core.default_unit_database.known_units['zH'])
zHz = core.Frequency(core.default_unit_database.known_units['zHz'])
zJ = core.Energy(core.default_unit_database.known_units['zJ'])
zK = core.Temperature(core.default_unit_database.known_units['zK'])
zN = core.Force(core.default_unit_database.known_units['zN'])
zOhm = core.Resistance(core.default_unit_database.known_units['zOhm'])
zPa = core.Pressure(core.default_unit_database.known_units['zPa'])
zS = core.ElectricalConductance(core.default_unit_database.known_units['zS'])
zT = core.MagneticFluxDensity(core.default_unit_database.known_units['zT'])
zV = core.ElectricPotential(core.default_unit_database.known_units['zV'])
zW = core.Power(core.default_unit_database.known_units['zW'])
zWb = core.MagneticFlux(core.default_unit_database.known_units['zWb'])
zcd = core.LuminousIntensity(core.default_unit_database.known_units['zcd'])
zeptoampere = core.ElectricCurrent(core.default_unit_database.known_units['zeptoampere'])
zeptobecquerel = core.Frequency(core.default_unit_database.known_units['zeptobecquerel'])
zeptocandela = core.LuminousIntensity(core.default_unit_database.known_units['zeptocandela'])
zeptocoulomb = core.Charge(core.default_unit_database.known_units['zeptocoulomb'])
zeptofarad = core.Capacitance(core.default_unit_database.known_units['zeptofarad'])
zeptogauss = core.MagneticFluxDensity(core.default_unit_database.known_units['zeptogauss'])
zeptogram = core.Mass(core.default_unit_database.known_units['zeptogram'])
zeptohenry = core.Inductance(core.default_unit_database.known_units['zeptohenry'])
zeptohertz = core.Frequency(core.default_unit_database.known_units['zeptohertz'])
zeptojoule = core.Energy(core.default_unit_database.known_units['zeptojoule'])
zeptokelvin = core.Temperature(core.default_unit_database.known_units['zeptokelvin'])
zeptoliter = core.Volume(core.default_unit_database.known_units['zeptoliter'])
zeptolumen = core.LuminousFlux(core.default_unit_database.known_units['zeptolumen'])
zeptolux = core.Illuminance(core.default_unit_database.known_units['zeptolux'])
zeptometer = core.Length(core.default_unit_database.known_units['zeptometer'])
zeptomole = core.Quantity(core.default_unit_database.known_units['zeptomole'])
zeptonewton = core.Force(core.default_unit_database.known_units['zeptonewton'])
zeptoohm = core.Resistance(core.default_unit_database.known_units['zeptoohm'])
zeptopascal = core.Pressure(core.default_unit_database.known_units['zeptopascal'])
zeptoradian = core.Angle(core.default_unit_database.known_units['zeptoradian'])
zeptosecond = core.Time(core.default_unit_database.known_units['zeptosecond'])
zeptosiemens = core.ElectricalConductance(core.default_unit_database.known_units['zeptosiemens'])
zeptosteradian = core.Angle(core.default_unit_database.known_units['zeptosteradian'])
zeptotesla = core.MagneticFluxDensity(core.default_unit_database.known_units['zeptotesla'])
zeptovolt = core.ElectricPotential(core.default_unit_database.known_units['zeptovolt'])
zeptowatt = core.Power(core.default_unit_database.known_units['zeptowatt'])
zeptoweber = core.MagneticFlux(core.default_unit_database.known_units['zeptoweber'])
zettaampere = core.ElectricCurrent(core.default_unit_database.known_units['zettaampere'])
zettabecquerel = core.Frequency(core.default_unit_database.known_units['zettabecquerel'])
zettacandela = core.LuminousIntensity(core.default_unit_database.known_units['zettacandela'])
zettacoulomb = core.Charge(core.default_unit_database.known_units['zettacoulomb'])
zettafarad = core.Capacitance(core.default_unit_database.known_units['zettafarad'])
zettagauss = core.MagneticFluxDensity(core.default_unit_database.known_units['zettagauss'])
zettagram = core.Mass(core.default_unit_database.known_units['zettagram'])
zettahenry = core.Inductance(core.default_unit_database.known_units['zettahenry'])
zettahertz = core.Frequency(core.default_unit_database.known_units['zettahertz'])
zettajoule = core.Energy(core.default_unit_database.known_units['zettajoule'])
zettakelvin = core.Temperature(core.default_unit_database.known_units['zettakelvin'])
zettaliter = core.Volume(core.default_unit_database.known_units['zettaliter'])
zettalumen = core.LuminousFlux(core.default_unit_database.known_units['zettalumen'])
zettalux = core.Illuminance(core.default_unit_database.known_units['zettalux'])
zettameter = core.Length(core.default_unit_database.known_units['zettameter'])
zettamole = core.Quantity(core.default_unit_database.known_units['zettamole'])
zettanewton = core.Force(core.default_unit_database.known_units['zettanewton'])
zettaohm = core.Resistance(core.default_unit_database.known_units['zettaohm'])
zettapascal = core.Pressure(core.default_unit_database.known_units['zettapascal'])
zettaradian = core.Angle(core.default_unit_database.known_units['zettaradian'])
zettasecond = core.Time(core.default_unit_database.known_units['zettasecond'])
zettasiemens = core.ElectricalConductance(core.default_unit_database.known_units['zettasiemens'])
zettasteradian = core.Angle(core.default_unit_database.known_units['zettasteradian'])
zettatesla = core.MagneticFluxDensity(core.default_unit_database.known_units['zettatesla'])
zettavolt = core.ElectricPotential(core.default_unit_database.known_units['zettavolt'])
zettawatt = core.Power(core.default_unit_database.known_units['zettawatt'])
zettaweber = core.MagneticFlux(core.default_unit_database.known_units['zettaweber'])
zg = core.Mass(core.default_unit_database.known_units['zg'])
zl = core.Volume(core.default_unit_database.known_units['zl'])
zlm = core.LuminousFlux(core.default_unit_database.known_units['zlm'])
zlx = core.Illuminance(core.default_unit_database.known_units['zlx'])
zm = core.Length(core.default_unit_database.known_units['zm'])
zmol = core.Quantity(core.default_unit_database.known_units['zmol'])
zrad = core.Angle(core.default_unit_database.known_units['zrad'])
zs = core.Time(core.default_unit_database.known_units['zs'])
zsr = core.Angle(core.default_unit_database.known_units['zsr'])

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
