import unit_array
meter = unit_array.UnitArray('m')
km = unit_array.UnitArray('km')
mm = unit_array.UnitArray('mm')
second = unit_array.UnitArray('s')
inch = unit_array.UnitArray('inch')
v1 = unit_array.Value(1.0, 1, 1, 0, meter, meter)
v2 = unit_array.Value(1.0, 1, 1, 3, meter, km)
v2 + v1
v3 = v1 * v2
v4 = v1+v2
v5 = unit_array.Value(3.4, 1.0, 1, second, second)
v6 = v1 - v2

x1 = unit_array.Value(1, 254, 1, -4, meter, inch)
x2 = unit_array.Value(1, 1, 1, -3, meter, mm);
print "v6 == 0: ", v6==0
print v1 < v2
v2+v5
-v2

