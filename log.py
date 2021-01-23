import math

x=math.log(190/255,180/255)

t2=(167.1952381/255)**0.45

t1=(185/255)**0.45
b=t1/t2
t=(111.9904762/255)**0.45
rest=t*b
rest=(rest**(2.15))*255

print(rest)
