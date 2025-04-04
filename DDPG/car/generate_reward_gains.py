import numpy as np

a=np.array([np.random.uniform(1,7,2)], dtype='int')
while len(a)<5:
  b=np.array(np.random.uniform(1,7,2),dtype='int')
  unique=True
  for aa in a:
    if all(b==aa):
      unique=False
  if unique:
    a=np.append(a,np.array([b]),axis=0)

print(a)