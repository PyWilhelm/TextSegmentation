import time

time1 = time.time()
a = list(range(100000))
for i in range(10000):
    b = sum(a)
time2 = time.time()
print(time2 - time1)