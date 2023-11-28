import ray
import time

@ray.remote #zde je možné specifikovat nějaké parametry spuštění např. @ray.remote(num_cpus=4)
def square(x):
    return x * x

ray.init() #pokud bych dal "ray://123.45.67.89:10001" tak potřebuji ray client - je to pro remote přístup
start = time.time()

futures = []

for i in range(100):
    futures.append(square.remote(i)) #futures je list příslibů - odkaz na budoucí výsledek - zhmotněný výsledek dostanu po zavolání fce get - lze volat get
    # na celý list příslibů - viz dole ray.get(futures) nebo jen třeba na jeden záznam ray.get(futures[10])
    # po zavolání get se čeká na dokončení výsledku výpočtu


#print("Pokracuji si klidně dál") #program tady může pokračovat - výpočet běží v clusteru - na výsledky si počkám až když zavolám fci get
data = ray.get(futures) #data je klasický list výsledků [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
#print(data)
print(time.time()-start)