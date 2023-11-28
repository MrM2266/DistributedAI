import ray

@ray.remote
class Counter:
    def __init__(self):
        self.i = 0

    def get_data(self):
        return self.i
    
    def incr(self, value):
        self.i += value


ray.init(address = "127.0.0.1:6379")

c = Counter.remote() #jedná se pouze o actor handle - nejedná se o klasický python objekt - objekt se vytvoří na daném clusteru
# pomocí actor handle se můžeme odkazovat na tuto instanci v clusteru
print(type(c))

for _ in range(10):
    c.incr.remote(10) #voláme fci na instanci v clusteru

print(ray.get(c.get_data.remote())) #počkáme si na výsledek fce a vypíšeme