# Dependencies in FastAPI

FastAPI has a powerful but intuitive Dependency Injection system. You declare
a dependency with `Depends`, and FastAPI takes care of calling it and injecting
the results.

```python
from fastapi import Depends, FastAPI
app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

## Why use dependencies

Dependencies are useful to share logic (code repetition), share database
connections, enforce security, authentication, role requirements, and much more.
This minimizes code repetition and reduces the chance of bugs.

## Classes as dependencies

You can also use a class as a dependency. FastAPI will use the class to create
an instance, and the instance will be passed to your path operation function.

```python
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit
```
