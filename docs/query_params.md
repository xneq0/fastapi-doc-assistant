# Query Parameters in FastAPI

When you declare function parameters that are not part of the path parameters,
they are automatically interpreted as query parameters.

```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

The query is the set of key-value pairs that go after the `?` in a URL,
separated by `&` characters. For example: `http://127.0.0.1:8000/items/?skip=0&limit=10`.

Because query parameters are not a fixed part of a path, they can be optional
and can have default values. In the example above `skip` defaults to 0 and
`limit` defaults to 10.

## Required query parameters

If you do not declare a default value, the parameter becomes required:

```python
@app.get("/items/{item_id}")
async def read_user_item(item_id: str, needy: str):
    return {"item_id": item_id, "needy": needy}
```

Here `needy` is a required query parameter of type `str`.
