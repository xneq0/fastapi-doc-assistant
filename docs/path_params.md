# Path Parameters in FastAPI

You can declare path "parameters" or "variables" with the same syntax used by
Python format strings:

```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of the path parameter `item_id` will be passed to your function as the
argument `item_id`. Because you declared it as `int`, FastAPI will validate that
the value is an integer and convert the string from the URL into an `int`.

## Data validation

If you pass a non-integer path value like `/items/foo`, you get a clear HTTP
error with status code 422, because `foo` is not a valid `int`. This automatic
data validation is provided by Pydantic.

## Order matters

When creating path operations, fixed paths like `/users/me` must be declared
before variable paths like `/users/{user_id}`, otherwise the variable path would
also match `/users/me`.
