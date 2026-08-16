# Handling Errors in FastAPI

There are many situations where you need to notify the client that something went
wrong. To return HTTP responses with errors, you use `HTTPException`.

```python
from fastapi import FastAPI, HTTPException
app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
```

Because it's a Python exception, you `raise` it, you don't `return` it. When you
raise an `HTTPException`, FastAPI stops the request and returns the HTTP error
response with the given status code and detail.

## Custom headers

You can add custom headers to the error response by passing the `headers`
argument, which is useful for some types of security.

## Custom exception handlers

You can add custom exception handlers with `@app.exception_handler()` to override
the default behavior for a given exception type.
