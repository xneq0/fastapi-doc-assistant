# Request Body in FastAPI

When you need to send data from a client to your API, you send it as a request
body. To declare one, you use Pydantic models.

```python
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item
```

With just that Python type declaration, FastAPI will read the body of the request
as JSON, validate the data, convert it to the declared types, and give you the
received data in the parameter `item`. It also generates JSON Schema and adds it
to the interactive API docs automatically.

## Request body + path + query

You can declare body, path, and query parameters at the same time; FastAPI will
recognize each of them and take the data from the correct place.
