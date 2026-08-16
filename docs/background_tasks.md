# Background Tasks in FastAPI

You can define background tasks to be run after returning a response. This is
useful for operations that need to happen after a request, but that the client
doesn't really have to wait for before receiving the response.

```python
from fastapi import BackgroundTasks, FastAPI
app = FastAPI()

def write_notification(email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        email_file.write(f"notification for {email}: {message}")

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

Common use cases include sending email notifications after an action, and
processing data: for example, accepting a file that must go through a slow process.

For heavier background computation that needs to run in separate processes or
even other servers, you may want to use a bigger tool like Celery instead.
