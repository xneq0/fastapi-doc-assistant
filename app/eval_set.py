"""评测集：真实开发者会问的问题 -> 正确答案应来自哪篇文档。

每条标注 expected_source（正确出处）。评测时看检索结果 top-k 里
是否命中正确来源，从而算出命中率(recall@k)。
故意混入一些「关键词型」问题（函数名/状态码/报错），BM25 在这类问题上更强。
"""
EVAL = [
    # 语义型（向量擅长）
    {"q": "How can I make a query parameter mandatory?", "src": "query_params.md"},
    {"q": "What is dependency injection good for?", "src": "dependencies.md"},
    {"q": "How do I run something after returning the response?", "src": "background_tasks.md"},
    {"q": "How do I send JSON data from the client to my API?", "src": "request_body.md"},
    {"q": "How does FastAPI validate the type of a URL variable?", "src": "path_params.md"},
    {"q": "How do I tell the client that a resource was not found?", "src": "error_handling.md"},

    # 关键词型（BM25 擅长：精确的类名/状态码/API 名）
    {"q": "How to use HTTPException?", "src": "error_handling.md"},
    {"q": "What does BackgroundTasks add_task do?", "src": "background_tasks.md"},
    {"q": "Why do I get a 422 status code?", "src": "path_params.md"},
    {"q": "How to use Depends with a class?", "src": "dependencies.md"},
    {"q": "What is the skip and limit query parameter?", "src": "query_params.md"},
    {"q": "How to declare a Pydantic BaseModel as request body?", "src": "request_body.md"},
]
