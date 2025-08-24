from fastapi import FastAPI, Body
from models.auto_sql import run_sqlcoder
app = FastAPI()

@app.post("/generate_sql")
def generate_sql(prompt: str = Body(...), max_new_tokens: int = Body(200)):
    """
    API endpoint to generate or rewrite SQL queries.
    """
    result = run_sqlcoder(prompt, max_new_tokens)
    return {"prompt": prompt, "sql": result}
