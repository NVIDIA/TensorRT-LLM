#!/usr/bin/env python
"""
Entry point for uvicorn with reload functionality.
This file creates a FastAPI app that can be imported by uvicorn.
"""

from fastapi import FastAPI

# Create a placeholder app that will be replaced when the server starts
app = FastAPI()


@app.get("/")
async def root():
    return {
        "message":
        "TensorRT-LLM OpenAI Server - Server is starting up, please wait..."
    }


@app.get("/health")
async def health():
    return {"status": "Server is starting up"}


@app.get("/v1/models")
async def models():
    return {"error": "Server is starting up, please wait..."}
