import uvicorn

if __name__ == "__main__":
    print("Starting Comment Classification API...")
    print("API: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
