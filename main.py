from sp_llm.whisper import load_model

if __name__ == "__main__":
    model = load_model("large-v2", device="cuda")
