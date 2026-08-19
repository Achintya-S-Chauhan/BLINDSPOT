def interpret_context(window_title: str, ocr_text: str):
    title = window_title.lower()
    text = ocr_text.lower()

    # STRONG contexts (high confidence)
    # ChatGPT only if NOT in VS Code
    if (
        ("chatgpt" in text or "gpt" in text)
        and "visual studio code" not in title
        and "vscode" not in title
    ):
        return "chatting", 3
    
    if "visual studio code" in title or "vscode" in title:
        return "coding", 3
        
    if "youtube" in title or "watch" in title:
        return "watching a video", 3

    if "terminal" in title or "powershell" in title or "cmd" in title:
        return "working in a terminal", 3

    # MEDIUM confidence
    if "chrome" in title or "edge" in title or "firefox" in title:
        if len(text) > 120:
            return "reading in a browser", 2
        return "browsing", 2

    # WEAK fallback
    return "using an application", 1
