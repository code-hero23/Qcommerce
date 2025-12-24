def label_sentiment(text):
    text = text.lower()

    positive_words = [
        "good", "fast", "excellent", "best", "cheap",
        "satisfied", "happy", "fulfilled", "awesome",
        "great", "quick", "support", "nice"
    ]

    negative_words = [
        "bad", "slow", "delay", "poor", "expensive",
        "worst", "damaged", "late", "missing", "issue",
        "problem"
    ]

    if any(word in text for word in positive_words):
        return "Positive"
    elif any(word in text for word in negative_words):
        return "Negative"
    else:
        return "Neutral"
