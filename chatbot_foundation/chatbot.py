from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initializes conversation history
conversation_history = []

while True:
    # Limit conversation history to avoid token overflow
    if len(conversation_history) > 12:  # Keeps last 6 exchanges
        conversation_history = conversation_history[-12:]

    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get user input
    input_text = input("> ")

    # Tokenize with truncation to prevent overflow
    inputs = tokenizer.encode_plus(
        history_string, input_text, return_tensors="pt", truncation=True, max_length=128
    )

    # Generate response with max_new_tokens to control length
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print(response)

    # Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
