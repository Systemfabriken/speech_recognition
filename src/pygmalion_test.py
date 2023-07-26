from transformers import GPT2LMHeadModel, GPT2TokenizerFast

directory = "./models/Pygmalion6b"

# Check if model and tokenizer exist in the directory
try:
    model = GPT2LMHeadModel.from_pretrained(directory)
    tokenizer = GPT2TokenizerFast.from_pretrained(directory, padding_side='left')
except:
    print("Model and/or tokenizer not found in the directory. Downloading and saving model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("PygmalionAI/pygmalion-6b")
    tokenizer = GPT2TokenizerFast.from_pretrained("PygmalionAI/pygmalion-6b", padding_side='left')
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        break

    # Example Persona and Dialogue History formatting
    persona = "Robot's Persona: I am a highly intelligent robot with extensive knowledge on various topics."
    dialogue_history = "Robot: Hi there! How can I assist you today?\nYou: Tell me a fun fact about space."
    character = "Robot"

    formatted_input = f"{persona}\n<START>\n{dialogue_history}\nYou: {user_input}\n{character}:"

    # encode the prompt
    input_ids = tokenizer.encode(formatted_input, return_tensors='pt')

    # generate the response
    sample_outputs = model.generate(input_ids)

    # decode the response
    response = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    
    print(f"{character}: ", response)
