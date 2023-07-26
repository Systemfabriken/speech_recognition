from transformers import GPT2LMHeadModel, GPT2TokenizerFast, Conversation, pipeline

directory = "./models/dialoGPT"

# Check if model and tokenizer exist in the directory
try:
    model = GPT2LMHeadModel.from_pretrained(directory)
    tokenizer = GPT2TokenizerFast.from_pretrained(directory, padding_side='left')
except:
    print("Model and/or tokenizer not found in the directory. Downloading and saving model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-large")
    tokenizer = GPT2TokenizerFast.from_pretrained("microsoft/DialoGPT-large", padding_side='left')
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

# nlp = pipeline('conversational', model=model, tokenizer=tokenizer)

# Initialize a conversation
# conversation = Conversation("Hello, I'm Robot, your home-security assistant.")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        break

    # Append the user input to the conversation
    # conversation.add_user_input(user_input, overwrite=True)

    # Get the model's response
    # conversation = nlp(conversation)

    # encode the prompt
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    generation_config = {
        "max_length": 150,
        "num_beams": 5,
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "pad_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "length_penalty": 0.5,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
    }

    # generate the response
    sample_outputs = model.generate(
        input_ids, 
        max_length=generation_config["max_length"], 
        num_beams=generation_config["num_beams"], 
        no_repeat_ngram_size=generation_config["no_repeat_ngram_size"], 
        early_stopping=generation_config["early_stopping"], 
        do_sample=generation_config["do_sample"], 
        top_k=generation_config["top_k"], 
        top_p=generation_config["top_p"], 
        temperature=generation_config["temperature"], 
        pad_token_id=generation_config["pad_token_id"], 
        bos_token_id=generation_config["bos_token_id"], 
        eos_token_id=generation_config["eos_token_id"], 
        length_penalty=generation_config["length_penalty"], 
        num_return_sequences=generation_config["num_return_sequences"], 
        repetition_penalty=generation_config["repetition_penalty"]
    )

    # decode the response
    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
    # response = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    
    # print("Robot: ", conversation.generated_responses[-1])
    # print("Robot: ", response)