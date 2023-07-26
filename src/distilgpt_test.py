from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# specify your directory
directory = "models"

# Check if model and tokenizer exist in the directory
try:
    model = GPT2LMHeadModel.from_pretrained(directory)
    tokenizer = GPT2Tokenizer.from_pretrained(directory)
except:
    print("Model and/or tokenizer not found in the directory. Downloading and saving model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

# model.eval()

# # let's say we want to generate a response to the phrase "Hello, how are you?"
# input_ids = tokenizer.encode("Hello, how are you?", return_tensors="pt")

# # we're going to generate the top five responses
# sample_outputs = model.generate(input_ids, do_sample=True, max_length=50, top_k=5)

# print("Output:\n" + 100 * '-')
# for i, sample_output in enumerate(sample_outputs):
#   print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

while True:
    question = input("You: ")

    if question.lower() == 'quit':
        break

    prompt = f"The assistant replies to '{question}': "

    # encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # generate the response
    sample_outputs = model.generate(input_ids, do_sample=True, max_length=150, top_k=5)

    # decode the response
    response = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

    # print the response
    print("Bot: ", response[len(prompt):])  # Exclude the prompt from the output