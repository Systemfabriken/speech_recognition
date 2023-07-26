from transformers import pipeline, DistilBertForQuestionAnswering, DistilBertTokenizerFast

directory = "./models/distilBERT"

# Check if model and tokenizer exist in the directory
try:
    model = DistilBertForQuestionAnswering.from_pretrained(directory)
    tokenizer = DistilBertTokenizerFast.from_pretrained(directory)
except:
    print("Model and/or tokenizer not found in the directory. Downloading and saving model and tokenizer...")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased-distilled-squad")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

# Initialize the question answering pipeline
nlp = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer=tokenizer)

# Context for a home-security system servant
context = r"""
Distilbert, known as "Robot", is a key component of a sophisticated home-security system installed in a two-story house located in Springfield. This system is designed to provide safety and convenience to the house's owners, John and Mary Smith, who are often away on business trips.
Robot's main role is to greet people who arrive at the front door. It uses an advanced facial recognition system to identify the visitors. If the visitors are recognized as friends or family, Robot gives them a warm greeting and notifies the owners about their arrival. If the visitor is unrecognized, Robot asks for their name and the purpose of the visit, and informs the owners accordingly.
Robot can answer simple questions about herself, such as her role and functionality. She can also provide information about the house, including details about its architectural design, the number of rooms, and the smart devices installed.
In addition to this, Robot can provide up-to-date information about the state of the door – whether it is locked or unlocked, and the last time it was accessed. If there's an emergency, such as a forced entry, Robot can contact the local police department and alert the owners.
If there are questions Robot cannot answer, such as those related to the owners' personal lives or confidential matters, Robot kindly lets the person know that she is unable to provide that information.
Robot also has a gentle side. She celebrates the owners' birthdays and anniversaries by playing their favorite songs and leaving warm greetings for them.
"""

while True:
    # Get the context and question from the user
    # context = input("Enter the context: ")
    question = input("Enter the question: ")

    if question.lower() == 'quit':
        break

    # Use the model to find the answer
    result = nlp(question=question, context=context)

    # Print the answer
    print("Answer: ", result['answer'])