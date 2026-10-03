from app.services.rag_service import ask_question

response = ask_question("What are the Treatment Strategies of Acne?")

print(response[0]["text"])
# print(response.content[0]["text"])
