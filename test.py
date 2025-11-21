from Core.BaseAgent import BaseAgent


agent = BaseAgent()

while True:
    text = input("You: ")
    if text == "quit":
        break

    agent.respond(text)
