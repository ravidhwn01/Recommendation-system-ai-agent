from app.agent.context import ContextManager
from app.agent.intent import IntentClassifier
from app.agent.decision_engine import DecisionEngine


messages = [
    {
        "role": "user",
        "content": "I need an assessment for a Java developer."
    }
]

context = ContextManager()

query = context.get_latest_user_message(messages)

classifier = IntentClassifier()

intent = classifier.classify(query)

engine = DecisionEngine()

decision = engine.decide(intent)

print("=" * 50)
print("Latest User Query:")
print(query)

print()

print("Intent:")
print(intent)

print()

print("Decision:")
print(decision)

print("=" * 50)