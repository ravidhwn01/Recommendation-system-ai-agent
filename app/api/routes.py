from fastapi import APIRouter

from app.agent.context import ContextManager
from app.agent.decision_engine import DecisionEngine
from app.agent.guardrails import GuardRails
from app.agent.intent import IntentClassifier
from app.agent.recommender import RecommendationEngine

from app.models.chat import ChatRequest
from app.models.recommendation import Recommendation
from app.models.response import ChatResponse

router = APIRouter()

context = ContextManager()

classifier = IntentClassifier()

decision_engine = DecisionEngine()

guard = GuardRails()

recommender = RecommendationEngine()


@router.get("/health")
async def health():

    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    query = context.get_latest_user_message(
        [m.model_dump() for m in request.messages]
    )

    if not guard.is_allowed(query):

        return ChatResponse(
            reply="Sorry, I can only answer questions related to SHL assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    intent = classifier.classify(query)

    decision = decision_engine.decide(intent)

    if decision == "recommend":

        docs = recommender.recommend(query)

        recommendations = []

        for doc in docs:

            recommendations.append(
                Recommendation(
                    name=doc.metadata.get("name", ""),
                    url=doc.metadata.get("url", ""),
                    test_type=doc.metadata.get("test_type", ""),
                )
            )

        return ChatResponse(
            reply="Here are the best matching SHL assessments.",
            recommendations=recommendations,
            end_of_conversation=False,
        )

    if decision == "compare":

        return ChatResponse(
            reply="Comparison module coming next.",
            recommendations=[],
            end_of_conversation=False,
        )

    if decision == "greet":

        return ChatResponse(
            reply="Hello! Tell me the role you're hiring for, and I'll recommend suitable SHL assessments.",
            recommendations=[],
            end_of_conversation=False,
        )

    return ChatResponse(
        reply="Could you provide more details about the role or assessment requirements?",
        recommendations=[],
        end_of_conversation=False,
    )