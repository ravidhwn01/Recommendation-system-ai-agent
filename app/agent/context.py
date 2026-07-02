from typing import Any


class ContextManager:
	"""
	Extract useful information from the complete
	conversation history.
	"""

	@staticmethod
	def get_latest_user_message(messages: list[dict[str, Any]]) -> str:
		"""
		Returns the latest user message.
		"""

		for message in reversed(messages):

			if message.get("role") == "user":
				return message.get("content", "").strip()

		return ""

	@staticmethod
	def get_all_user_text(messages: list[dict[str, Any]]) -> str:
		"""
		Concatenate every user message so context accumulates across turns.
		"""

		parts = [
			message.get("content", "").strip()
			for message in messages
			if message.get("role") == "user"
			and message.get("content", "").strip()
		]

		return " ".join(parts)

	@staticmethod
	def get_conversation(messages: list[dict[str, Any]]) -> str:
		"""
		Convert the conversation history into plain text.
		"""

		conversation = []

		for message in messages:

			role = message.get("role", "user").upper()

			content = message.get("content", "")

			conversation.append(
				f"{role}: {content}"
			)

		return "\n".join(conversation)
