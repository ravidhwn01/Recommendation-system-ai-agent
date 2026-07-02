class PromptBuilder:

    def build_recommendation_prompt(
        self,
        query: str,
        documents,
    ):

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        return f"""
You are an SHL Assessment Recommendation Assistant.

Answer ONLY from the provided context.

Context:

{context}

User Query:

{query}

Recommend the best assessments.
"""

    def build_comparison_prompt(
        self,
        assessment1,
        assessment2,
    ):

        return f"""
Compare the following two SHL assessments.

Assessment 1

{assessment1.page_content}

Assessment 2

{assessment2.page_content}

Compare

Purpose

Duration

Skills

Job Levels

Remote Testing

Adaptive Testing
"""