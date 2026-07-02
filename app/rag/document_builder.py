from langchain_core.documents import Document

from app.data.schemas import Assessment


class DocumentBuilder:

    def build(
        self,
        assessments: list[Assessment],
    ) -> list[Document]:
        documents = []

        for assessment in assessments:
            page_content = f"""
Assessment Name:
{assessment.name}

Description:
{assessment.description}

Assessment Type:
{assessment.test_type}

Duration:
{assessment.duration}

Job Levels:
{", ".join(assessment.job_levels)}

Languages:
{", ".join(assessment.languages)}

Skills:
{", ".join(assessment.skills)}

Remote Testing:
{assessment.remote_testing}

Adaptive:
{assessment.adaptive}
"""

            metadata = {
                "name": assessment.name,
                "url": assessment.url,
                "duration": assessment.duration,
                "test_type": assessment.test_type,
                "remote_testing": assessment.remote_testing,
                "adaptive": assessment.adaptive,
                "job_levels": ", ".join(assessment.job_levels),
                "languages": ", ".join(assessment.languages),
                "skills": ", ".join(assessment.skills),
            }

            # Chroma metadata values must be str/int/float/bool (no None).
            metadata = {
                key: value
                for key, value in metadata.items()
                if value is not None
            }

            documents.append(
                Document(
                    metadata=metadata,
                    page_content=page_content.strip(),
                )
            )

        return documents