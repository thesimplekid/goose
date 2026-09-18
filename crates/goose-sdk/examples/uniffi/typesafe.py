"""Evaluate structured state with Jev through the GDK bindings."""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "generated"))

from goose import (  # noqa: E402
    TypeSafeAnswer,
    TypeSafeQuestion,
    typesafe_client,
    typesafe_default_model,
)


async def main() -> None:
    client = typesafe_client(os.environ["TYPESAFE_API_KEY"])
    result = await client.evaluate(
        typesafe_default_model(),
        json.dumps({"ticket": "I was charged twice. Please fix this ASAP."}),
        {
            "urgent": TypeSafeQuestion.Noul(
                instructions="Is this urgent?", criteria={}
            ),
            "team": TypeSafeQuestion.Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments and refunds", "support": None},
            ),
            "severity": TypeSafeQuestion.Score(
                instructions="How severe is the problem?",
                criteria=["Minor", "Major", "Critical"],
            ),
        },
    )
    answer = result.answers["team"]
    if isinstance(answer, TypeSafeAnswer.Choice):
        print(answer.choice, answer.confidence, answer.probabilities)
    print(result.model, result.input_tokens, result.output_tokens)


if __name__ == "__main__":
    asyncio.run(main())
