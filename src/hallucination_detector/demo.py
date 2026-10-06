"""Run the first end-to-end claim and evidence baseline."""

from .baseline import extract_claims, verify_claim
from .domain import Evidence


def main() -> None:
    response = "The Pacific Ocean is the largest ocean. Canberra is the capital of Australia."
    evidence = (
        Evidence("e1", "The Pacific Ocean is the largest and deepest ocean on Earth.", "reference-a"),
        Evidence("e2", "Canberra is the capital city of Australia.", "reference-b"),
    )
    for claim in extract_claims(response):
        result = verify_claim(claim, evidence)
        print(f"{claim.id}: {result.label} ({result.similarity:.2f}) - {result.explanation}")


if __name__ == "__main__":
    main()