RISK_TAXONOMY = [
    "Market Risk",
    "Liquidity Risk",
    "Credit Risk",
    "Operational Risk",
    "Cybersecurity Risk",
    "Regulatory/Legal Risk",
    "Supply Chain Risk",
    "ESG/Climate Risk",
    "Reputational Risk",
    "Model Risk",
]


def canonical_labels() -> list[str]:
    return list(RISK_TAXONOMY)


def to_key(label: str) -> str:
    return label.replace("/", " ").replace(" ", "_").lower()
