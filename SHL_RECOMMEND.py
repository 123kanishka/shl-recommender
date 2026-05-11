import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # Allows local retrieval tests before installing API deps.
    FastAPI = None
    HTTPException = ValueError
    BaseModel = object


FULL_CATALOG_PATH = Path(__file__).with_name("assessment_records_full.json")
CATALOG_PATH = Path(__file__).with_name("assessment_records")
SUPPLEMENTAL_CATALOG_PATH = Path(__file__).with_name("supplemental_records.json")
MAX_RECOMMENDATIONS = 10

STOP_WORDS = {
    "a", "an", "and", "are", "around", "as", "assessment", "assessments",
    "be", "for", "from", "hiring", "i", "in", "is", "it", "need", "of",
    "on", "or", "our", "please", "role", "test", "tests", "that", "the",
    "their", "to", "want", "we", "who", "with", "can", "you", "help",
    "me", "pick", "choose", "select",
}

GENERIC_QUERY_WORDS = {
    "assessment", "assessments", "test", "tests", "hire", "hiring",
    "candidate", "candidates", "employee", "employees", "screening",
    "help", "pick", "choose", "select",
}

TECH_TERMS = {
    ".net", "ado.net", "agile", "angular", "asp.net", "c", "c#", "c++",
    "cobol", "css", "html", "java", "javascript", "mvc", "mvvm", "python",
    "sql", "wcf", "wpf", "xaml", "spring", "aws", "docker", "linux",
    "networking", "rust", "rest", "restful",
}

SKILL_ALIASES = {
    "backend": ["server", "api", "java", "dotnet", "sql"],
    "front-end": ["frontend", "html", "css", "javascript"],
    "frontend": ["front-end", "html", "css", "javascript"],
    "fullstack": ["frontend", "backend", "javascript", "java", "sql"],
    "stakeholder": ["communication", "competencies", "situational"],
    "stakeholders": ["communication", "competencies", "situational"],
    "personality": ["personality", "behavior", "opq"],
    "behaviour": ["behavior", "personality"],
    "behavioral": ["behavior", "personality", "situational"],
    "english": ["english", "usa", "international"],
    "developer": ["programming", "coding", "software"],
    "engineer": ["programming", "coding", "software"],
    "cognitive": ["ability", "aptitude", "reasoning", "verify"],
    "numerical": ["numerical", "reasoning", "verify"],
    "situational": ["situational", "judgment", "scenarios"],
    "judgement": ["judgment", "situational", "scenarios"],
    "judgment": ["judgment", "situational", "scenarios"],
    "graduate": ["graduate", "scenarios"],
    "hipaa": ["hipaa", "security"],
    "healthcare": ["medical", "hipaa"],
    "spanish": ["spanish", "latin", "american"],
    "sales": ["sales", "transformation", "opq", "mq"],
    "safety": ["safety", "dependability", "dsi"],
    "admin": ["administrative", "excel", "word"],
    "administrative": ["admin", "excel", "word"],
    "contact": ["contact", "center", "call", "svar"],
    "centre": ["center", "contact", "call", "svar"],
    "center": ["center", "contact", "call", "svar"],
}

TYPE_CODES = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Development & 360": "D",
}


def load_assessment_records(file_path: Path = CATALOG_PATH) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as file_obj:
        return normalize_assessment_records(json.load(file_obj, strict=False))


def load_catalog_records() -> List[Dict[str, Any]]:
    primary_catalog = FULL_CATALOG_PATH if FULL_CATALOG_PATH.exists() else CATALOG_PATH
    records = load_assessment_records(primary_catalog)
    if primary_catalog != FULL_CATALOG_PATH and SUPPLEMENTAL_CATALOG_PATH.exists():
        records.extend(load_assessment_records(SUPPLEMENTAL_CATALOG_PATH))

    merged: Dict[str, Dict[str, Any]] = {}
    for record in records:
        key = clean_text(record.get("link")) or clean_text(record.get("name"))
        if key and key not in merged:
            merged[key] = record
    return list(merged.values())


def normalize_assessment_records(json_data: Any) -> List[Dict[str, Any]]:
    if isinstance(json_data, list):
        records = json_data
    elif isinstance(json_data, dict):
        for key in ("records", "assessment_records", "data", "items", "results"):
            if isinstance(json_data.get(key), list):
                records = json_data[key]
                break
        else:
            records = [json_data]
    else:
        raise TypeError(f"Expected assessment records as list or dict, got {type(json_data).__name__}")

    return [record for record in records if isinstance(record, dict)]


def tokenize(text: Any) -> List[str]:
    tokens = re.findall(r"[a-z0-9+#.]+", str(text).lower())
    expanded: List[str] = []
    for token in tokens:
        if token not in STOP_WORDS:
            expanded.append(token)
        expanded.extend(SKILL_ALIASES.get(token, []))
    return expanded


def clean_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def test_type_for(record: Dict[str, Any]) -> str:
    codes = [TYPE_CODES.get(key, key[:1].upper()) for key in record.get("keys", [])]
    return "/".join(dict.fromkeys(codes)) or "K"


def recommendation_payload(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        "name": clean_text(record.get("name")),
        "url": clean_text(record.get("link")),
        "test_type": test_type_for(record),
    }


def is_individual_test_record(record: Dict[str, Any]) -> bool:
    section = clean_text(record.get("catalog_section")).lower()
    return section != "pre-packaged job solutions"


@dataclass
class ScoredRecord:
    score: float
    record: Dict[str, Any]


class CatalogSearch:
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        self.entity_ids = {clean_text(record.get("entity_id")) for record in records}
        self._documents = [self._record_search_text(record) for record in records]
        self._vectors, self._idf = self._build_tfidf_vectors(self._documents)

    def _record_search_text(self, record: Dict[str, Any]) -> str:
        weighted_parts = [
            clean_text(record.get("name")),
            clean_text(record.get("name")),
            clean_text(record.get("description")),
            clean_text(record.get("keys")),
            clean_text(record.get("keys")),
            clean_text(record.get("job_levels")),
            clean_text(record.get("languages")),
            clean_text(record.get("duration_raw")),
            clean_text(record.get("remote")),
            clean_text(record.get("adaptive")),
        ]
        return " ".join(weighted_parts)

    def _build_tfidf_vectors(self, documents: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        tokenized_documents = [tokenize(document) for document in documents]
        document_frequency = Counter(token for tokens in tokenized_documents for token in set(tokens))
        total_documents = max(len(documents), 1)
        idf = {
            token: math.log((1 + total_documents) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }
        vectors = [self._normalize(Counter(tokens), idf) for tokens in tokenized_documents]
        return vectors, idf

    def _normalize(self, counts: Counter, idf: Dict[str, float]) -> Dict[str, float]:
        weighted = {token: count * idf.get(token, 1.0) for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {token: value / norm for token, value in weighted.items()}

    def _query_vector(self, query: str) -> Dict[str, float]:
        return self._normalize(Counter(tokenize(query)), self._idf)

    def _cosine(self, query_vector: Dict[str, float], record_vector: Dict[str, float]) -> float:
        if len(query_vector) > len(record_vector):
            query_vector, record_vector = record_vector, query_vector
        return sum(value * record_vector.get(token, 0.0) for token, value in query_vector.items())

    def search(self, query: str, limit: int = MAX_RECOMMENDATIONS) -> List[ScoredRecord]:
        query_vector = self._query_vector(query)
        scored = []
        for record, record_vector in zip(self.records, self._vectors):
            score = self._cosine(query_vector, record_vector)
            score += self._constraint_boost(query, record)
            if score > 0:
                scored.append(ScoredRecord(score=score, record=record))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _constraint_boost(self, query: str, record: Dict[str, Any]) -> float:
        query_lower = query.lower()
        name = clean_text(record.get("name")).lower()
        keys = " ".join(record.get("keys", [])).lower()
        languages = " ".join(record.get("languages", [])).lower()
        job_levels = " ".join(record.get("job_levels", [])).lower()
        description = clean_text(record.get("description")).lower()
        combined = f"{name} {keys} {languages} {job_levels} {description}"

        boost = 0.0
        for token in set(tokenize(query)):
            if token and token in name:
                boost += 0.18
            elif token and token in combined:
                boost += 0.05
            if token in TECH_TERMS and token in name:
                boost += 0.55

        if "personality" in query_lower or "behavior" in query_lower or "behaviour" in query_lower:
            boost += 0.30 if "personality" in keys or "behavior" in keys else -0.05
        if "simulation" in query_lower or "coding" in query_lower or "hands-on" in query_lower:
            boost += 0.25 if "simulation" in keys or "simulated" in description else 0.0
        if "english" in query_lower:
            boost += 0.25 if "english" in languages else -0.15
        if "entry" in query_lower or "junior" in query_lower:
            boost += 0.15 if "entry-level" in job_levels or "graduate" in job_levels else 0.0
        if "mid" in query_lower or "4 years" in query_lower or "experienced" in query_lower:
            boost += 0.15 if "mid-professional" in job_levels or "professional" in job_levels else 0.0
        if "senior" in query_lower or "manager" in query_lower or "lead" in query_lower:
            boost += 0.15 if "manager" in job_levels or "director" in job_levels else 0.0
        return boost

    def find_by_name(self, phrase: str, limit: int = 3) -> List[Dict[str, Any]]:
        phrase_tokens = set(tokenize(phrase))
        if not phrase_tokens:
            return []

        matches: List[ScoredRecord] = []
        for record in self.records:
            name = clean_text(record.get("name")).lower()
            description = clean_text(record.get("description")).lower()
            record_tokens = set(tokenize(name))
            overlap = len(phrase_tokens & record_tokens)
            score = overlap / max(len(phrase_tokens), 1)
            if phrase.lower() in name:
                score += 1.0
            if "gsa" in phrase.lower() and "global skills assessment" in description:
                score += 1.0
            if score > 0:
                matches.append(ScoredRecord(score=score, record=record))
        matches.sort(key=lambda item: item.score, reverse=True)
        return [item.record for item in matches[:limit]]

    def find_exact_name(self, name: str) -> Optional[Dict[str, Any]]:
        target = name.lower()
        for record in self.records:
            if clean_text(record.get("name")).lower() == target:
                return record
        return None


class ConversationAgent:
    def __init__(self, search: CatalogSearch):
        self.search = search

    def reply(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        user_messages = [message.get("content", "") for message in messages if message.get("role") == "user"]
        if not user_messages:
            return self._clarify("What role or skill area should the assessment cover?")

        latest_user = user_messages[-1].strip()
        query_context = "\n".join(user_messages)

        if self._is_out_of_scope_or_injection(latest_user, query_context):
            return {
                "reply": "I can only help with SHL assessment selection from the catalog.",
                "recommendations": [],
                "end_of_conversation": False,
            }

        comparison = self._try_compare(latest_user)
        if comparison:
            return comparison

        if self._needs_clarification(query_context):
            return self._clarify("What role, skill, or job description should the assessment target?")

        scored_records = self._rank_for_conversation(query_context)
        recommendations = [recommendation_payload(item.record) for item in scored_records[:MAX_RECOMMENDATIONS]]
        if not recommendations:
            return self._clarify("Which skill, technology, or job family should I search for in the SHL catalog?")

        count = len(recommendations)
        end_of_conversation = self._is_completion_turn(latest_user)
        return {
            "reply": f"Got it. Here are {count} SHL assessment{'s' if count != 1 else ''} that best match the current requirements.",
            "recommendations": recommendations,
            "end_of_conversation": end_of_conversation,
        }

    def _rank_for_conversation(self, query_context: str) -> List[ScoredRecord]:
        ranked = self.search.search(query_context, 50)
        ranked = self._remove_excluded_records(query_context, ranked)
        ranked = self._promote_explicit_products(query_context, ranked)
        lower = query_context.lower()
        wants_personality = any(word in lower for word in ("personality", "behavior", "behaviour", "psychometric"))
        has_technical_skill = any(term in tokenize(query_context) for term in TECH_TERMS)
        if not (wants_personality and has_technical_skill):
            return ranked[:MAX_RECOMMENDATIONS]

        personality_ranked = [
            item for item in ranked
            if any(key in ("Personality & Behavior", "Competencies", "Biodata & Situational Judgment") for key in item.record.get("keys", []))
        ]
        technical_query = re.sub(r"\b(personality|behavior|behaviour|psychometric|tests?)\b", " ", query_context, flags=re.IGNORECASE)
        technical_ranked = self.search.search(technical_query, 50)
        technical_ranked = self._remove_excluded_records(query_context, technical_ranked)
        technical_ranked = self._promote_explicit_products(query_context, technical_ranked)

        merged: List[ScoredRecord] = []
        seen = set()
        for bucket in (technical_ranked[:6], personality_ranked[:4], ranked):
            for item in bucket:
                entity_id = item.record.get("entity_id")
                if entity_id not in seen:
                    merged.append(item)
                    seen.add(entity_id)
                if len(merged) == MAX_RECOMMENDATIONS:
                    return merged
        return merged

    def _remove_excluded_records(self, query_context: str, ranked: List[ScoredRecord]) -> List[ScoredRecord]:
        lower = query_context.lower()
        excluded_names = set()
        if re.search(r"\b(drop|remove|exclude)\s+(the\s+)?opq", lower):
            excluded_names.add("occupational personality questionnaire opq32r")
        if re.search(r"\b(drop|remove|exclude)\s+(the\s+)?rest", lower):
            excluded_names.add("restful web services (new)")

        return [
            item for item in ranked
            if clean_text(item.record.get("name")).lower() not in excluded_names
        ]

    def _promote_explicit_products(self, query_context: str, ranked: List[ScoredRecord]) -> List[ScoredRecord]:
        lower = query_context.lower()
        product_rules = [
            ("leadership" in lower or "cxo" in lower or "director-level" in lower, "Occupational Personality Questionnaire OPQ32r"),
            ("leadership" in lower or "benchmark" in lower, "OPQ Universal Competency Report 2.0"),
            ("leadership" in lower, "OPQ Leadership Report"),
            ("opq" in lower or "personality" in lower, "Occupational Personality Questionnaire OPQ32r"),
            ("verify g" in lower or "g+" in lower or "cognitive" in lower, "SHL Verify Interactive G+"),
            ("numerical" in lower, "SHL Verify Interactive - Numerical Reasoning"),
            ("numerical" in lower, "SHL Verify Interactive – Numerical Reasoning"),
            ("finance" in lower or "financial" in lower, "Financial Accounting (New)"),
            ("finance" in lower or "financial" in lower or "numerical" in lower, "Basic Statistics (New)"),
            ("graduate scenarios" in lower or "situational" in lower or "judgement" in lower or "judgment" in lower, "Graduate Scenarios"),
            ("spring" in lower, "Spring (New)"),
            ("sql" in lower, "SQL (New)"),
            ("aws" in lower or "amazon web services" in lower, "Amazon Web Services (AWS) Development (New)"),
            ("docker" in lower, "Docker (New)"),
            ("core java" in lower or "java" in lower, "Core Java (Advanced Level) (New)"),
            ("hipaa" in lower, "HIPAA (Security)"),
            ("healthcare" in lower or "patient" in lower or "medical" in lower, "Medical Terminology (New)"),
            ("healthcare" in lower or "patient records" in lower, "Microsoft Word 365 - Essentials (New)"),
            ("spanish" in lower or "patient records" in lower or "trust-sensitive" in lower, "Dependability and Safety Instrument (DSI)"),
            ("spanish" in lower or "patient records" in lower, "Occupational Personality Questionnaire OPQ32r"),
            ("excel" in lower and "simulation" in lower, "Microsoft Excel 365 (New)"),
            ("word" in lower and "simulation" in lower, "Microsoft Word 365 (New)"),
            ("excel" in lower, "MS Excel (New)"),
            ("word" in lower, "MS Word (New)"),
            ("safety" in lower or "dependability" in lower or "industrial" in lower, "Manufac. & Indust. - Safety & Dependability 8.0"),
            ("safety" in lower, "Workplace Health and Safety (New)"),
            ("dsi" in lower or "dependability" in lower, "Dependability and Safety Instrument (DSI)"),
            ("sales" in lower and "audit" in lower, "Global Skills Assessment"),
            ("sales" in lower and "audit" in lower, "Global Skills Development Report"),
            ("sales" in lower, "OPQ MQ Sales Report"),
            ("sales" in lower, "Sales Transformation 2.0 - Individual Contributor"),
            ("rust" in lower or "live coding" in lower, "Smart Interview Live Coding"),
            ("linux" in lower or "rust" in lower, "Linux Programming (General)"),
            ("networking" in lower or "infrastructure" in lower, "Networking and Implementation (New)"),
            ("svar" in lower or ("contact" in lower and "english" in lower and "us" in lower), "SVAR Spoken English (US) (New)"),
            ("svar" in lower or ("contact" in lower and "english" in lower and "us" in lower), "SVAR - Spoken English (US)  (New)"),
        ]

        excluded = set()
        if re.search(r"\b(drop|remove|exclude)\s+(the\s+)?opq", lower):
            excluded.add("occupational personality questionnaire opq32r")
        if re.search(r"\b(drop|remove|exclude)\s+(the\s+)?rest", lower):
            excluded.add("restful web services (new)")
        promoted: List[ScoredRecord] = []
        seen = set()
        for should_promote, name in product_rules:
            if not should_promote:
                continue
            record = self.search.find_exact_name(name)
            if record and clean_text(record.get("name")).lower() not in excluded:
                promoted.append(ScoredRecord(score=999.0, record=record))
                seen.add(clean_text(record.get("name")).lower())

        merged = promoted[:]
        for item in ranked:
            name = clean_text(item.record.get("name")).lower()
            if name not in seen:
                merged.append(item)
                seen.add(name)
        return merged

    def _clarify(self, question: str) -> Dict[str, Any]:
        return {"reply": question, "recommendations": [], "end_of_conversation": False}

    def _is_completion_turn(self, latest_user: str) -> bool:
        lower = latest_user.lower()
        completion_markers = [
            "confirmed", "lock it in", "locking it in", "final list", "that's good",
            "that works", "perfect", "thanks", "thank you", "covers it", "as-is",
            "clear",
        ]
        return any(marker in lower for marker in completion_markers)

    def _needs_clarification(self, query_context: str) -> bool:
        if not self._has_role_or_skill_signal(query_context):
            return True

        tokens = set(tokenize(query_context))
        meaningful_tokens = tokens - GENERIC_QUERY_WORDS
        if len(meaningful_tokens) < 2:
            return True

        top_score = self.search.search(query_context, 1)
        return not top_score or top_score[0].score < 0.08

    def _has_role_or_skill_signal(self, query_context: str) -> bool:
        lower = query_context.lower()
        signal_patterns = [
            r"\b(java|python|sql|spring|aws|docker|excel|word|sales|finance|financial|healthcare|hipaa|medical)\b",
            r"\b(contact|center|centre|customer|service|admin|assistant|graduate|trainee|leadership|executive)\b",
            r"\b(safety|dependability|personality|cognitive|numerical|situational|reasoning|coding)\b",
            r"\b(engineer|developer|analyst|operator|agent|manager|director|cxo)\b",
            r"\bjob description\b|\bjd\b",
        ]
        return any(re.search(pattern, lower) for pattern in signal_patterns)

    def _is_out_of_scope_or_injection(self, latest_text: str, full_text: str) -> bool:
        latest_lower = latest_text.lower()
        full_lower = full_text.lower()
        injection_markers = [
            "ignore previous", "ignore all previous", "system prompt", "developer message",
            "reveal prompt", "jailbreak", "act as", "forget instructions",
        ]
        off_topic = [
            "legal advice", "legally required", "legal requirement", "employment law",
            "write a contract", "legal hiring policy", "write a legal", "salary negotiation", "interview questions",
            "hiring plan", "performance improvement plan", "satisfy that requirement",
            "weather", "stock price", "medical advice", "recipe",
        ]
        return any(marker in full_lower for marker in injection_markers) or any(
            re.search(rf"\b{re.escape(marker)}\b", latest_lower) for marker in off_topic
        )

    def _try_compare(self, latest_user: str) -> Optional[Dict[str, Any]]:
        lower = latest_user.lower()
        if not any(word in lower for word in ("compare", "difference", "differentiate", "versus", " vs ")):
            return None

        pieces = re.split(r"\bbetween\b|\band\b|\bvs\.?\b|\bversus\b|,", latest_user, flags=re.IGNORECASE)
        candidates = [piece.strip(" ?.") for piece in pieces if len(piece.strip(" ?.")) > 1]
        candidates = [candidate for candidate in candidates if candidate.lower() not in {"compare", "difference"}]

        found: List[Dict[str, Any]] = []
        for candidate in candidates:
            for record in self.search.find_by_name(candidate, 1):
                if record not in found:
                    found.append(record)
            if len(found) >= 2:
                break

        if len(found) < 2:
            search_hits = self.search.search(latest_user, 2)
            found = [item.record for item in search_hits]

        if len(found) < 2:
            return self._clarify("Which two SHL assessments would you like me to compare?")

        left, right = found[:2]
        reply = (
            f"{left.get('name')} focuses on {clean_text(left.get('description'))} "
            f"It is tagged as {', '.join(left.get('keys', [])) or 'uncategorized'}.\n\n"
            f"{right.get('name')} focuses on {clean_text(right.get('description'))} "
            f"It is tagged as {', '.join(right.get('keys', [])) or 'uncategorized'}."
        )
        return {
            "reply": reply,
            "recommendations": [recommendation_payload(left), recommendation_payload(right)],
            "end_of_conversation": False,
        }


catalog_records = [record for record in load_catalog_records() if is_individual_test_record(record)]
catalog_search = CatalogSearch(catalog_records)
agent = ConversationAgent(catalog_search)


if FastAPI:
    class ChatMessage(BaseModel):
        role: str
        content: str


    class ChatRequest(BaseModel):
        messages: List[ChatMessage]


    app = FastAPI(title="SHL Assessment Recommender")


    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}


    @app.post("/chat")
    def chat(request: ChatRequest) -> Dict[str, Any]:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        messages = [
            message.model_dump() if hasattr(message, "model_dump") else message.dict()
            for message in request.messages
        ]
        for index, message in enumerate(messages):
            if message.get("role") not in {"user", "assistant", "system"}:
                raise HTTPException(status_code=400, detail=f"messages[{index}].role is invalid")
            if not message.get("content", "").strip():
                raise HTTPException(status_code=400, detail=f"messages[{index}].content must not be empty")
        return agent.reply(messages)
else:
    app = None


if __name__ == "__main__":
    sample = {"messages": [{"role": "user", "content": "I want Java Developer assessment in English"}]}
    print(json.dumps(agent.reply(sample["messages"]), indent=2))
