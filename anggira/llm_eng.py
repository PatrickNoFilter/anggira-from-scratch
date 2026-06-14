"""
Anggira LLM Engineering — RAG, Structured Outputs, Guardrails, Caching, Evaluation, MCP

Phase 15: LLM Engineering from Scratch

Implements:
- RAG pipeline: document chunking, TF-IDF embedding, retrieval, generation
- Structured Outputs: JSON schema validation with constrained generation
- Embeddings: TF-IDF vectorizer, cosine similarity search
- Guardrails: topic classifier, PII detection (regex-based), content safety filter
- Prompt Caching: LRU cache for LLM responses with TTL
- Evaluation: accuracy, precision, recall, F1, BLEU score, perplexity
- MCP client: basic Model Context Protocol tool invocation simulation
"""

import json
import math
import random
import re
import time
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════
# UTILITY: Mock LLM
# ════════════════════════════════════════════════════════════════

class MockLLM:
    """A mock LLM that returns canned responses for testing.

    In a real system this would call an external API. Here we simulate
    generation for demo and test purposes.
    """

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}

    def generate(self, prompt: str, **kwargs) -> str:
        """Return a canned response or a generic reply."""
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        return f"MockLLM response to: {prompt[:60]}..."


# ════════════════════════════════════════════════════════════════
# EMBEDDINGS: TF-IDF Vectorizer & Cosine Similarity
# ════════════════════════════════════════════════════════════════

class TfidfVectorizer:
    """Simple TF-IDF vectorizer built from scratch.

    Attributes:
        idf: dict mapping term -> inverse document frequency
        vocabulary: dict mapping term -> column index
    """

    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.vocabulary: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split on non-alphanumeric characters."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def fit(self, documents: List[str]) -> 'TfidfVectorizer':
        """Fit IDF weights from a corpus of documents."""
        N = len(documents)
        df: Dict[str, int] = {}
        for doc in documents:
            terms = set(self._tokenize(doc))
            for term in terms:
                df[term] = df.get(term, 0) + 1
        # Build vocabulary sorted for determinism
        sorted_terms = sorted(df.keys())
        self.vocabulary = {t: i for i, t in enumerate(sorted_terms)}
        self.idf = {t: math.log((N + 1) / (f + 1)) + 1.0
                    for t, f in df.items()}
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents to TF-IDF vectors."""
        if not self.vocabulary:
            raise ValueError("Vectorizer not fitted — call fit() first")
        dim = len(self.vocabulary)
        vectors = []
        for doc in documents:
            terms = self._tokenize(doc)
            tf = Counter(terms)
            vec = [0.0] * dim
            for term, count in tf.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    vec[idx] = count * self.idf.get(term, 1.0)
            vectors.append(vec)
        return vectors

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ════════════════════════════════════════════════════════════════

@dataclass
class Document:
    """A document with metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None


class DocumentChunker:
    """Split documents into overlapping chunks.

    Supports character-level and sentence-level chunking with
    configurable overlap.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks by character length.

        Tries to break at sentence boundaries when possible.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Try to break at sentence end  (. ! ?)
            if end < len(text):
                # Search backward for a sentence boundary
                search_start = max(start, end - self.chunk_size // 2)
                sentence_end = -1
                for m in re.finditer(r'[.!?]\s', text[search_start:end]):
                    sentence_end = search_start + m.end()
                if sentence_end > start:
                    end = sentence_end
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.overlap if end < len(text) else len(text)
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split multiple documents into chunked documents."""
        chunked = []
        for doc in documents:
            chunks = self.chunk_text(doc.text)
            for i, chunk_text in enumerate(chunks):
                chunked.append(Document(
                    text=chunk_text,
                    metadata={**doc.metadata, 'chunk_index': i,
                              'num_chunks': len(chunks)},
                    doc_id=f"{doc.doc_id or 'doc'}#chunk{i}"
                ))
        return chunked


class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self):
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []

    def add(self, documents: List[Document],
            vectors: List[List[float]]) -> None:
        """Add documents and their vectors to the store."""
        self.documents.extend(documents)
        self.vectors.extend(vectors)

    def search(self, query_vector: List[float], k: int = 3
               ) -> List[Tuple[Document, float]]:
        """Return top-k most similar documents with scores."""
        if not self.vectors:
            return []
        scored = [(doc, cosine_similarity(query_vector, vec))
                  for doc, vec in zip(self.documents, self.vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    End-to-end: chunk → embed → retrieve → generate.
    """

    def __init__(self, llm: Optional[MockLLM] = None,
                 chunker: Optional[DocumentChunker] = None,
                 vectorizer: Optional[TfidfVectorizer] = None,
                 vector_store: Optional[VectorStore] = None):
        self.llm = llm or MockLLM()
        self.chunker = chunker or DocumentChunker()
        self.vectorizer = vectorizer or TfidfVectorizer()
        self.store = vector_store or VectorStore()
        self._indexed = False

    def index(self, documents: List[Document]) -> None:
        """Chunk, embed, and store documents."""
        chunked = self.chunker.chunk_documents(documents)
        texts = [d.text for d in chunked]
        vectors = self.vectorizer.fit_transform(texts)
        self.store.add(chunked, vectors)
        self._indexed = True

    def retrieve(self, query: str, k: int = 3
                 ) -> List[Tuple[Document, float]]:
        """Retrieve top-k relevant documents for a query."""
        query_vec = self.vectorizer.transform([query])[0]
        return self.store.search(query_vec, k=k)

    def answer(self, query: str, k: int = 3,
               system_prompt: str = "") -> str:
        """Retrieve documents and generate an answer."""
        if not self._indexed:
            return "No documents indexed. Call index() first."

        results = self.retrieve(query, k=k)
        if not results:
            return "No relevant documents found."

        context = "\n\n".join(
            f"[Doc {i + 1}] {doc.text}"
            for i, (doc, _) in enumerate(results)
        )
        prompt = (f"{system_prompt}\n\n"
                  f"Context:\n{context}\n\n"
                  f"Question: {query}\n\n"
                  f"Answer based on the context above:")
        return self.llm.generate(prompt)

    def generate(self, prompt: str) -> str:
        """Direct generation without retrieval."""
        return self.llm.generate(prompt)


# ════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUTS: JSON Schema Validation & Constrained Gen
# ════════════════════════════════════════════════════════════════

class SchemaValidator:
    """Validate Python dicts against a JSON Schema-like specification.

    Supports: type, properties (for objects), items (for arrays),
    required, enum, minimum, maximum, minLength, maxLength, pattern.
    """

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate(self, data: Any, path: str = "$") -> List[str]:
        """Validate data against the schema. Return list of errors (empty = valid)."""
        errors = []
        self._validate(self.schema, data, path, errors)
        return errors

    def _validate(self, schema: Dict[str, Any], data: Any,
                  path: str, errors: List[str]) -> None:
        schema_type = schema.get("type")

        # null / None check
        if data is None:
            if schema_type is not None:
                errors.append(f"{path}: expected {schema_type}, got null")
            return

        # type check
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int),
            "number": lambda v: isinstance(v, (int, float)),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        if schema_type in type_checks and not type_checks[schema_type](data):
            errors.append(f"{path}: expected {schema_type}, got {type(data).__name__}")
            return

        # enum
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"{path}: value {data!r} not in enum {schema['enum']}")

        # string constraints
        if schema_type == "string" and isinstance(data, str):
            if "minLength" in schema and len(data) < schema["minLength"]:
                errors.append(f"{path}: length {len(data)} < minLength {schema['minLength']}")
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                errors.append(f"{path}: length {len(data)} > maxLength {schema['maxLength']}")
            if "pattern" in schema and not re.search(schema["pattern"], data):
                errors.append(f"{path}: does not match pattern {schema['pattern']!r}")

        # number constraints
        if schema_type in ("integer", "number") and isinstance(data, (int, float)):
            if "minimum" in schema and data < schema["minimum"]:
                errors.append(f"{path}: {data} < minimum {schema['minimum']}")
            if "maximum" in schema and data > schema["maximum"]:
                errors.append(f"{path}: {data} > maximum {schema['maximum']}")

        # array items
        if schema_type == "array" and isinstance(data, list):
            if "items" in schema:
                for i, item in enumerate(data):
                    self._validate(schema["items"], item, f"{path}[{i}]", errors)

        # object properties
        if schema_type == "object" and isinstance(data, dict):
            required = schema.get("required", [])
            for key in required:
                if key not in data:
                    errors.append(f"{path}: missing required key {key!r}")
            props = schema.get("properties", {})
            for key, value in data.items():
                if key in props:
                    self._validate(props[key], value, f"{path}.{key}", errors)
                elif "additionalProperties" in schema and not schema["additionalProperties"]:
                    errors.append(f"{path}: unexpected key {key!r}")

    def is_valid(self, data: Any) -> bool:
        """Quick validity check."""
        return len(self.validate(data)) == 0


class StructuredGenerator:
    """Generate structured (typed) outputs by constraining a mock LLM.

    Uses schema-based validation to ensure outputs conform to a schema.
    """

    def __init__(self, llm: Optional[MockLLM] = None):
        self.llm = llm or MockLLM()

    def generate_structured(self, prompt: str, schema: Dict[str, Any],
                            max_attempts: int = 3) -> Tuple[Any, List[str]]:
        """Generate and validate a structured output against a schema.

        Returns:
            Tuple of (parsed_data, errors). If validation passes, errors is empty.
        """
        schema_hint = json.dumps(schema, indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"Respond with valid JSON matching this schema:\n"
            f"{schema_hint}\n\n"
            f"JSON output:"
        )

        for attempt in range(max_attempts):
            raw = self.llm.generate(full_prompt)
            # Try to extract JSON from the response
            json_str = self._extract_json(raw)
            if json_str is None:
                continue
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            validator = SchemaValidator(schema)
            errors = validator.validate(data)
            if not errors:
                return data, []
            # If we have attempts left, ask for correction
            full_prompt = (
                f"Fix the following JSON to match the schema.\n\n"
                f"Schema:\n{schema_hint}\n\n"
                f"Previous response had errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\nCorrected JSON:"
            )

        return data if 'data' in locals() else None, \
               errors if 'errors' in locals() else ["max attempts exceeded"]

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract a JSON object or array from text."""
        # Try to find a JSON block between ```json and ```
        m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if m:
            return m.group(1).strip()
        # Try to find a top-level { ... } or [ ... ]
        for bracket in ('{', '['):
            start = text.find(bracket)
            if start >= 0:
                end = text.rfind('}' if bracket == '{' else ']')
                if end > start:
                    return text[start:end + 1]
        return None


# ════════════════════════════════════════════════════════════════
# GUARDRAILS: Topic Classification, PII Detection, Content Safety
# ════════════════════════════════════════════════════════════════

@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class TopicClassifier:
    """Simple keyword-based topic classifier for guardrailing.

    Classifies text into predefined topics using keyword matching.
    Can be used to block or flag off-topic requests.
    """

    def __init__(self):
        # topic -> set of keywords
        self.topics: Dict[str, set] = {
            "technical": {"code", "algorithm", "function", "api", "software",
                          "programming", "python", "data", "model", "system"},
            "medical": {"diagnosis", "treatment", "patient", "symptom",
                        "disease", "drug", "therapy", "clinical", "healthcare"},
            "financial": {"stock", "investment", "trading", "portfolio",
                          "bank", "loan", "interest", "dividend", "market"},
            "creative": {"poem", "story", "write", "creative", "art",
                         "fiction", "narrative", "character", "plot"},
            "harmful": {"hack", "exploit", "attack", "malware", "virus",
                        "bomb", "weapon", "illegal", "drugs", "violence"},
        }

    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify text into topics with confidence scores."""
        text_lower = text.lower()
        scores = []
        for topic, keywords in self.topics.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                confidence = matches / len(keywords)
                scores.append((topic, confidence))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def is_topical(self, text: str, allowed_topics: List[str],
                   threshold: float = 0.05) -> GuardrailResult:
        """Check if text falls within allowed topics."""
        scores = self.classify(text)
        if not scores:
            # No topic detected — allow by default
            return GuardrailResult(True, "no topic detected")
        top_topic, top_score = scores[0]
        if top_topic in allowed_topics or top_score < threshold:
            return GuardrailResult(True)
        return GuardrailResult(
            False,
            f"Topic '{top_topic}' not in allowed: {allowed_topics}",
            {"detected_topic": top_topic, "score": top_score}
        )


class PIIDetector:
    """Regex-based PII (Personally Identifiable Information) detection.

    Detects emails, phone numbers, SSNs, credit cards, IP addresses,
    and other common PII patterns.
    """

    def __init__(self):
        self.patterns: Dict[str, str] = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            "api_key": r'\b(?:sk-|pk-|api-)[a-zA-Z0-9]{16,}\b',
        }

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect all PII patterns in text.

        Returns:
            dict mapping pattern name -> list of matched strings
        """
        findings: Dict[str, List[str]] = {}
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[name] = matches
        return findings

    def check(self, text: str, action: str = "block"
              ) -> GuardrailResult:
        """Check text for PII.

        Args:
            text: Text to check
            action: 'block' or 'warn'

        Returns:
            GuardrailResult with passed=False if PII detected and action='block'
        """
        findings = self.detect(text)
        if not findings:
            return GuardrailResult(True)
        if action == "block":
            return GuardrailResult(
                False,
                f"PII detected: {list(findings.keys())}",
                {"pii_findings": findings}
            )
        # warn mode — return details but pass
        return GuardrailResult(
            True,
            f"PII detected (warn only): {list(findings.keys())}",
            {"pii_findings": findings}
        )


class ContentSafetyFilter:
    """Filter content for safety violations.

    Blocks or flags content containing hate speech, toxicity,
    or other harmful material using keyword/regex patterns.
    """

    def __init__(self):
        self.harmful_patterns: Dict[str, List[str]] = {
            "hate_speech": [
                r'\b(?:hate|racist|bigot|nazi)\b',
            ],
            "toxicity": [
                r'\b(?:kill\s+yourself|die|stupid)\b',
            ],
            "nsfw": [
                r'\b(?:nsfw|porn|explicit)\b',
            ],
        }

    def check(self, text: str) -> GuardrailResult:
        """Check text for harmful content."""
        text_lower = text.lower()
        violations: Dict[str, List[str]] = {}
        for category, patterns in self.harmful_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    violations.setdefault(category, []).extend(matches)
        if violations:
            return GuardrailResult(
                False,
                f"Content safety violation: {list(violations.keys())}",
                {"violations": violations}
            )
        return GuardrailResult(True, "Content passed safety check")


class GuardrailPipeline:
    """Composite guardrail pipeline that runs all checks."""

    def __init__(self, topic_classifier: Optional[TopicClassifier] = None,
                 pii_detector: Optional[PIIDetector] = None,
                 safety_filter: Optional[ContentSafetyFilter] = None):
        self.topic_classifier = topic_classifier or TopicClassifier()
        self.pii_detector = pii_detector or PIIDetector()
        self.safety_filter = safety_filter or ContentSafetyFilter()

    def check_input(self, text: str,
                    allowed_topics: Optional[List[str]] = None
                    ) -> List[GuardrailResult]:
        """Run all guardrails on input text."""
        results = []
        # Topic check
        if allowed_topics:
            results.append(
                self.topic_classifier.is_topical(text, allowed_topics)
            )
        # PII check
        results.append(self.pii_detector.check(text, action="block"))
        # Safety check
        results.append(self.safety_filter.check(text))
        return results

    def check_output(self, text: str) -> List[GuardrailResult]:
        """Run output guardrails (PII leak, safety)."""
        return [
            self.pii_detector.check(text, action="block"),
            self.safety_filter.check(text),
        ]


# ════════════════════════════════════════════════════════════════
# PROMPT CACHING: LRU Cache with TTL
# ════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """A cached LLM response with metadata."""
    response: str
    timestamp: float
    ttl: float  # seconds


class PromptCache:
    """LRU cache for LLM responses with time-to-live eviction.

    Attributes:
        capacity: max number of entries in cache
        default_ttl: default time-to-live in seconds
    """

    def __init__(self, capacity: int = 128, default_ttl: float = 300.0):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def _is_expired(self, entry: CacheEntry) -> bool:
        return time.time() - entry.timestamp > entry.ttl

    def get(self, key: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return entry.response

    def set(self, key: str, response: str,
            ttl: Optional[float] = None) -> None:
        """Cache a response with optional custom TTL."""
        # Evict if at capacity
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)  # remove LRU
        self._cache[key] = CacheEntry(
            response=response,
            timestamp=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl,
        )

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        now = time.time()
        active = sum(
            1 for e in self._cache.values()
            if now - e.timestamp <= e.ttl
        )
        expired = len(self._cache) - active
        return {
            "capacity": self.capacity,
            "active": active,
            "expired": expired,
            "total": len(self._cache),
            "default_ttl": self.default_ttl,
        }


class CachedLLM:
    """LLM wrapper with automatic prompt caching."""

    def __init__(self, llm: Optional[MockLLM] = None,
                 cache: Optional[PromptCache] = None):
        self.llm = llm if llm is not None else MockLLM()
        self.cache = cache if cache is not None else PromptCache()
        self._hits = 0
        self._misses = 0

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate with caching — returns cached response if available."""
        cache_key = prompt.strip()
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        response = self.llm.generate(prompt, **kwargs)
        self.cache.set(cache_key, response)
        return response

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cache_info": self.cache.stats(),
        }


# ════════════════════════════════════════════════════════════════
# EVALUATION: Metrics for LLM Outputs
# ════════════════════════════════════════════════════════════════

class Metrics:
    """Evaluation metrics for classification, generation, and language modeling.

    Implements:
    - Accuracy, Precision, Recall, F1 (classification)
    - BLEU score (generation)
    - Perplexity (language modeling)
    """

    @staticmethod
    def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
        """Compute accuracy score."""
        if not y_true or len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must be non-empty same-length lists")
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    @staticmethod
    def confusion_matrix(y_true: List[Any], y_pred: List[Any],
                         labels: Optional[List[Any]] = None
                         ) -> Dict[str, Dict[str, int]]:
        """Compute confusion matrix.

        Returns dict mapping actual -> {predicted: count}.
        """
        classes = labels or sorted(set(y_true) | set(y_pred))
        cm: Dict[str, Dict[str, int]] = {}
        for actual in classes:
            cm[str(actual)] = {}
            for predicted in classes:
                cm[str(actual)][str(predicted)] = 0
        for a, p in zip(y_true, y_pred):
            cm[str(a)][str(p)] = cm[str(a)].get(str(p), 0) + 1
        return cm

    @staticmethod
    def precision_recall_f1(y_true: List[Any], y_pred: List[Any],
                            pos_label: Any = 1) -> Dict[str, float]:
        """Compute precision, recall, and F1 for binary classification."""
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def f1_score(y_true: List[Any], y_pred: List[Any],
                 average: str = "binary",
                 pos_label: Any = 1) -> float:
        """Compute F1 score.

        Args:
            average: 'binary' for binary, 'macro' for multi-class
        """
        if average == "binary":
            return Metrics.precision_recall_f1(y_true, y_pred, pos_label)["f1"]

        # Macro F1
        classes = sorted(set(y_true) | set(y_pred))
        f1_scores = []
        for cls in classes:
            # One-vs-rest
            y_true_bin = [1 if t == cls else 0 for t in y_true]
            y_pred_bin = [1 if p == cls else 0 for p in y_pred]
            metrics = Metrics.precision_recall_f1(y_true_bin, y_pred_bin, pos_label=1)
            f1_scores.append(metrics["f1"])
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    @staticmethod
    def bleu(reference: str, candidate: str, max_n: int = 4) -> float:
        """Compute BLEU score (unigram to 4-gram, with brevity penalty).

        Args:
            reference: reference text
            candidate: candidate (generated) text
            max_n: max n-gram order (default 4)

        Returns:
            BLEU score between 0 and 1
        """
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()

        if len(cand_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        # Compute n-gram precision
        log_precisions = 0.0
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(
                tuple(ref_tokens[i:i + n])
                for i in range(len(ref_tokens) - n + 1)
            )
            cand_ngrams = Counter(
                tuple(cand_tokens[i:i + n])
                for i in range(len(cand_tokens) - n + 1)
            )

            matches = sum(min(cand_ngrams[ng], ref_ngrams.get(ng, 0))
                          for ng in cand_ngrams)
            total = max(sum(cand_ngrams.values()), 1)
            precision = matches / total

            if precision == 0:
                return 0.0
            log_precisions += (1.0 / max_n) * math.log(precision)

        # Brevity penalty
        ref_len = len(ref_tokens)
        cand_len = len(cand_tokens)
        if cand_len < ref_len:
            bp = math.exp(1.0 - ref_len / max(cand_len, 1))
        else:
            bp = 1.0

        return bp * math.exp(log_precisions)

    @staticmethod
    def perplexity(probabilities: List[float]) -> float:
        """Compute perplexity from a list of token probabilities.

        Args:
            probabilities: list of token probabilities (each > 0)

        Returns:
            Perplexity score (lower is better)
        """
        if not probabilities or any(p <= 0 for p in probabilities):
            return float('inf')
        log_sum = sum(math.log(p) for p in probabilities)
        return math.exp(-log_sum / len(probabilities))


# ════════════════════════════════════════════════════════════════
# MCP CLIENT: Model Context Protocol — Tool Invocation Simulation
# ════════════════════════════════════════════════════════════════

@dataclass
class MCPTool:
    """A tool exposed via the Model Context Protocol."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Callable[..., Any]

    def __call__(self, **kwargs) -> Any:
        return self.handler(**kwargs)


class MCPServer:
    """Simulated MCP server exposing tools.

    In a real MCP setup, this would run as a separate process and
    communicate via JSON-RPC over stdio. Here we simulate the
    tool registry and invocation for demonstration purposes.
    """

    def __init__(self, name: str = "demo-mcp-server", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._tools: Dict[str, MCPTool] = {}

    def register_tool(self, name: str, description: str,
                      parameters: Dict[str, Any],
                      handler: Callable[..., Any]) -> 'MCPServer':
        """Register a tool with the server."""
        self._tools[name] = MCPTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )
        return self

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions (equivalent to MCP tools/list)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]
                  ) -> Dict[str, Any]:
        """Call a tool (equivalent to MCP tools/call)."""
        if name not in self._tools:
            return {"error": f"Tool '{name}' not found",
                    "available": list(self._tools.keys())}
        tool = self._tools[name]
        try:
            result = tool(**arguments)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}


class MCPClient:
    """MCP client that connects to an MCPServer and invokes tools.

    Simulates the client side of the Model Context Protocol.
    """

    def __init__(self, server: Optional[MCPServer] = None):
        self.server = server or MCPServer()

    def discover_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools (tools/list)."""
        return self.server.list_tools()

    def invoke(self, tool_name: str,
               arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool (tools/call)."""
        return self.server.call_tool(tool_name, arguments)

    def build_llm_request(self, user_query: str) -> Dict[str, Any]:
        """Build an MCP-formatted request for the LLM to call tools.

        Returns a dict suitable for turning into an LLM prompt.
        """
        tools = self.discover_tools()
        return {
            "query": user_query,
            "available_tools": tools,
            "instruction": (
                "You have access to the tools above. "
                "To invoke a tool, respond with:\n"
                '```tool_call\n{"tool": "<name>", '
                '"arguments": {...}}\n```'
            ),
        }


# ════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════

def demo() -> None:
    """Demonstrate all LLM Engineering components."""
    print("=" * 68)
    print("  LLM ENGINEERING MODULE — Full Demo")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 1. EMBEDDINGS & SIMILARITY
    # ------------------------------------------------------------------
    print("\n" + "─" * 68)
    print("  1. EMBEDDINGS: TF-IDF & Cosine Similarity")
    print("─" * 68)

    docs = [
        "Python is a popular programming language for data science",
        "Machine learning models learn patterns from data",
        "The transformer architecture revolutionized natural language processing",
        "Deep neural networks consist of multiple layers of neurons",
        "Reinforcement learning trains agents through reward signals",
    ]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)

    query = "neural network language model"
    q_vec = vectorizer.transform([query])[0]

    scored = [(docs[i], cosine_similarity(q_vec, vectors[i]))
              for i in range(len(docs))]
    scored.sort(key=lambda x: x[1], reverse=True)

    print(f"  Query: \"{query}\"")
    for text, score in scored:
        print(f"    {score:.4f}  {text[:60]}")

    # ------------------------------------------------------------------
    # 2. RAG PIPELINE
    # ------------------------------------------------------------------
    print("\n" + "─" * 68)
    print("  2. RAG: Document Indexing & Retrieval-Augmented Generation")
    print("─" * 68)

    rag_llm = MockLLM(responses={
        "Answer based on the context": (
            "Anguilla is a British Overseas Territory in the Caribbean. "
            "It is known for its coral reefs and luxury resorts."
        ),
    })
    rag = RAGPipeline(llm=rag_llm)

    rag.index([
        Document("Anguilla is a British Overseas Territory in the Caribbean. "
                 "The island is known for its white sand beaches and "
                 "coral reefs. Tourism is the main industry.",
                 doc_id="anguilla-1"),
        Document("Greece is a country in southeastern Europe with thousands "
                 "of islands throughout the Aegean and Ionian seas. "
                 "Athens is the capital city.",
                 doc_id="greece-1"),
    ])

    results = rag.retrieve("What is Anguilla known for?")
    print(f"  Retrieved {len(results)} documents:")
    for doc, score in results:
        print(f"    [{score:.4f}] {doc.text[:60]}...")

    answer = rag.answer("What is Anguilla known for?")
    print(f"  Generated answer: {answer}")

    # ------------------------------------------------------------------
    # 3. STRUCTURED OUTPUTS
    # ------------------------------------------------------------------
    print("\n" + "─" * 68)
    print("  3. STRUCTURED OUTPUTS: Schema Validation & Constrained Gen")
    print("─" * 68)

    schema = {
        "type": "object",
        "required": ["name", "age", "role"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "role": {
                "type": "string",
                "enum": ["admin", "user", "moderator"]
            },
            "email": {"type": "string", "pattern": r"@.*\."},
        },
    }

    # Valid data
    validator = SchemaValidator(schema)
    valid_data = {"name": "Alice", "age": 30,
                  "role": "admin", "email": "alice@example.com"}
    v_errors = validator.validate(valid_data)
    print(f"  Valid data errors: {v_errors}")

    # Invalid data
    invalid_data = {"name": "", "age": 200, "role": "unknown"}
    iv_errors = validator.validate(invalid_data)
    print(f"  Invalid data errors: {len(iv_errors)}")
    for e in iv_errors:
        print(f"    - {e}")

    # Structured generation
    sg = StructuredGenerator()
    struct_result, struct_errors = sg.generate_structured(
        "Create a user profile for Bob who is 25 and a moderator.",
        schema,
    )
    print(f"  Structured generation result: {struct_result}")
    if struct_errors:
        print(f"  Errors: {struct_errors}")

    # ------------------------------------------------------------------
    # 4. GUARDRAILS
    # ------------------------------------------------------------------
    print("\n" + "─" * 68)
    print("  4. GUARDRAILS: Topic Classifier, PII Detection, Safety Filter")
    print("─" * 68)

    guard = GuardrailPipeline()

    # Topic check
    topic_result = guard.topic_classifier.is_topical(
        "Write a poem about a cat.",
        allowed_topics=["technical"]
    )
    print(f"  Topic guardrail (poem vs technical): "
          f"{'PASS' if topic_result.passed else 'BLOCK'}")
    print(f"    Reason: {topic_result.reason}")

    # PII check
    pii_result = guard.pii_detector.check(
        "Contact me at alice@example.com or call 555-123-4567"
    )
    print(f"  PII guardrail: "
          f"{'PASS' if pii_result.passed else 'BLOCK'}")
    print(f"    Reason: {pii_result.reason}")

    # Safety check
    safe_result = guard.safety_filter.check("I love programming in Python!")
    unsafe_result = guard.safety_filter.check(
        "I hate you, you are so stupid"
    )
    print(f"  Safety (safe): {'PASS' if safe_result.passed else 'BLOCK'}")
    print(f"  Safety (unsafe): "
          f"{'PASS' if unsafe_result.passed else 'BLOCK'}")
    print(f"    Reason: {unsafe_result.reason}")

    # ------------------------------------------------------------------
    # 5. PROMPT CACHING
    # ------------------------------------------------------------------
    print("\n" + "─" * 68)
    print("  5. PROMPT CACHING: LRU Cache with TTL")
    print("─" * 68)

    cache = PromptCache(capacity=3, default_ttl=60.0)
    cllm = CachedLLM(llm=MockLLM(), cache=cache)

    response1 = cllm.generate("Hello, how are you?")
    response2 = cllm.generate("Hello, how are you?")  # cache hit
    response3 = cllm.generate("What is the capital of France?")

    stats = cllm.cache_stats()
    print(f"  Cache hit rate: {stats['hit_rate']:.0%} "
          f"({stats['hits']} hits, {stats['misses']} misses)")
    print(f"  Cache active entries: {stats['cache_info']['active']}")
    print(f"  Same prompt returned same response: "
          f"{response1 == response2}")

    # Test LRU eviction
    cllm.generate("Third prompt")
    cllm.generate("Fourth prompt — evicts LRU")
    print(f"  After eviction: {cllm.cache.stats()['active']} active entries")

    # ------------------------------------------------------------------
    # 6. EVALUATION METRICS
    # ------------------------------------------------------------------
    print("\n" + "─" * 68)
    print("  6. EVALUATION: Accuracy, Precision, Recall, F1, BLEU, Perplexity")
    print("─" * 68)

    # Classification
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
    acc = Metrics.accuracy(y_true, y_pred)
    prf = Metrics.precision_recall_f1(y_true, y_pred)
    f1 = Metrics.f1_score(y_true, y_pred)
    f1_macro = Metrics.f1_score(y_true, y_pred, average="macro")

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prf['precision']:.4f}")
    print(f"  Recall:    {prf['recall']:.4f}")
    print(f"  F1 (bin):  {prf['f1']:.4f}")
    print(f"  F1 (macro):{f1_macro:.4f}")

    # BLEU
    reference = "the cat sat on the mat"
    candidate = "the cat sat on a mat"
    bleu_score = Metrics.bleu(reference, candidate)
    print(f"\n  BLEU score:")
    print(f"    Reference: {reference}")
    print(f"    Candidate: {candidate}")
    print(f"    BLEU: {bleu_score:.4f}")

    # Perplexity
    probs = [0.8, 0.9, 0.7, 0.95, 0.85]
    ppl = Metrics.perplexity(probs)
    print(f"\n  Perplexity (token probs): {ppl:.4f}")

    # ------------------------------------------------------------------
    # 7. MCP CLIENT
    # ------------------------------------------------------------------
    print("\n" + "─" * 68)
    print("  7. MCP: Model Context Protocol — Tool Invocation")
    print("─" * 68)

    # Create server and register tools
    server = MCPServer(name="demo-server", version="1.0.0")

    def _add(a: int, b: int) -> int:
        return a + b

    def _get_weather(city: str) -> str:
        weather_data = {
            "new york": "Sunny, 22°C",
            "london": "Rainy, 15°C",
            "tokyo": "Cloudy, 18°C",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")

    server.register_tool(
        "add", "Add two numbers",
        {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}
        }, "required": ["a", "b"]},
        _add,
    )
    server.register_tool(
        "get_weather", "Get weather for a city",
        {"type": "object", "properties": {
            "city": {"type": "string"}
        }, "required": ["city"]},
        _get_weather,
    )

    client = MCPClient(server)

    # Discover tools
    tools = client.discover_tools()
    print(f"  Available tools ({len(tools)}):")
    for t in tools:
        print(f"    - {t['name']}: {t['description']}")

    # Invoke tools
    add_result = client.invoke("add", {"a": 5, "b": 3})
    weather_result = client.invoke("get_weather", {"city": "London"})
    print(f"  add(5, 3) = {add_result}")
    print(f"  get_weather('London') = {weather_result}")

    # Build LLM request
    llm_req = client.build_llm_request("What is the weather in Tokyo?")
    print(f"  LLM request built with {len(llm_req['available_tools'])} tools")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  DEMO COMPLETE — All 7 components demonstrated successfully!")
    print("=" * 68)


if __name__ == '__main__':
    demo()
