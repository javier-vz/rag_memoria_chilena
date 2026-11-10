"""Pruebas de la síntesis redactada y del control formal de evidencias."""
import json
from types import SimpleNamespace
from core.retrieval import CatalogSearch
from core.generation import answer_with_groq, make_context, validate_narrative


class SimulatedGroq:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))
        )])


def sample():
    hits = CatalogSearch().search('sufragio femenino', encoder=None, top_k=4)
    docs = make_context(hits)
    a, b = docs[:2]
    body = {
        'respuesta': (
            f"Para estudiar el sufragio femenino en Chile, destacan «{a['campos']['titulo']}» "
            f"[{a['mcid']}] y «{b['campos']['titulo']}» [{b['mcid']}]. "
            "Estas fichas permiten localizar obras vinculadas a la consulta, pero no sustituir su lectura."
        ),
        'evidencias': [
            {'mcid': a['mcid'], 'campo': 'titulo', 'fragmento': a['campos']['titulo']},
            {'mcid': b['mcid'], 'campo': 'titulo', 'fragmento': b['campos']['titulo']},
        ],
    }
    return hits, body


def test_narrative_accepts_bibliographic_text():
    hits, body = sample()
    text, evidences, errors = validate_narrative(body, make_context(hits))
    assert text and not errors and len(evidences) == 2
    groq = SimulatedGroq(body)
    response = answer_with_groq('sufragio femenino', hits, 'fake-key', client=groq)
    assert response.status == 'control_formal'
    assert groq.calls == 1
    assert response.text == text


def test_narrative_rejects_fabricated_source():
    hits, body = sample()
    body['respuesta'] += ' [MC9999999]'
    text, evidence, errors = validate_narrative(body, make_context(hits))
    assert not text and errors


def test_narrative_rejects_historical_inference():
    hits, body = sample()
    body['respuesta'] += ' Ese documento otorgó el derecho a votar a todas las mujeres.'
    text, evidence, errors = validate_narrative(body, make_context(hits))
    assert not text and errors
