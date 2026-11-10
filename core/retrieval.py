"""Recuperación reproducible sobre 34 fichas de Memoria Chilena.

Índices: TF-IDF léxico, MiniLM multilingüe precomputado y aristas curatoriales.
No atribuir los metadatos al contenido íntegro de los libros.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
STOP = set('a al de del la las los lo el en y e o que cual cuales sobre con por para su sus hacia otros otras un una unos unas se me interesa informacion fuentes documentan documentos documentales textos texto libros libro obras obra publicaciones publicacion que estan relacionados relacionan vinculados permiten investigar hablan tratan asociados tiene chile chilena chileno'.split())
ALIAS = {'quichua': 'quechua', 'piru': 'peru', 'yncas': 'incas', 'elecciones': 'eleccion', 'electorales': 'electoral', 'festejos': 'festejo', 'celebraciones': 'celebracion', 'inicio': 'inicios', 'sufrajio': 'sufragio'}


def tokens(texto: str) -> list[str]:
    v = unicodedata.normalize('NFKD', str(texto).lower())
    v = ''.join(c for c in v if not unicodedata.combining(c))
    return [ALIAS.get(w, w) for w in re.sub('[^a-z0-9]+', ' ', v).split() if w not in STOP and len(w) > 1]


def safe_str(v) -> str:
    return '' if pd.isna(v) else str(v).strip()


def label_fecha(raw: str) -> str:
    """La fecha no tiene por qué ser de publicación; no asumir su significado."""
    s = safe_str(raw)
    if re.fullmatch(r'\d{4}', s) and 1500 <= int(s) <= 2026:
        return s
    m = re.fullmatch(r'start=(\d{4});end=(\d{4})', s)
    if m and m[1] == m[2] and 1500 <= int(m[1]) <= 2026:
        return m[1]
    return ''


@dataclass
class Hit:
    doc: dict
    score: float
    lexical_score: float
    semantic_score: float | None
    anchor_score: float
    graph_related: list[dict]


class CatalogSearch:
    def __init__(self, base: Path = BASE):
        base = Path(base)
        data = pd.read_csv(base / 'data/catalogo.csv', dtype=str, keep_default_na=False)
        npz = np.load(base / 'data/embeddings.npz', allow_pickle=False)
        ids = npz['ids'].astype(str).tolist()
        assert len(ids) == len(set(ids)) == len(data), 'IDs de embeddings duplicados o incompletos'
        assert set(ids) == set(data.mcid), 'Los embeddings no corresponden al catálogo'
        assert str(npz['model_id']) == MODEL_ID, 'Modelo de embeddings inesperado'
        self.docs = data.set_index('mcid').loc[ids].reset_index()
        self.vectors = np.asarray(npz['vectors'], dtype='float32')
        assert self.vectors.shape == (len(ids), 384) and np.isfinite(self.vectors).all(), 'Dimensiones del índice inválidas'
        norms = np.linalg.norm(self.vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=.02), 'Embeddings no normalizados'
        self.ids = ids
        self.N = len(ids)
        normal = lambda s: ' '.join(tokens(s))
        self.word = TfidfVectorizer(preprocessor=normal, ngram_range=(1,2), sublinear_tf=True)
        self.chars = TfidfVectorizer(preprocessor=normal, analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True)
        corpus = self.docs.texto_lexico.tolist()
        self.xw = self.word.fit_transform(corpus)
        self.xc = self.chars.fit_transform(corpus)
        counts = Counter(t for title in self.docs.titulo for t in set(tokens(title)))
        self.idf = {w: math.log((self.N+1)/(v+1))+1 for w,v in counts.items()}
        self.default_idf = math.log(self.N+1)+1
        self.neighbors: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        edges = pd.read_csv(base / 'data/relaciones.csv', dtype=str, keep_default_na=False)
        for r in edges.itertuples(index=False):
            if r.mcid_1 not in ids or r.mcid_2 not in ids:
                raise ValueError('Una relación apunta fuera del catálogo')
            w = max(0., float(r.peso))
            self.neighbors[r.mcid_1].append((r.mcid_2,w,r.etiquetas))
            self.neighbors[r.mcid_2].append((r.mcid_1,w,r.etiquetas))
        self.edge_count = len(edges)

    def anchor(self, question: str, title: str) -> float:
        q,t=tokens(question),tokens(title)
        if not q or not t:return 0.0
        common=set(q)&set(t)
        weighted=lambda vals:sum(self.idf.get(w,self.default_idf) for w in vals)
        value=.65*weighted(common)/max(weighted(set(q)),1e-12)+.35*weighted(common)/max(weighted(set(t)),1e-12)
        bigrams=set(zip(t,t[1:]))
        if any((q[i],q[i+1]) in bigrams for i in range(len(q)-1)):value+=.18
        years=[x for x in q if re.fullmatch(r'(?:1[5-9]|20)\d\d',x)]
        if years and any(x in t for x in years) and len(common)>1:value+=.10
        return min(float(value),1.)

    def search(self, question: str, *, encoder=None, top_k: int=4, use_graph: bool=True) -> list[Hit]:
        if not question.strip():return []
        k=max(1,min(top_k,self.N))
        lex=.7*cosine_similarity(self.word.transform([question]),self.xw).ravel()+.3*cosine_similarity(self.chars.transform([question]),self.xc).ravel()
        sem=None
        if encoder is not None:
            query=np.asarray(encoder.encode([question], convert_to_numpy=True, normalize_embeddings=True),dtype='float32')[0]
            if query.shape != (384,) or not np.isfinite(query).all():raise ValueError('Embedding de consulta inválido')
            sem=self.vectors@query
            order_l=np.argsort(-lex,kind='stable')
            order_s=np.argsort(-sem,kind='stable')
            lr={int(i):r+1 for r,i in enumerate(order_l)}
            sr={int(i):r+1 for r,i in enumerate(order_s)}
            base=np.array([1/(60+sr[i])+(1/(60+lr[i]) if lex[i]>0 else 0) for i in range(self.N)])
        else:
            base=lex.copy()
        bmax=float(np.max(base))
        if bmax>0:base=base/bmax
        anc=np.array([self.anchor(question,t) for t in self.docs.titulo])
        raw=.52*base+.48*anc
        # Añadir una contribución pequeña SOLO por relaciones curatoriales específicas.
        if use_graph:
            index={mcid:i for i,mcid in enumerate(self.ids)}
            bonus=np.zeros(self.N)
            for mcid,related in self.neighbors.items():
                i=index[mcid]
                vals=[raw[index[n]] for n,_,_ in related if raw[index[n]] > .15]
                if vals and raw[i] > .08:bonus[i]=.035*max(vals)
            raw=raw+bonus
        order=sorted(range(self.N),key=lambda i:(-float(raw[i]),self.ids[i]))
        hits=[]
        for i in order[:k]:
            doc=self.docs.iloc[i].to_dict()
            similar=sorted(self.neighbors.get(doc['mcid'],[]),key=lambda v:-v[1])[:5]
            related=[{'mcid':n,'titulo':self.docs.iloc[self.ids.index(n)].titulo,'etiquetas':label} for n,_,label in similar]
            hits.append(Hit(doc=doc,score=float(raw[i]),lexical_score=float(lex[i]),semantic_score=float(sem[i]) if sem is not None else None,anchor_score=float(anc[i]),graph_related=related))
        return hits
