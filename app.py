"""Buscador personal de fuentes históricas de Memoria Chilena."""
from __future__ import annotations
import os
import streamlit as st
from core.retrieval import CatalogSearch, MODEL_ID, label_fecha
from core.generation import answer_with_groq

st.set_page_config(page_title="Explorar Memoria Chilena", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
/* Contraste accesible, sin imitar la identidad del portal de origen. */
:root { --navy:#172e42; --blue:#165679; --blue-light:#e7f0f5; --ink:#253646; --sub:#52687a; --line:#cbd9e3; }
[data-testid="stAppViewContainer"] { background: #fafbfc; }
[data-testid="stHeader"] { background: #fafbfc; }
section.main > div.block-container { max-width: 1060px; padding-top: 2.25rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background:#edf3f7; }
h1,h2,h3 { color: var(--navy); letter-spacing:-.02em; }
h1 { font-size: clamp(1.75rem,3.5vw,2.3rem) !important; font-weight:710!important; }
h2 { font-size: 1.26rem !important; font-weight:660 !important; }
a {color:var(--blue)!important; text-decoration-thickness:1.5px;}
a:hover {color:#0a3e5e!important;}
.hero-label {color:#39708d; font-size:.78rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase;}
.hero-sub {color:var(--sub); font-size:1.05rem; line-height:1.55; max-width:760px; margin:.4rem 0 1.3rem;}
.stats {color:#52687a;font-size:.89rem; padding:.35rem 0 1.05rem;}
/* Pestañas de navegación horizontales, con estado activo inequívoco. */
[data-baseweb="tab-list"] { background:#e7eef4; padding:7px; border:1px solid #cbd9e3; border-radius:11px; gap:6px; }
button[role="tab"] { background:transparent; color:#234257 !important; border-radius:8px !important; padding:.7rem 1.18rem !important; font-weight:650 !important; transition: background-color .15s ease; }
button[role="tab"]:hover {background:#d2e1ea !important; color:#123e5c !important;}
button[role="tab"][aria-selected="true"] {background:#165679 !important; color:#fff !important; box-shadow:0 1px 3px #0000001b;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display:none !important;}
/* Botón primario, contraste y foco visible. */
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] { background:#155579; color:white; border:1px solid #155579; font-weight:700; border-radius:8px; min-height:2.8rem; }
.stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {background:#103e5b; color:white; border-color:#103e5b;}
.stButton button[kind="secondary"] {border:1px solid #bbcedd; background:white; color:#24465e; border-radius:8px;}
.stButton button[kind="secondary"]:hover {border-color:#165679; color:#165679; background:#edf4f8;}
.stButton button:focus-visible, button[role="tab"]:focus-visible {outline:3px solid #e3a842 !important; outline-offset:2px;}
[data-testid="stForm"] {background:white; border:1px solid #d1dee7; border-radius:12px; padding:1.1rem 1.15rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#d1dee7 !important;}
.small-note {font-size:.89rem;color:var(--sub);line-height:1.5;}
[data-testid="stCaptionContainer"] {color:#52687a !important;}
/* Evitar excesos de anchura en el texto. */
p, li {line-height:1.57;}
@media (max-width:650px) {
 section.main > div.block-container {padding-top:1.25rem;}
 button[role="tab"] {padding:.6rem .67rem !important; font-size:.87rem !important;}
}
</style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def catalogo():
    return CatalogSearch()

@st.cache_resource(show_spinner="Preparando búsqueda semántica…")
def mini_lm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, device="cpu")


def groq_key():
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip() or os.getenv("GROQ_API_KEY", "").strip()
    except Exception:
        return os.getenv("GROQ_API_KEY", "").strip()


@st.cache_data(show_spinner=False, ttl=3600)
def buscar(query: str, semantica: bool, rdf: bool, cantidad: int):
    modelo = None
    if semantica:
        try:
            modelo = mini_lm()
        except Exception:
            modelo = None
    encontrados = catalogo().search(query, encoder=modelo, use_graph=rdf, top_k=cantidad)
    return encontrados, bool(modelo is not None)


def lista_breve(hits):
    if not hits:
        return "No se encontraron fichas para esta consulta en la muestra disponible."
    lineas = ["Estas fichas pueden servir como punto de partida:"]
    for h in hits:
        d = h.doc
        aux = [x for x in (d.get("autor", ""), label_fecha(d.get("fecha", ""))) if x]
        detalle = f" — {' · '.join(aux)}" if aux else ""
        lineas.append(f"- **[{d['titulo']}]({d['ficha']})**{detalle}")
    return "\n".join(lineas)


def mostrar_fuentes(hits):
    st.markdown("### Fichas encontradas")
    if not hits:
        st.info("Prueba otro tema, título o persona.")
        return
    for i, h in enumerate(hits, start=1):
        d = h.doc
        with st.container(border=True):
            st.markdown(f"**{i}. [{d['titulo']}]({d['ficha']})**")
            meta = [x for x in (d.get("mcid", ""), d.get("autor", ""), label_fecha(d.get("fecha", ""))) if x]
            if meta:
                st.caption(" · ".join(meta))
            if d.get("colecciones_legibles"):
                st.write("**Colección:**", d["colecciones_legibles"])
            with st.expander("Ver detalles del catálogo"):
                if d.get("materias"):
                    st.write("**Materias:**", d["materias"])
                if d.get("notas"):
                    st.write("**Notas:**", d["notas"])
                if d.get("derechos"):
                    st.caption("Derechos declarados: " + d["derechos"])
                if h.graph_related:
                    rel = "; ".join(r["titulo"] for r in h.graph_related[:3])
                    st.caption("Relaciones curatoriales del catálogo: " + rel + ". No indican necesariamente vínculos históricos.")


def mostrar_resultado(resultado: dict):
    st.markdown("### Resultado")
    if resultado["claims"]:
        for c in resultado["claims"]:
            st.markdown("- " + c["texto"])
        st.caption("Las citas se contrastaron con campos catalográficos; no verifican el contenido de los libros.")
    else:
        st.markdown(lista_breve(resultado["hits"]))
        if resultado["modo"] == "Con Qwen":
            if resultado["estado"] == "cuota":
                st.info("Groq alcanzó temporalmente su cuota. Se muestra una selección documental rápida.")
            elif resultado["estado"] != "catalogo":
                st.caption("La respuesta de Qwen no estuvo disponible o no pasó la revisión formal de citas; se muestran las fuentes sin generar afirmaciones.")
        else:
            st.caption("Selección automática por metadatos; no se ha examinado el texto completo de las obras.")
    mostrar_fuentes(resultado["hits"])


st.markdown("<div class='hero-label'>Exploración de fuentes históricas</div>", unsafe_allow_html=True)
st.title("Explorador de Memoria Chilena")
st.markdown("<p class='hero-sub'>Busca documentos históricos por tema, título o autor y consulta sus fichas originales.</p>", unsafe_allow_html=True)
st.markdown(f"<div class='stats'>{catalogo().N} fichas en esta muestra · 4 colecciones temáticas · enlaces al catálogo de origen</div>", unsafe_allow_html=True)

buscar_tab, coleccion_tab, acerca_tab = st.tabs(["Consultar", "Colección", "Acerca de"])

with buscar_tab:
    with st.form("consulta", clear_on_submit=False):
        pregunta = st.text_input("¿Qué quieres investigar?", placeholder="Por ejemplo: ¿Qué obras tratan el sufragio femenino en Chile?")
        modo = st.radio("Tipo de respuesta", ["Rápida", "Con Qwen"], horizontal=True, index=0, help="Rápida muestra fichas sin esperar al modelo. Con Qwen redacta con los metadatos de esas fichas.")
        enviar = st.form_submit_button("Buscar fuentes", type="primary", use_container_width=True)

    st.caption("Prueba una consulta")
    ex1, ex2 = st.columns(2)
    with ex1:
        btn1 = st.button("Sufragio femenino", use_container_width=True)
        btn3 = st.button("Lenguas indígenas andinas", use_container_width=True)
    with ex2:
        btn2 = st.button("Instituto Nacional", use_container_width=True)
        btn4 = st.button("Centenario de 1910", use_container_width=True)

    with st.expander("Opciones de búsqueda"):
        semantica = st.checkbox("Búsqueda semántica (MiniLM)", value=False, help="Puede tardar más la primera vez, porque descarga el modelo.")
        rdf = st.checkbox("Aprovechar relaciones RDF", value=True)
        cantidad = st.slider("Número de fichas", min_value=3, max_value=6, value=4)

    nuevos = [
        (btn1, "¿Qué obras tratan el sufragio femenino en Chile?"),
        (btn2, "¿Qué fuentes documentan los inicios del Instituto Nacional?"),
        (btn3, "¿Qué textos coloniales están asociados con las lenguas indígenas andinas?"),
        (btn4, "¿Qué documentos permiten estudiar las celebraciones del centenario de 1910?"),
    ]
    query = (pregunta or "").strip() if enviar else ""
    for activo, ejemplo in nuevos:
        if activo:
            query = ejemplo
            break
    if query:
        if len(query) < 3:
            st.warning("Escribe una consulta un poco más específica.")
        else:
            with st.spinner("Buscando en las fichas…"):
                hits, semantic_ok = buscar(query, semantica, rdf, cantidad)
            if semantica and not semantic_ok:
                st.info("La búsqueda semántica no pudo cargarse; se utilizó búsqueda léxica.")
            claims = []
            estado = "catalogo"
            if modo == "Con Qwen":
                key = groq_key()
                if not key:
                    st.info("Qwen requiere GROQ_API_KEY en Streamlit Secrets. Por ahora se muestran las fuentes directamente.")
                    estado = "sin_api"
                else:
                    with st.spinner("Redactando a partir de los metadatos…"):
                        respuesta = answer_with_groq(query, hits, key)
                    claims, estado = respuesta.claims, respuesta.status
            st.session_state["ultimo_resultado"] = {"pregunta":query,"modo":modo,"hits":hits,"claims":claims,"estado":estado}

    if "ultimo_resultado" in st.session_state:
        r = st.session_state["ultimo_resultado"]
        st.divider()
        st.caption("Consulta: " + r["pregunta"])
        mostrar_resultado(r)

with coleccion_tab:
    st.markdown("### Colección consultable")
    corpus = catalogo().docs.copy()
    st.caption("La muestra incluye 34 fichas; no es la totalidad del portal.")
    cols = sorted(x for x in corpus.colecciones_legibles.unique().tolist() if x)
    l, r = st.columns([1, 1.5])
    with l:
        tema = st.selectbox("Colección", ["Todas"] + cols)
    with r:
        term = st.text_input("Filtrar por título o autor", key="filtro_catalogo")
    if tema != "Todas":
        corpus = corpus[corpus.colecciones_legibles == tema]
    if term:
        corpus = corpus[(corpus.titulo + " " + corpus.autor).str.contains(term, case=False, regex=False)]
    st.caption(f"{len(corpus)} fichas")
    for row in corpus.itertuples(index=False):
        with st.container(border=True):
            st.markdown(f"**[{row.titulo}]({row.ficha})**")
            details = [x for x in (row.mcid, row.autor, row.colecciones_legibles) if x]
            st.caption(" · ".join(details))

with acerca_tab:
    st.markdown("### Sobre el explorador")
    st.write("Esta aplicación nació del interés por explorar fuentes históricas con herramientas de recuperación de información. Trabaja con una muestra de 34 fichas de Memoria Chilena.")
    st.write("La búsqueda combina coincidencias de texto, relaciones RDF del catálogo y, cuando se activa, similitud semántica MiniLM. Qwen puede redactar una síntesis a partir de los campos recuperados.")
    st.markdown("**Alcances y límites**")
    st.write("La aplicación no analiza íntegramente las obras ni demuestra relaciones históricas solo porque dos fichas compartan una categoría. Las citas a metadatos no sustituyen la verificación documental.")
    st.markdown("**Fuente de los registros**")
    st.markdown("Las fichas provienen de [Memoria Chilena](https://www.memoriachilena.gob.cl/602/w3-channel.html), un portal de la Biblioteca Nacional de Chile. **Este explorador no está afiliado al portal ni a la institución.**")
    st.write("Si activas Qwen, tu consulta se procesa mediante Groq. Evita incluir información personal sensible.")

st.divider()
st.caption("Explorador documental · 34 fichas catalográficas · Abrir siempre la fuente original para verificar información histórica.")
