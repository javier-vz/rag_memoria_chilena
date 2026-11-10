"""Buscador de fuentes históricas: interfaz Streamlit para fichas documentales."""
from __future__ import annotations

import os
import re

import streamlit as st
from core.generation import answer_with_groq, catalog_fallback
from core.retrieval import CatalogSearch, MODEL_ID, label_fecha


st.set_page_config(
    page_title="Buscador de fuentes históricas",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Contrastes altos (azul tinta, cobalto, naranja), superficies blancas: no pasteles.
st.markdown("""<style>
:root { --ink:#111C32; --navy:#0B1E3B; --blue:#1749B5; --orange:#FF8500; --line:#8FA0B6; --muted:#35465B; }
[data-testid="stAppViewContainer"] {background:#FFFFFF; color:var(--ink);}
[data-testid="stHeader"] {background:#FFFFFF;}
section.main > div.block-container {max-width:1020px; padding-top:2.0rem; padding-bottom:2rem;}
h1, h2, h3 {color:var(--navy);letter-spacing:-.025em;}
h1 {font-size:clamp(1.8rem,3.4vw,2.55rem)!important;line-height:1.19!important;font-weight:760!important;}
h2 {font-size:1.3rem!important;}
h3 {font-size:1.16rem!important;}
p, li {line-height:1.6;}
a {color:#1646AD!important; text-decoration:underline; text-underline-offset:3px;}
a:hover {color:#AD3B00!important;}
.mc-label {font-size:.78rem;letter-spacing:.11em;font-weight:800;color:#18335A;text-transform:uppercase;margin-bottom:8px;}
.mc-intro {font-size:1.04rem;line-height:1.6;color:#263B57;max-width:780px;margin:.45rem 0 1rem;}
.mc-meta {border-left:4px solid var(--orange);padding:.5rem .8rem;background:#FFFFFF;color:#34455A;font-size:.91rem;font-weight:500;margin:1rem 0 1.4rem;}
/* Navegación inequívoca, con relleno sólido. */
[data-baseweb="tab-list"] {display:flex; gap:5px; flex-wrap:nowrap; overflow-x:auto; scrollbar-width:thin; background:var(--navy)!important; padding:7px!important; border:1px solid var(--navy); border-radius:11px; margin-bottom:1rem;}
button[role="tab"] {background:transparent!important; color:#FFFFFF!important; font-weight:700!important; border-radius:7px!important; padding:.64rem 1.25rem!important; min-height:44px!important; flex-shrink:0;}
button[role="tab"]:hover {background:#1E477C!important; color:white!important;}
button[role="tab"][aria-selected="true"] {background:var(--orange)!important; color:#111C32!important; box-shadow:0 2px 0 #00000030!important;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display:none!important;}
/* Controles legibles de alto contraste. */
.stFormSubmitButton button[kind="primary"], .stButton button[kind="primary"] {background:var(--blue)!important;border:2px solid var(--blue)!important;color:white!important;font-weight:750!important;border-radius:8px!important;min-height:47px!important;}
.stFormSubmitButton button[kind="primary"]:hover, .stButton button[kind="primary"]:hover {background:#0C327F!important;border-color:#0C327F!important;}
.stButton button[kind="secondary"] {background:white!important; color:var(--navy)!important; border:1.7px solid #3E567A!important; border-radius:8px!important; min-height:46px!important; font-weight:600!important;}
.stButton button[kind="secondary"]:hover {background:var(--navy)!important; color:white!important;border-color:var(--navy)!important;}
.stButton button:focus-visible,button[role="tab"]:focus-visible,input:focus-visible,textarea:focus-visible {outline:3px solid var(--orange)!important;outline-offset:2px!important;}
[data-testid="stForm"] {background:#FFFFFF!important; border:1.6px solid #6F859D!important; border-radius:11px!important; padding:1.05rem 1.1rem!important;}
[data-testid="stTextInput"] input {font-size:16px!important;min-height:44px!important; border-color:#677D95!important;}
[data-testid="stExpander"] {border:1.5px solid #607894!important;border-radius:9px!important;background:#FFFFFF!important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#71859A!important;}
[data-testid="stCaptionContainer"] {color:#344963!important;}
.mc-answer-note {font-size:.84rem;color:#35465B;margin-top:.6rem;}
/* Bloques compactos; el espaciado en móvil tiene prioridad. */
@media (max-width: 760px) {
 section.main > div.block-container {padding-top:1rem!important;padding-left:.85rem!important;padding-right:.85rem!important;}
 [data-baseweb="tab-list"] {gap:3px;padding:4px!important;border-radius:8px;}
 button[role="tab"] {padding:.65rem .85rem!important;font-size:.91rem!important;}
 [data-testid="stForm"] {padding:.72rem!important;}
 [data-testid="stHorizontalBlock"] {flex-wrap:wrap!important;gap:.5rem!important;}
 [data-testid="column"] {min-width:100%!important; width:100%!important;flex:1 1 100%!important;}
 h1 {font-size:1.85rem!important;}
 h2 {font-size:1.22rem!important;}
 .mc-intro {font-size:.98rem;}
 .mc-meta {font-size:.86rem;}
}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def catalogo():
    return CatalogSearch()


@st.cache_resource(show_spinner="Cargando modelo semántico…")
def mini_lm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID, device="cpu")


def groq_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip() or os.getenv("GROQ_API_KEY", "").strip()
    except Exception:
        return os.getenv("GROQ_API_KEY", "").strip()


# NO decorar con st.cache_data: los objetos Hit no siempre son serializables.
def buscar(query: str, semantica: bool, rdf: bool, cantidad: int):
    modelo = None
    if semantica:
        try:
            modelo = mini_lm()
        except Exception:
            modelo = None
    hits = catalogo().search(query, encoder=modelo, use_graph=rdf, top_k=cantidad)
    return hits, modelo is not None


def enlace_citas(texto: str, hits: list) -> str:
    links = {h.doc["mcid"]: h.doc.get("ficha", "") for h in hits}

    def sustituir(match):
        mid = match.group(1)
        url = links.get(mid, "")
        return f"[{mid}]({url})" if url.startswith("https://") else match.group(0)

    return re.sub(r"\[(MC\d+)\]", sustituir, texto)


def mostrar_fuentes(hits: list) -> None:
    st.markdown("### Fuentes consultadas")
    if not hits:
        st.info("No hay fichas en esta selección. Intenta otra búsqueda.")
        return
    for num, hit in enumerate(hits, 1):
        d = hit.doc
        with st.container(border=True):
            st.markdown(f"**{num}. [{d['titulo']}]({d['ficha']})**")
            meta = [x for x in (d.get("mcid", ""), d.get("autor", "")) if x]
            if meta:
                st.caption(" · ".join(meta))
            if d.get("colecciones_legibles"):
                st.write("**Colección:**", d["colecciones_legibles"])
            with st.expander("Datos de la ficha"):
                if d.get("materias"):
                    st.write("**Materias catalogadas:**", d["materias"])
                if d.get("notas"):
                    st.write("**Notas:**", d["notas"])
                fecha = label_fecha(d.get("fecha", ""))
                if fecha:
                    st.write("**Fecha consignada en el registro:**", fecha)
                if d.get("derechos"):
                    st.caption("Derechos declarados: " + d["derechos"])
                if hit.graph_related:
                    rels = "; ".join(r["titulo"] for r in hit.graph_related[:3])
                    st.caption("Otros registros relacionados por etiquetas curatoriales: " + rels)


def mostrar_resultado(result: dict) -> None:
    st.markdown("### Respuesta")
    st.markdown(enlace_citas(result["texto"], result["hits"]))
    if result["modo"] == "Con Qwen":
        if result["estado"] == "control_formal":
            st.caption("Síntesis de Qwen basada en metadatos con citas contrastadas; no equivale a una verificación de los libros.")
        elif result["estado"] == "cuota":
            st.info("Groq alcanzó temporalmente su límite gratuito. Se muestra la selección de fuentes.")
        elif result["estado"] == "sin_api":
            st.info("Qwen requiere GROQ_API_KEY en Streamlit Secrets. Se muestra la búsqueda documental.")
        else:
            st.caption("La respuesta generada no pasó los controles de citas o el servicio no estuvo disponible. Se muestra una orientación basada en el catálogo.")
    else:
        st.caption("Resultado rápido basado en metadatos; los libros no fueron analizados íntegramente.")
    mostrar_fuentes(result["hits"])


st.markdown("<div class='mc-label'>Historia de Chile · Fuentes documentales</div>", unsafe_allow_html=True)
st.title("Buscador de fuentes históricas")
st.markdown(
    "<p class='mc-intro'>Busca obras, temas y autores en una selección de fichas documentales. "
    "Puedes leer una orientación breve y abrir cada fuente original.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='mc-meta'>{catalogo().N} fichas seleccionadas · "
    "Procedencia: portal Memoria Chilena · Esta herramienta no es el portal original</div>",
    unsafe_allow_html=True,
)

buscar_tab, coleccion_tab, acerca_tab = st.tabs(["Consultar", "Colecciones", "Acerca de"])

with buscar_tab:
    modo = st.radio(
        "Tipo de respuesta",
        ["Rápida", "Con Qwen"],
        horizontal=True,
        help="Rápida muestra una orientación inmediata. Qwen redacta una síntesis a partir de los metadatos.",
    )
    with st.form("consulta", clear_on_submit=False):
        pregunta = st.text_input(
            "¿Qué tema quieres investigar?",
            placeholder="Ej.: ¿Qué documentos tratan sobre el sufragio femenino en Chile?",
        )
        enviar = st.form_submit_button("Buscar fuentes", type="primary", use_container_width=True)

    st.markdown("**Consultas sugeridas**")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        b1 = st.button("Sufragio femenino", key="ej_suf", use_container_width=True)
        b3 = st.button("Lenguas indígenas andinas", key="ej_and", use_container_width=True)
    with col2:
        b2 = st.button("Instituto Nacional", key="ej_ins", use_container_width=True)
        b4 = st.button("Centenario de 1910", key="ej_cen", use_container_width=True)

    with st.expander("Opciones de búsqueda"):
        semantica = st.checkbox("Activar similitud semántica (MiniLM)", value=False,
                                help="La primera carga del modelo puede tardar. Para empezar, la búsqueda léxica es inmediata.")
        rdf = st.checkbox("Incluir relaciones RDF", value=True)
        cantidad = st.slider("Fichas por consulta", min_value=3, max_value=6, value=4)

    examples = [
        (b1, "¿Qué obras tratan el sufragio femenino en Chile?"),
        (b2, "¿Qué fuentes documentan los inicios del Instituto Nacional?"),
        (b3, "¿Qué textos coloniales están asociados con las lenguas indígenas andinas?"),
        (b4, "¿Qué documentos permiten estudiar las celebraciones del centenario de 1910?"),
    ]
    query = pregunta.strip() if enviar else ""
    for clicked, example in examples:
        if clicked:
            query = example
            break

    if query:
        if len(query) < 3:
            st.warning("Escribe un tema o una pregunta más específica.")
        else:
            with st.spinner("Buscando fuentes…"):
                hits, semantic_ok = buscar(query, semantica, rdf, cantidad)
            if semantica and not semantic_ok:
                st.info("MiniLM no estuvo disponible; se utilizó recuperación léxica.")
            estado = "catalogo"
            if modo == "Con Qwen":
                api = groq_key()
                if not api:
                    estado = "sin_api"
                    texto = catalog_fallback(hits)
                else:
                    # Caché por sesión para no repetir llamadas gratuitas de Groq.
                    cache = st.session_state.setdefault("respuestas_groq", {})
                    key = (query, tuple(h.doc["mcid"] for h in hits))
                    if key not in cache:
                        with st.spinner("Redactando una síntesis con Qwen…"):
                            cache[key] = answer_with_groq(query, hits, api)
                    answer = cache[key]
                    texto, estado = answer.text, answer.status
            else:
                texto = catalog_fallback(hits)
            st.session_state["ultimo_resultado"] = {
                "pregunta": query, "modo": modo, "hits": hits, "texto": texto, "estado": estado,
            }

    if "ultimo_resultado" in st.session_state:
        r = st.session_state["ultimo_resultado"]
        st.divider()
        st.caption("Consulta: " + r["pregunta"])
        mostrar_resultado(r)

with coleccion_tab:
    st.markdown("### Fichas de la muestra")
    corpus = catalogo().docs.copy()
    st.caption(f"{len(corpus)} registros disponibles. No representan la totalidad de Memoria Chilena.")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        tema = st.selectbox("Colección", ["Todas"] + sorted(x for x in corpus.colecciones_legibles.unique() if x))
    with c2:
        term = st.text_input("Título o autor", key="filtro_coleccion")
    if tema != "Todas":
        corpus = corpus[corpus.colecciones_legibles == tema]
    if term:
        corpus = corpus[(corpus.titulo + " " + corpus.autor).str.contains(term, case=False, regex=False)]
    st.caption(f"{len(corpus)} fichas")
    for row in corpus.itertuples(index=False):
        with st.container(border=True):
            st.markdown(f"**[{row.titulo}]({row.ficha})**")
            st.caption(" · ".join(x for x in (row.mcid, row.autor, row.colecciones_legibles) if x))

with acerca_tab:
    st.markdown("### Sobre esta herramienta")
    st.write("Este buscador reúne una muestra de fichas históricas para experimentar con recuperación de información. No es una interfaz del portal Memoria Chilena ni está afiliado a la Biblioteca Nacional de Chile.")
    st.write("La recuperación combina coincidencias de palabras y títulos, relaciones catalográficas RDF y, cuando se activa, embeddings MiniLM. La respuesta con Qwen se genera mediante la API de Groq y utiliza únicamente los metadatos recuperados.")
    st.markdown("**Alcance de los resultados**")
    st.write("Se indexan fichas, no las obras digitalizadas completas. Una relación temática no implica acuerdo entre documentos, ni autoriza afirmaciones sobre sus contenidos. Las fechas de catálogo no son necesariamente fechas de publicación.")
    st.markdown("**Fuente de los datos**")
    st.markdown("Los registros enlazan a [Memoria Chilena](https://www.memoriachilena.gob.cl/602/w3-channel.html), portal de la Biblioteca Nacional de Chile. Las fuentes originales deben consultarse antes de interpretar los documentos.")
    st.write("Las consultas realizadas con Qwen se envían a Groq. Evita ingresar datos personales sensibles.")

st.divider()
st.caption("Buscador de fuentes históricas · Selección documental con enlaces a los registros originales")
