"""GraphRAG de Memoria Chilena: recuperación híbrida y respuestas con fuentes."""
from __future__ import annotations

import os
import re

import streamlit as st
from core.generation import answer_with_groq, catalog_fallback
from core.retrieval import CatalogSearch, MODEL_ID, label_fecha


st.set_page_config(
    page_title="GraphRAG de Memoria Chilena",
    page_icon="📚",
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
/* Franja azul marino y naranja real: sin fondos pastel */
.mc-hero { background:#071A33; color:#FFFFFF; border-top:7px solid #FF8200; padding:1.35rem 1.6rem 1.4rem; border-radius:0 0 12px 12px; margin:1rem 0 .85rem; }
.mc-hero strong { display:block; color:#FFFFFF; font-size:1.22rem; font-weight:800; margin-bottom:.4rem; }
.mc-hero p { color:#FFFFFF; line-height:1.55; font-size:1rem; margin:0 0 .65rem; }
.mc-hero a { color:#FFB000!important; font-weight:800; text-decoration:underline; }
.mc-meta { border-left:6px solid #FF8200; background:#071A33!important; color:#FFFFFF!important; border-radius:5px; padding:.8rem 1rem; font-weight:680; }
.mc-meta strong { color:#FFFFFF; }
.mc-label { color:#0B1E3B!important; font-size:.82rem; }
/* Navegación inequívoca, con relleno sólido. */
[data-baseweb="tab-list"] {display:flex; gap:5px; flex-wrap:nowrap; overflow-x:auto; scrollbar-width:thin; background:var(--navy)!important; padding:7px!important; border:1px solid var(--navy); border-radius:11px; margin-bottom:1rem;}
button[role="tab"] {background:transparent!important; color:#FFFFFF!important; font-weight:700!important; border-radius:7px!important; padding:.64rem 1.25rem!important; min-height:44px!important; flex-shrink:0;}
button[role="tab"]:hover {background:#0057DA!important; color:white!important;}
button[role="tab"][aria-selected="true"], [data-testid="stTabs"] button[aria-selected="true"] {background:#FF8200!important; color:#071A33!important; box-shadow:none!important; font-weight:850!important;}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display:none!important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {background:#071A33!important; border-radius:9px!important; padding:6px!important;}
[data-testid="stTabs"] button[role="tab"] { color:#FFFFFF!important; background:#071A33!important; border-radius:6px!important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background:#FF8200!important; color:#071A33!important; border:0!important; }
/* Controles legibles de alto contraste. */
.stFormSubmitButton button[kind="primary"], .stButton button[kind="primary"] {background:#FF8200!important;border:2px solid #FF8200!important;color:#071A33!important;font-weight:750!important;border-radius:8px!important;min-height:47px!important;}
.stFormSubmitButton button[kind="primary"]:hover, .stButton button[kind="primary"]:hover {background:#FFB000!important;border-color:#FFB000!important;}
.stButton button[kind="secondary"] {background:white!important; color:var(--navy)!important; border:1.7px solid #3E567A!important; border-radius:8px!important; min-height:46px!important; font-weight:600!important;}
.stButton button[kind="secondary"]:hover {background:#0B1E3B!important; color:#FFFFFF!important;border-color:#0B1E3B!important;}
.stButton button:focus-visible,button[role="tab"]:focus-visible,input:focus-visible,textarea:focus-visible {outline:3px solid var(--orange)!important;outline-offset:2px!important;}
[data-testid="stForm"] {background:#FFFFFF!important; border:2px solid #0B1E3B!important; border-radius:11px!important; padding:1.05rem 1.1rem!important;}
[data-testid="stTextInput"] input {font-size:16px!important;min-height:44px!important; border-color:#0B1E3B!important;}
[data-testid="stExpander"] {border:2px solid #0B1E3B!important;border-radius:9px!important;background:#FFFFFF!important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#71859A!important;}
[data-testid="stCaptionContainer"] {color:#344963!important;}
.mc-answer-note {font-size:.84rem;color:#35465B;margin-top:.6rem;}
/* Solo móvil: controles grandes, contenido compacto y pestañas sin scroll lateral. */
@media (max-width: 760px) {
 section.main > div.block-container {
   padding: .8rem .85rem 1.8rem!important;
   max-width:100%!important;
 }
 h1 {font-size:1.75rem!important;line-height:1.18!important;margin-bottom:.45rem!important;}
 h2 {font-size:1.19rem!important;}
 h3 {font-size:1.1rem!important;}
 p, li {line-height:1.45!important;}
 .mc-label {font-size:.7rem!important;margin-bottom:.4rem!important;letter-spacing:.075em;}
 .mc-hero {margin:.65rem 0 .65rem!important;padding:.85rem .95rem!important;border-top-width:5px!important;border-radius:0 0 9px 9px!important;}
 .mc-hero strong {font-size:1.04rem!important;line-height:1.28!important;margin-bottom:.35rem!important;}
 .mc-hero p {font-size:.88rem!important;line-height:1.44!important;margin:0 0 .5rem!important;}
 .mc-hero a {font-size:.89rem!important;}
 .mc-meta {font-size:.81rem!important;line-height:1.38!important;margin:.6rem 0 .85rem!important;padding:.56rem .72rem!important;border-left-width:4px!important;}
 /* Las tres pestañas quedan siempre visibles y se pueden pulsar con un dedo. */
 [data-testid="stTabs"] [data-baseweb="tab-list"] {
   display:flex!important;justify-content:space-between!important;
   width:100%!important;max-width:100%!important;
   flex-wrap:nowrap!important;overflow-x:visible!important;
   gap:3px!important;padding:4px!important;margin-bottom:.7rem!important;
   border-radius:8px!important;box-sizing:border-box!important;
 }
 [data-testid="stTabs"] button[role="tab"] {
   min-width:0!important;flex:1 1 0!important;
   min-height:46px!important;max-width:none!important;
   padding:.55rem .15rem!important;
   font-size:.86rem!important;line-height:1.2!important;
   white-space:nowrap!important;text-align:center!important;
   justify-content:center!important;
 }
 [data-testid="stTabs"] [data-baseweb="tab-panel"] {min-width:0!important;}
 /* Formulario principal y modo de respuesta. */
 [data-testid="stForm"] {padding:.72rem!important;border-radius:9px!important;}
 [data-testid="stTextInput"] input {font-size:16px!important;min-height:48px!important;width:100%!important;}
 .stFormSubmitButton button[kind="primary"] {min-height:50px!important;width:100%!important;}
 [data-testid="stRadio"] [role="radiogroup"] {display:flex!important;gap:.8rem!important;flex-wrap:wrap!important;}
 [data-testid="stRadio"] label {min-height:40px!important;align-items:center!important;}
 /* Dos columnas en escritorio, una en móvil; textos largos no desbordan. */
 [data-testid="stHorizontalBlock"] {flex-wrap:wrap!important;gap:.4rem!important;}
 [data-testid="column"] {min-width:100%!important;width:100%!important;flex:1 1 100%!important;}
 [data-testid="stButton"] button,
 .stButton button[kind="secondary"] {min-height:48px!important;width:100%!important;white-space:normal!important;line-height:1.25!important;padding:.58rem .75rem!important;}
 [data-testid="stExpander"] {overflow:hidden!important;border-radius:8px!important;}
 [data-testid="stExpander"] summary {min-height:46px!important;}
 [data-testid="stVerticalBlockBorderWrapper"] {min-width:0!important;}
 [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
   overflow-wrap:anywhere!important;word-break:normal!important;
 }
 [data-testid="stMarkdownContainer"] a {overflow-wrap:anywhere!important;}
}
/* En teléfonos muy estrechos, mantener legible el nombre de cada pestaña. */
@media (max-width: 360px) {
 section.main > div.block-container {padding-left:.65rem!important;padding-right:.65rem!important;}
 [data-testid="stTabs"] button[role="tab"] {font-size:.8rem!important;padding:.52rem .08rem!important;}
 .mc-hero {padding:.75rem .8rem!important;}
 .mc-hero p {font-size:.85rem!important;}
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


st.markdown("<div class='mc-label'>GraphRAG · Patrimonio documental chileno</div>", unsafe_allow_html=True)
st.title("GraphRAG de Memoria Chilena")
st.markdown(
    "<div class='mc-hero'>"
    "<strong>Explora las fuentes del portal Memoria Chilena</strong>"
    "<p>Construimos un buscador sobre 34 fichas documentales del portal Memoria Chilena. "
    "Combina palabras clave (TF-IDF), embeddings MiniLM y relaciones RDF; "
    "Qwen, mediante Groq, puede redactar respuestas con citas a las fichas originales.</p>"
    "<a href='https://www.memoriachilena.gob.cl/602/w3-channel.html' target='_blank' rel='noopener noreferrer'>Ver portal Memoria Chilena ↗</a>"
    "</div>", unsafe_allow_html=True,
)
st.markdown(
    f"<div class='mc-meta'><strong>{catalogo().N} fichas documentales</strong> · "
    "4 colecciones temáticas · 23 conexiones RDF específicas</div>",
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
    # Mantener el orden de lectura al apilar las columnas en el celular.
    col1, col2 = st.columns(2, gap="small")
    with col1:
        b1 = st.button("Sufragio femenino", key="ej_suf", use_container_width=True)
    with col2:
        b2 = st.button("Instituto Nacional", key="ej_ins", use_container_width=True)
    col3, col4 = st.columns(2, gap="small")
    with col3:
        b3 = st.button("Lenguas indígenas andinas", key="ej_and", use_container_width=True)
    with col4:
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
    st.markdown("### Fichas del portal Memoria Chilena")
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
    st.write("Este GraphRAG trabaja con una selección de 34 fichas obtenidas del portal Memoria Chilena. Su objetivo es recuperar documentos por tema, título y autor, y facilitar el acceso a los registros originales.")
    st.write("La recuperación combina TF-IDF de palabras y caracteres, embeddings MiniLM multilingües y 23 relaciones RDF curatoriales específicas. Qwen (Groq) redacta respuestas a partir de los campos de las fichas recuperadas, con controles formales de citas.")
    st.markdown("**Alcance de los resultados**")
    st.write("Se indexan fichas, no las obras digitalizadas completas. Una relación temática no implica acuerdo entre documentos, ni autoriza afirmaciones sobre sus contenidos. Las fechas de catálogo no son necesariamente fechas de publicación.")
    st.markdown("**Fuente de los datos**")
    st.markdown("Los registros enlazan a [Memoria Chilena](https://www.memoriachilena.gob.cl/602/w3-channel.html), portal de la Biblioteca Nacional de Chile. Las fuentes originales deben consultarse antes de interpretar los documentos.")
    st.write("Las consultas realizadas con Qwen se envían a Groq. Evita ingresar datos personales sensibles.")

st.divider()
st.caption("GraphRAG de Memoria Chilena · 34 fichas del portal · TF-IDF + MiniLM + RDF + Qwen/Groq")
