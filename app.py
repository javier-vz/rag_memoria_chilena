"""Memoria Chilena · Explorador documental con recuperación híbrida y Groq."""
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from core.retrieval import CatalogSearch, MODEL_ID, label_fecha
from core.generation import answer_with_groq, catalog_fallback

st.set_page_config(page_title='Memoria Chilena · Explorador documental',page_icon='📚',layout='wide',initial_sidebar_state='expanded')
st.markdown("""<style>
/* Presentación editorial discreta, sin gráficos ornamentales */
:root { --ink:#213446; --rule:#d7dfe4; }
[data-testid="stAppViewContainer"] {background:#fbfaf7;}
[data-testid="stSidebar"] {background:#f1f3f3;}
h1,h2,h3 {letter-spacing:-.025em;}
h1 {font-size:2rem!important;font-weight:620!important;color:#243746!important;}
h2 {font-size:1.23rem!important;font-weight:620!important;}
[data-testid="stChatMessage"] {border:1px solid #e0e6e9;border-radius:.5rem;background:#fff;}
a {color:#285c78!important;}
.mc-kicker {letter-spacing:.1em;text-transform:uppercase;font-size:.72rem;color:#687c89;margin-bottom:.6rem;}
.mc-muted {color:#60717c;font-size:.92rem;}
.mc-rule {border-top:1px solid #dce2e6;margin:1.5rem 0;}
</style>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def engine():
    return CatalogSearch()

@st.cache_resource(show_spinner='Cargando modelo de búsqueda semántica…')
def encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID,device='cpu')

def get_key():
    try:
        k=st.secrets.get('GROQ_API_KEY','')
        if k:return str(k).strip()
    except (FileNotFoundError, KeyError, AttributeError):pass
    return os.environ.get('GROQ_API_KEY','').strip()

@st.cache_data(show_spinner=False,ttl=3600)
def cached_search(question: str, use_graph: bool, semantic: bool):
    m=None
    if semantic:
        try:m=encoder()
        except Exception: m=None
    hits=engine().search(question,encoder=m,use_graph=use_graph,top_k=4)
    return hits, m is not None


def source_list(hits):
    st.markdown('### Fuentes recuperadas')
    for pos,h in enumerate(hits,1):
        d=h.doc
        with st.expander(f"{pos}. {d['titulo']} · {d['mcid']}",expanded=(pos==1)):
            st.markdown(f"**Ficha original:** [{d['mcid']}]({d['ficha']})")
            if d.get('autor'):st.write('**Autoría catalogada:**',d['autor'])
            if label_fecha(d.get('fecha','')):st.write('**Fecha registrada en el catálogo:**',label_fecha(d['fecha']))
            st.write('**Colección:**',d.get('colecciones_legibles','No indicada'))
            if d.get('materias'):st.write('**Materias:**',d['materias'])
            if d.get('derechos'):st.caption('Derechos declarados en la ficha: '+d['derechos'])
            if h.graph_related:
                st.caption('Vínculos curatoriales específicos: '+ '; '.join(r['titulo'] for r in h.graph_related[:3])+'. No son hechos históricos comprobados.')


def show_answer(question, use_groq, use_graph, use_semantic):
    if len(question.strip())<3:
        st.warning('Escribe una consulta más específica.');return
    with st.spinner('Consultando el catálogo…'):
        hits,semantic_ok=cached_search(question,use_graph,use_semantic)
    if use_semantic and not semantic_ok:
        st.info('La búsqueda semántica no estuvo disponible. Se usó recuperación léxica con anclaje por títulos.')
    api=get_key()
    if use_groq and api:
        with st.spinner('Redactando con Qwen a partir de las fichas…'):
            ans=answer_with_groq(question,hits,api)
        answer=ans.text
        formal=(ans.status=='control_formal')
        note=('Las citas se comprobaron contra campos del catálogo. No constituyen una validación histórica del contenido de los libros.'
              if formal else 'Se muestra una selección basada en metadatos; la generación no estuvo disponible o no pasó los controles formales.')
    else:
        answer=catalog_fallback(hits)
        note='Selección documental automática a partir de metadatos. No se analizaron los textos completos.'
    with st.chat_message('assistant'):
        st.markdown(answer)
        st.caption(note)
        source_list(hits)
    return {'answer': answer, 'hits': hits, 'note': note}

with st.sidebar:
    st.markdown('**Memoria Chilena**')
    section=st.radio('Navegación',['Consultar','Colección','Acerca de'],label_visibility='collapsed')
    st.divider()
    if section=='Consultar':
        st.markdown('**Opciones de consulta**')
        use_semantic=st.toggle('Búsqueda semántica',value=True)
        use_graph=st.toggle('Relaciones RDF',value=True)
        use_groq=st.toggle('Redacción con Qwen',value=True,disabled=not bool(get_key()))
        if not get_key():st.caption('Qwen estará disponible cuando se configure GROQ_API_KEY en Streamlit Secrets.')
        st.caption('El modelo trabaja con fichas catalográficas. No verifica afirmaciones sobre el contenido íntegro de las obras.')

st.markdown('<div class="mc-kicker">Biblioteca Nacional de Chile · Proyecto independiente de investigación</div>',unsafe_allow_html=True)
st.title('Explorador documental de Memoria Chilena')
st.markdown('<p class="mc-muted">Búsqueda de fuentes históricas mediante metadatos, similitud semántica y relaciones curatoriales.</p>',unsafe_allow_html=True)

if section=='Consultar':
    st.markdown('<div class="mc-rule"></div>',unsafe_allow_html=True)
    if 'history' not in st.session_state:st.session_state.history=[]
    if not st.session_state.history:
        st.markdown('**Prueba una pregunta**')
        examples=['¿Qué obras tratan el sufragio femenino en Chile?',
                  '¿Qué textos coloniales están asociados con las lenguas indígenas andinas?',
                  '¿Qué documentos permiten estudiar las celebraciones del centenario de 1910?',
                  '¿Qué fuentes documentan los inicios del Instituto Nacional?']
        for i,q in enumerate(examples):
            if st.button(q,key=f'ex{i}',use_container_width=True):st.session_state.pending=q;st.rerun()
    for turn in st.session_state.history[-8:]:
        with st.chat_message('user'):st.write(turn['question'])
        with st.chat_message('assistant'):
            st.markdown(turn['answer'])
            st.caption(turn['note'])
            source_list(turn['hits'])
    q=st.chat_input('Pregunta por una obra, tema, autor o colección…')
    if not q and 'pending' in st.session_state:q=st.session_state.pop('pending')
    if q:
        with st.chat_message('user'):st.write(q)
        result=show_answer(q,use_groq,use_graph,use_semantic)
        if result:st.session_state.history.append({'question':q,**result})

elif section=='Colección':
    corpus=engine().docs
    st.markdown('### Catálogo consultable')
    st.caption(f'{len(corpus)} fichas documentales de cuatro ámbitos temáticos. Este corpus es una muestra, no la totalidad de Memoria Chilena.')
    all_cols=sorted(corpus.colecciones_legibles.dropna().unique())
    choice=st.selectbox('Colección', ['Todas']+all_cols)
    term=st.text_input('Filtrar por título o autor')
    d=corpus.copy()
    if choice!='Todas':d=d[d.colecciones_legibles==choice]
    if term:d=d[(d.titulo+' '+d.autor).str.contains(term,case=False,regex=False)]
    st.caption(f'{len(d)} resultados')
    for r in d.itertuples(index=False):
        st.markdown(f"**[{r.titulo}]({r.ficha})**")
        subtitle=' · '.join(x for x in [r.mcid,r.autor,r.colecciones_legibles] if x)
        st.caption(subtitle)
        st.divider()

else:
    st.markdown('### Acerca del proyecto')
    st.write('Esta aplicación explora una muestra de fichas de Memoria Chilena mediante recuperación léxica TF-IDF, embeddings MiniLM multilingües y relaciones RDF catalográficas filtradas.')
    st.write('Cuando Groq está configurado, Qwen redacta breves orientaciones a partir de los campos de las fichas. Un control verifica que las citas se refieran a registros recuperados y que sus fragmentos existan en los metadatos. No sustituye la verificación historiográfica.')
    st.markdown('**Límites**')
    st.write('No se han indexado íntegramente los libros. Las fechas son campos catalográficos y no equivalen necesariamente a fechas de publicación. Compartir una categoría o minisitio no demuestra relación causal ni acuerdo entre documentos.')
    st.markdown('**Fuentes y atribución**')
    st.markdown('Las fichas enlazan al sitio original [Memoria Chilena](https://www.memoriachilena.gob.cl/602/w3-channel.html), de la Biblioteca Nacional de Chile. Esta interfaz es un proyecto de investigación independiente, no un servicio oficial de la institución.')
    st.write('Consultas enviadas a Qwen se procesan a través de Groq. No introduzcas datos personales sensibles.')

st.divider()
st.caption('Memoria Chilena · Prototipo de investigación documental · Metadatos de fuente original, inferencias de recuperación no equivalen a hechos históricos.')
