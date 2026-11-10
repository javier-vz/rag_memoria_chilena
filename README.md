# Buscador de fuentes históricas

Aplicación en Streamlit para consultar **una selección de 34 fichas documentales** publicadas en el portal [Memoria Chilena](https://www.memoriachilena.gob.cl/602/w3-channel.html). Permite buscar por títulos, temas o autores y abrir los registros originales. **No es el portal Memoria Chilena ni una aplicación de la Biblioteca Nacional de Chile.**

## Qué hace

- Recuperación por palabras clave y títulos (TF-IDF) y relaciones RDF específicas del catálogo.
- Búsqueda semántica con MiniLM multilingüe, opcional; la primera carga puede demorar.
- Dos modos de respuesta: **Rápida**, sin llamadas a modelos de generación; y **Con Qwen**, que redacta una síntesis breve de metadatos utilizando Groq.
- Citas con identificadores y enlaces a las fichas originales. Validación formal de evidencias catalográficas; **no se afirma haber leído las obras completas**.
- Interfaz adaptada a escritorio y celular, con pestañas, botones de alto contraste y fichas desplegables.

Los metadatos históricos pueden contener errores o variantes, y compartir una categoría no demuestra que las obras tengan igual contenido, género o postura. La aplicación sirve para **descubrir fuentes**, no para validar interpretaciones históricas.

## Publicar en GitHub y Streamlit Community Cloud

1. Descomprime el proyecto y sube el **contenido** de su carpeta a GitHub, con `app.py` en la raíz.
2. En [Streamlit Community Cloud](https://share.streamlit.io/) selecciona el repositorio, la rama principal y el archivo `app.py`.
3. En **App settings → Secrets** guarda la clave de Groq:

   ```toml
   GROQ_API_KEY = "tu_clave"
   ```

4. Guarda los cambios y reinicia la app si la versión anterior continúa visible.

No subas la clave a GitHub. `.streamlit/secrets.toml` está excluido mediante `.gitignore`. El archivo `secrets.toml.example` es solo un ejemplo sin credenciales.

## Ejecución local

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Para usar Qwen en local, define la variable de entorno `GROQ_API_KEY` o crea `.streamlit/secrets.toml` en tu ordenador (sin versionarlo).

## Estructura

```text
app.py
core/
  retrieval.py
  generation.py
data/
  catalogo.csv
  relaciones.csv
  embeddings.npz
.streamlit/
  config.toml
  secrets.toml.example
requirements.txt
tests/
README.md
.gitignore
```

## Alcance y limitaciones

El índice contiene **34 fichas** y **23 conexiones RDF específicas**. Los embeddings de los documentos están precalculados. La app no redistribuye los documentos PDF y no necesita un servidor RDF.

Cuando se activa Qwen, se envían a Groq la consulta y los campos catalográficos recuperados. Las evidencias se verifican contra los campos de la ficha: el proceso **no valida por sí solo la veracidad de las interpretaciones de una obra**. Si falla el servicio o el control de citas, se presenta una orientación no generativa, basada en el catálogo.

## Pruebas

```bash
PYTHONPATH=. pytest -q tests
```

Si trabajas con Windows, ejecuta `python -m unittest discover -s tests` desde la raíz o configura `PYTHONPATH` para que Python encuentre el paquete `core`.
