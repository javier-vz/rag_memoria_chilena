# Memoria Chilena — explorador documental

Aplicación de investigación **independiente** construida con Streamlit. Consulta un corpus piloto de **34 fichas** de [Memoria Chilena](https://www.memoriachilena.gob.cl/602/w3-channel.html) (Biblioteca Nacional de Chile).

La interfaz es deliberadamente sobria: una pregunta, una respuesta orientativa y enlaces a las fichas de origen. El sistema **no transcribe ni verifica el contenido íntegro** de los libros.

## Características

- Recuperación por coincidencias de palabras y caracteres (TF-IDF).
- Embeddings precomputados de los 34 registros y codificación de la pregunta con `paraphrase-multilingual-MiniLM-L12-v2`.
- Reordenamiento por coincidencias de título y expansión muy moderada de **23 relaciones catalográficas específicas**; no se tratan como hechos históricos.
- Respuestas breves mediante **Qwen 3.8 27B en Groq**, con comprobación formal de IDs y fragmentos literales de metadatos.
- Modo **sin Groq**: muestra fuentes recuperadas y una descripción conservadora.
- La primera carga del modelo de embeddings descarga archivos desde Hugging Face. Si no está disponible, la búsqueda cae a TF-IDF.

## Publicar en GitHub

1. Crea un repositorio nuevo (p. ej. `memoria-chilena-rag`).
2. Sube **el contenido de esta carpeta**, dejando `app.py` en la raíz.
3. No subas `secrets.toml`, PDFs, archivos `etapa*.zip` ni carpetas de trabajo de Jupyter.
4. Publica el proyecto en [Streamlit Community Cloud](https://share.streamlit.io/) usando `app.py` como punto de entrada.
5. En **App settings → Secrets**, pega:

   ```toml
   GROQ_API_KEY = "tu_clave_real"
   ```

   Esta clave **NO** se agrega a GitHub. El archivo `.streamlit/secrets.toml.example` es solo un ejemplo sin credenciales.

## Ejecución local

En la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Para usar Groq localmente, crea `.streamlit/secrets.toml` con la configuración anterior (el archivo está excluido por `.gitignore`). Si no tienes clave, la aplicación sigue funcionando como catálogo.

## Contenido del proyecto

```
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
README.md
.gitignore
tests/
```

## Procedencia, límites y derechos

Las fuentes bibliográficas pertenecen al catálogo de Memoria Chilena. Se incluyen **metadatos, enlaces y vectores derivados**, no libros completos ni PDF. Los derechos de los objetos digitales y de cada obra deben consultarse en sus fichas originales. La presencia de una ficha en un minisitio indica una relación curatorial, no necesariamente una afirmación histórica sobre la obra.

Los embeddings se generaron durante la etapa 15 (modelo MiniLM de 384 dimensiones). El corpus tiene 34 fichas de cuatro ámbitos: religiosidad andina, sufragio femenino, formación de la nación y centenario de 1910. Es un **corpus piloto**, no una réplica de todo el portal.

Las citas automáticas validan que existe la ficha y que el fragmento procede de sus metadatos, **no** que el contenido histórico de un libro confirme la respuesta. La fecha catalográfica registrada no siempre es fecha de publicación; la aplicación evita dar ese salto.

Consultas procesadas por Qwen se envían al servicio Groq. Revisa los límites y el uso de API antes de hacer pública la aplicación. Para una implementación pública con alto tráfico, incorpora controles adicionales de cuota y abuso.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

## Relación con los cuadernos anteriores

Esta versión para producción ligera **no incluye los 21 notebooks**; se entrega deliberadamente independiente de los directorios de investigación. El catálogo y el índice se obtuvieron de la etapa 15. Las reglas de presentación de evidencia siguen el criterio de las etapas 19–21.
