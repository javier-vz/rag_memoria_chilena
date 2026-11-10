# Explorador de Memoria Chilena

Una herramienta de consulta documental desarrollada por interés en la exploración de fuentes históricas. **No está afiliada a Memoria Chilena ni representa a la Biblioteca Nacional de Chile.** Los datos bibliográficos se enlazan a sus fichas originales.

## Funciones

- Interfaz sobria y legible, con **tres pestañas horizontales**: Consultar, Colección y Acerca de.
- Modo **Rápida** (predeterminado) para mostrar fichas sin llamar a Groq.
- Opción **Con Qwen** para redactar respuestas basadas únicamente en los metadatos recuperados.
- Búsqueda léxica TF-IDF, relaciones RDF y búsqueda semántica MiniLM opcional. La primera descarga del modelo MiniLM puede tardar; para consultas inmediatas, deja desactivada esa opción.
- Selección de 34 fichas de cuatro colecciones, con enlaces al sitio fuente. **No incluye el contenido íntegro de las obras.**
- Citas sometidas a comprobaciones formales de identificador y fragmentos catalográficos; esto no verifica cada afirmación histórica.

## Publicar en GitHub y Streamlit Cloud

1. Descomprime el ZIP y sube el **contenido** de `memoria-chilena-streamlit-v3/` a tu repositorio de GitHub, dejando `app.py` en la raíz.
2. Ve a [Streamlit Community Cloud](https://share.streamlit.io/) y selecciona el repositorio, la rama y `app.py`.
3. En **App settings → Secrets**, añade:

   ```toml
   GROQ_API_KEY = "tu_clave"
   ```

4. Guarda los secretos y reinicia la app si es necesario. La clave no se incluye en GitHub.

No subas `.streamlit/secrets.toml`, archivos `.env`, PDF ni exportaciones de trabajo. El `.gitignore` los excluye. `.streamlit/secrets.toml.example` es una plantilla sin credenciales.

## Ejecutar localmente

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Para Qwen en local, define `GROQ_API_KEY` como variable de entorno o crea `.streamlit/secrets.toml` (archivo excluido de Git).

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

## Alcance del corpus

Esta versión contiene **34 fichas catalográficas** y **23 relaciones específicas** de Memoria Chilena. Se trabaja con metadatos, no con documentos digitalizados completos. Las materias y asociaciones curatoriales no prueban por sí mismas hechos históricos; las fechas no siempre son fechas de publicación. Verifica cada afirmación de interés en la ficha y en el documento original cuando esté disponible.

## Pruebas

```bash
python -m unittest discover -s tests -v
```
