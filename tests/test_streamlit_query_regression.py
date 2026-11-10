"""Regresión del error de serialización en la búsqueda."""
import ast
import unittest
from pathlib import Path
from core.retrieval import CatalogSearch


class TestQueryCaching(unittest.TestCase):
    def test_search_not_cache_data(self):
        app = Path(__file__).resolve().parents[1] / 'app.py'
        module = ast.parse(app.read_text(encoding='utf8'))
        func = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'buscar')
        self.assertEqual(func.decorator_list, [], 'No almacenar los resultados Hit con st.cache_data')

    def test_example_queries_produce_results(self):
        searcher = CatalogSearch()
        samples = [
            '¿Qué obras tratan el sufragio femenino en Chile?',
            '¿Qué fuentes documentan los inicios del Instituto Nacional?',
            '¿Qué textos coloniales están asociados con las lenguas indígenas andinas?',
            '¿Qué documentos permiten estudiar las celebraciones del centenario de 1910?',
        ]
        for q in samples:
            with self.subTest(q=q):
                hits = searcher.search(q, encoder=None, use_graph=True, top_k=4)
                self.assertEqual(len(hits), 4)
                self.assertTrue(all(h.doc['mcid'] for h in hits))


if __name__ == '__main__':
    unittest.main()
