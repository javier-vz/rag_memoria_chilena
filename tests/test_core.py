import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from core.retrieval import CatalogSearch, label_fecha
from core.generation import make_context, validate_claims, answer_with_groq

class FakeEncoder:
    def encode(self, texts, **kwargs):
        x=np.ones((len(texts),384),dtype='float32')
        return x/np.linalg.norm(x,axis=1,keepdims=True)

class FakeGroq:
    class Chat:
        class Completions:
            def create(self,**kwargs):
                payload=__import__('json').loads(kwargs['messages'][1]['content'])
                f=payload['fuentes'][0]
                title=f['campos']['titulo']
                claim={'afirmaciones':[{'texto':f'El catálogo incluye «{title}» [{f["mcid"]}].',
                                         'mcid':f['mcid'],'campo':'titulo','fragmento':title}]}
                choice=SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=__import__('json').dumps(claim)))
                return SimpleNamespace(choices=[choice])
        completions=Completions()
    chat=Chat()

class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.searcher=CatalogSearch()
    def test_catalog(self):
        self.assertEqual(self.searcher.N,34)
        self.assertEqual(self.searcher.edge_count,23)
        self.assertEqual(self.searcher.vectors.shape,(34,384))
    def test_title_anchor(self):
        hits=self.searcher.search('Vocabulario de la lengua aymara',encoder=FakeEncoder(),top_k=5)
        self.assertEqual(hits[0].doc['mcid'],'MC0014761')
    def test_suffrage(self):
        hits=self.searcher.search('El voto femenino',encoder=None,top_k=3)
        self.assertIn('voto femenino',hits[0].doc['titulo'].lower())
    def test_dates(self):
        self.assertEqual(label_fecha('start=900;end=900'),'')
        self.assertEqual(label_fecha('start=1953;end=1953'),'1953')
    def test_generation(self):
        hits=self.searcher.search('El voto femenino',encoder=FakeEncoder(),top_k=4)
        a=answer_with_groq('El voto femenino',hits,'test-secret',client=FakeGroq())
        self.assertEqual(a.status,'formato_anterior')
        self.assertIn('[MC',a.text)
    def test_reject_fake_id(self):
        hits=self.searcher.search('El voto femenino',encoder=None,top_k=3)
        good,bad=validate_claims({'afirmaciones':[{'texto':'Texto falso [MC9999999]','mcid':'MC9999999','campo':'titulo','fragmento':'Texto falso'}]},make_context(hits))
        self.assertEqual(len(good),0)
        self.assertEqual(len(bad),1)

if __name__=='__main__':unittest.main()
