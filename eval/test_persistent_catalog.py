import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from .staged_service import attach_catalog

class PersistentCatalogTests(unittest.TestCase):
    def agent(self):
        return SimpleNamespace(executor=SimpleNamespace(toolbox_router={'rain':[SimpleNamespace(tool_name='a'),SimpleNamespace(tool_name='b')],'blur':[SimpleNamespace(tool_name='c')]}),_append_episode_event=Mock(),_episode_state=lambda:{})
    def test_exact_order_and_identity(self):
        a=self.agent(); original=a.executor.toolbox_router['rain'][1]
        attach_catalog(a,{'rain':['b','a'],'blur':['c']})
        self.assertIs(a.executor.toolbox_router['rain'][0],original)
        self.assertEqual(a._append_episode_event.call_args.kwargs['outcome']['mode'],'persistent_local_services')
    def test_invalid_catalog_is_atomic(self):
        for catalog in [{'rain':['a']},{'rain':['a'],'blur':['missing']},{'rain':['a','a'],'blur':['c']},{'rain':[],'blur':['c']}]:
            a=self.agent(); original=a.executor.toolbox_router
            with self.assertRaises(ValueError):attach_catalog(a,catalog)
            self.assertIs(a.executor.toolbox_router,original)

class AdapterContractTests(unittest.TestCase):
    def test_adapter_keys_and_shapes(self):
        from .comparison_service import validate_adapter
        tensor=lambda shape:SimpleNamespace(shape=shape)
        parameters={'backbone.weight':tensor((2,2)),'llm.lora_A.weight':tensor((1,2)),'vision_proj.weight':tensor((2,2))}
        good={'llm.lora_A.weight':tensor((1,2)),'vision_proj.weight':tensor((2,2))}
        validate_adapter(good,parameters)
        with self.assertRaises(ValueError):validate_adapter({'llm.lora_A.weight':tensor((1,2))},parameters)
        with self.assertRaises(ValueError):validate_adapter(dict(good,extra=tensor((1,))),parameters)
        with self.assertRaises(ValueError):validate_adapter({**good,'vision_proj.weight':tensor((1,2))},parameters)
