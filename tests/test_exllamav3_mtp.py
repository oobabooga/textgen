"""Loader wiring tests without CUDA or an installed ExLlamaV3."""
import ast
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


source = Path('modules/exllamav3.py').read_text()
tree = ast.parse(source)
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_integrated_mtp')
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Exllamav3Model')
loader = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'from_pretrained')
unload = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'unload')
subset = ast.Module(body=[helper, ast.ClassDef(name='Loader', bases=[], keywords=[], body=[loader, unload], decorator_list=[])], type_ignores=[])
ast.fix_missing_locations(subset)


class TestMTP(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.args = types.SimpleNamespace(model_dir=self.temp.name, ctx_size=1024, cache_type='q8', gpu_split='8,6', enable_tp=True, tp_backend='native', model_draft=None, draft_max=3, exllamav3_mtp=True)
        self.config = types.SimpleNamespace(model_classes={'mtp': object}, stc=types.SimpleNamespace(has_tensor=lambda key: key == 'mtp.fc.trellis'), config_dict={})
        self.instances = []
        def model_factory(config, component='text'):
            obj = Mock(name=component)
            obj.component = component
            self.instances.append(obj)
            return obj
        self.model = types.SimpleNamespace(from_config=Mock(side_effect=model_factory))
        self.cache = Mock(side_effect=lambda model, **kw: types.SimpleNamespace(model=model, kwargs=kw))
        self.generator = Mock(return_value=object())
        self.parallel = Mock()
        self.ns = dict(Path=Path, shared=types.SimpleNamespace(args=self.args), Config=types.SimpleNamespace(from_directory=Mock(return_value=self.config)), Model=self.model, Cache=self.cache, CacheLayer_fp16=object(), CacheLayer_quant=object(), Tokenizer=types.SimpleNamespace(from_config=Mock(return_value=object())), Generator=self.generator, ConcurrentGenerator=Mock(return_value=self.parallel), logger=logging.getLogger('test'))
        exec(compile(subset, 'modules/exllamav3.py', 'exec'), self.ns)
        mm = types.ModuleType('exllamav3.tokenizer.mm_embedding')
        mm.FIRST_MM_EMBEDDING_INDEX = 1
        mm.global_allocator = types.SimpleNamespace(next_token_index=0)
        self.modules_patch = patch.dict(sys.modules, {'exllamav3': types.ModuleType('exllamav3'), 'exllamav3.tokenizer': types.ModuleType('exllamav3.tokenizer'), 'exllamav3.tokenizer.mm_embedding': mm})
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    def load(self):
        return self.ns['Loader'].from_pretrained('target')[0]

    def test_integrated_mtp_with_tp_and_cache(self):
        result = self.load()
        main, mtp = self.instances
        mtp.load.assert_called_once_with(progressbar=True, use_per_device=[8.0, 6.0])
        main.load.assert_called_once_with(progressbar=True, use_per_device=[8.0, 6.0], tensor_p=True, tp_backend='native', draft_model=mtp)
        args = self.generator.call_args.kwargs
        self.assertIs(args['draft_model'], mtp)
        self.assertEqual(args['num_draft_tokens'], 3)
        self.assertEqual(args['cache'].kwargs['max_history'], 3)
        self.assertEqual(args['draft_cache'].kwargs['max_history'], 3)
        result.unload()
        self.parallel.stop.assert_called_once()
        mtp.unload.assert_called_once()
        main.unload.assert_called_once()

    def test_missing_component_and_weights(self):
        self.config.model_classes = {}
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.load()
        self.config.model_classes = {'mtp': object}
        self.config.stc.has_tensor = lambda key: False
        with self.assertRaisesRegex(ValueError, 'weights are missing'):
            self.load()
        self.model.from_config.assert_not_called()

    def test_conflict_and_invalid_depth(self):
        self.args.model_draft = 'other'
        with self.assertRaisesRegex(ValueError, 'external draft'):
            self.load()
        self.args.model_draft = None
        self.args.draft_max = 0
        with self.assertRaisesRegex(ValueError, 'at least 1'):
            self.load()

    def test_disabled_and_external_draft(self):
        self.args.exllamav3_mtp = False
        self.config.model_classes = {}
        result = self.load()
        self.assertIsNone(result.draft_model)
        self.assertEqual(self.generator.call_args.kwargs['num_draft_tokens'], 0)
        self.assertNotIn('max_history', result.cache.kwargs)
        self.assertNotIn('draft_model', self.instances[0].load.call_args.kwargs)
        Path(self.temp.name, 'draft').mkdir()
        self.args.model_draft = 'draft'
        result = self.load()
        self.assertIs(result.draft_model, self.instances[-1])
        self.assertEqual(self.generator.call_args.kwargs['num_draft_tokens'], 3)

    def test_failure_releases_models(self):
        self.generator.side_effect = RuntimeError('generator failed')
        with self.assertRaisesRegex(RuntimeError, 'generator failed'):
            self.load()
        for model in self.instances:
            model.unload.assert_called_once()

    def test_unquantized_weight_is_detected(self):
        self.config.stc.has_tensor = lambda key: key == 'mtp.fc.weight'
        self.load()


if __name__ == '__main__':
    unittest.main()
