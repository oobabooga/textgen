import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def load_chat_module():
    shared = types.ModuleType('modules.shared')
    shared.args = types.SimpleNamespace(multi_user=False)
    shared.model_name = 'test-model'
    shared.stop_everything = False

    utils = types.ModuleType('modules.utils')
    for name in ('delete_file', 'get_available_characters', 'get_available_users', 'sanitize_filename', 'save_file'):
        setattr(utils, name, lambda *args, **kwargs: None)

    extensions = types.ModuleType('modules.extensions')
    extensions.apply_extensions = lambda _name, value, *args, **kwargs: value

    html_generator = types.ModuleType('modules.html_generator')
    html_generator.TOOL_APPROVAL_PENDING = 'pending'
    html_generator.chat_html_wrapper = lambda *args, **kwargs: ''
    html_generator.convert_to_markdown = lambda value: value
    html_generator.extract_thinking_block = lambda value: (None, value)
    html_generator.make_thumbnail = lambda value: value

    image_utils = types.ModuleType('modules.image_utils')
    image_utils.open_image_safely = lambda *args, **kwargs: None

    logging_colors = types.ModuleType('modules.logging_colors')
    logging_colors.logger = types.SimpleNamespace(error=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

    reasoning = types.ModuleType('modules.reasoning')
    reasoning.THINKING_FORMATS = []
    reasoning.extract_reasoning = lambda value: (None, value)

    text_generation = types.ModuleType('modules.text_generation')
    text_generation.generate_reply = lambda *args, **kwargs: iter(())
    text_generation.get_encoded_length = lambda *args, **kwargs: 0
    text_generation.get_max_prompt_length = lambda *args, **kwargs: 0

    web_search = types.ModuleType('modules.web_search')
    web_search.add_web_search_attachments = lambda *args, **kwargs: None

    pil = types.ModuleType('PIL')
    pil.Image = types.SimpleNamespace()

    dependencies = {
        'modules.shared': shared,
        'modules.utils': utils,
        'modules.extensions': extensions,
        'modules.html_generator': html_generator,
        'modules.image_utils': image_utils,
        'modules.logging_colors': logging_colors,
        'modules.reasoning': reasoning,
        'modules.text_generation': text_generation,
        'modules.web_search': web_search,
        'PIL': pil,
    }
    with patch.dict(sys.modules, dependencies):
        sys.modules.pop('modules.chat', None)
        chat = importlib.import_module('modules.chat')
        try:
            yield chat
        finally:
            sys.modules.pop('modules.chat', None)


class GenerateChatReplyWrapperTest(unittest.TestCase):
    def make_state(self, history, selected_tools=True):
        return {
            'history': history,
            'unique_id': 'test',
            'name1': 'You',
            'name2': 'Assistant',
            'mode': 'instruct',
            'chat_style': 'cai-chat',
            'character_menu': 'Assistant',
            'start_with': '',
            'selected_tools': ['test_tool'] if selected_tools else [],
            'mcp_servers': '',
            'instruction_template_str': '',
            'chat_template_str': '',
        }

    def run_wrapper(self, history, *, regenerate=False, continue_=False, selected_tools=True, detects_tool_call=False):
        with load_chat_module() as chat:
            parser_calls = []
            tool_use = types.ModuleType('modules.tool_use')
            tool_use.has_mcp_config = lambda: False
            tool_use.load_tools = lambda _selected: ([{'function': {'name': 'test_tool'}}], {})
            tool_use.load_mcp_tools = lambda _servers: ([], {})
            tool_use.execute_tool = lambda *args, **kwargs: ''

            tool_parsing = types.ModuleType('modules.tool_parsing')
            tool_parsing.detect_tool_call_format = lambda _template: (None, [], False)
            tool_parsing.get_tool_call_id = lambda: 'tool-call-id'

            def parse_tool_call(answer, _tool_names, return_prefix=False, parsers=None):
                parser_calls.append(answer)
                return ([], '') if return_prefix else detects_tool_call

            tool_parsing.parse_tool_call = parse_tool_call

            state = self.make_state(history, selected_tools)
            with patch.dict(sys.modules, {'modules.tool_use': tool_use, 'modules.tool_parsing': tool_parsing}):
                with patch.object(chat, 'character_is_loaded', return_value=True), \
                     patch.object(chat, 'check_model_loaded_or_raise'), \
                     patch.object(chat, 'set_viewing_unique_id'), \
                     patch.object(chat, 'save_history'), \
                     patch.object(chat, 'chat_html_wrapper', return_value='html'), \
                     patch.object(chat, 'generate_chat_reply', side_effect=lambda *args, **kwargs: iter((state['history'],))):
                    result = list(chat.generate_chat_reply_wrapper('', state, regenerate=regenerate, _continue=continue_))

            return result, parser_calls

    def test_regenerate_with_empty_history_and_tools_is_a_noop(self):
        result, parser_calls = self.run_wrapper({'internal': [], 'visible': [], 'metadata': {}}, regenerate=True)
        self.assertEqual(result, [('html', {'internal': [], 'visible': [], 'metadata': {}})])
        self.assertEqual(parser_calls, [])

    def test_continue_with_empty_internal_history_and_tools_is_a_noop(self):
        history = {'internal': [], 'visible': [['', 'Greeting']], 'metadata': {}}
        result, parser_calls = self.run_wrapper(history, continue_=True)
        self.assertEqual(result, [('html', history)])
        self.assertEqual(parser_calls, [])

    def test_regenerate_after_clearing_history_is_a_noop(self):
        result, parser_calls = self.run_wrapper({'internal': [], 'visible': [], 'metadata': {}}, regenerate=True)
        self.assertEqual(result, [('html', {'internal': [], 'visible': [], 'metadata': {}})])
        self.assertEqual(parser_calls, [])

    def test_regenerate_with_only_a_greeting_is_safe(self):
        history = {'internal': [['<|BEGIN-VISIBLE-CHAT|>', 'Greeting']], 'visible': [['', 'Greeting']], 'metadata': {}}
        result, parser_calls = self.run_wrapper(history, regenerate=True)
        self.assertEqual(result, [('html', history)])
        self.assertEqual(parser_calls, ['Greeting', 'Greeting'])

    def test_regenerate_after_a_response_keeps_tool_parsing_reachable(self):
        history = {'internal': [['Question', 'Answer']], 'visible': [['Question', 'Answer']], 'metadata': {}}
        result, parser_calls = self.run_wrapper(history, regenerate=True)
        self.assertEqual(result, [('html', history)])
        self.assertEqual(parser_calls, ['Answer', 'Answer'])

    def test_detected_tool_call_reaches_the_tool_parser(self):
        history = {'internal': [['Question', '<tool_call>']], 'visible': [['Question', '<tool_call>']], 'metadata': {}}
        result, parser_calls = self.run_wrapper(history, regenerate=True, detects_tool_call=True)
        self.assertEqual(result, [('html', history)])
        self.assertEqual(parser_calls, ['<tool_call>', '<tool_call>'])

    def test_empty_history_without_tools_is_a_noop(self):
        result, parser_calls = self.run_wrapper({'internal': [], 'visible': [], 'metadata': {}}, regenerate=True, selected_tools=False)
        self.assertEqual(result, [('html', {'internal': [], 'visible': [], 'metadata': {}})])
        self.assertEqual(parser_calls, [])


if __name__ == '__main__':
    unittest.main()
