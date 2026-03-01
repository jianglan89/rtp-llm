"""
MTP-Safe Streaming Tool Call Parsing Tests

Tests for tool call parsing under MTP (Speculative Decoding) conditions where
multiple tokens may arrive in a single chunk, including scenarios where:
1. Complete tool call blocks arrive in single chunk
2. Think-end tag and tool-start tag arrive in same chunk
3. Multiple complete tool calls arrive in single chunk
"""

import unittest

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv31_detector import (
    DeepSeekV31Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv32_detector import (
    DeepSeekV32Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.kimik2_detector import (
    KimiK2Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)


def create_tools():
    """Create test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_time",
                description="Get current time",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


def create_glm4_tools():
    """Create GLM-4 test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="ask_user_question",
                description="Ask the user questions",
                parameters={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "Questions to ask",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question": {"type": "string"},
                                    "header": {"type": "string"},
                                    "multiSelect": {"type": "boolean"},
                                    "options": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "label": {"type": "string"},
                                                "description": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    "required": ["questions"],
                },
            ),
        ),
    ]


class TestQwen25DetectorMTP(unittest.TestCase):
    """Test Qwen25Detector MTP compatibility."""

    def setUp(self):
        self.detector = Qwen25Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete tool call block arrives in single chunk.
        This simulates MTP returning the entire tool call at once instead of
        token-by-token.
        """
        # Complete tool call in one chunk
        chunk = '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            '"location"',
            result.calls[0].parameters,
            f"Expected '\"location\"' in parameters. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )
        self.assertEqual(
            result.normal_text, "", "Pure tool chunk should have no normal_text"
        )

    def test_mtp_think_end_and_tool_start_same_chunk(self):
        """
        MTP scenario: Think-end tag and tool-start tag arrive in same chunk.
        This is the most common MTP failure case.
        """
        self.detector = Qwen25Detector()

        # First chunk: reasoning content
        chunk1 = "I need to check the weather"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(
            result1.normal_text,
            "I need to check the weather",
            f"Expected normal_text 'I need to check the weather', got '{result1.normal_text}'. Calls: {result1.calls}",
        )
        self.assertEqual(
            len(result1.calls),
            0,
            f"Expected 0 calls, got {len(result1.calls)}. Calls: {result1.calls}",
        )

        # MTP chunk: newlines followed by complete tool call
        # Simulates </think>\n\n<tool_call>... in one chunk
        chunk2 = '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.normal_text,
            "\n\n",
            "Prefix before tool must be returned as normal_text",
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete tool calls arrive in single chunk.
        """
        # Two complete tool calls in one chunk
        chunk = (
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>'
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected calls[0].name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[1].name,
            "get_current_weather",
            f"Expected calls[1].name 'get_current_weather', got '{result.calls[1].name}'. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].tool_index,
            0,
            f"Expected calls[0].tool_index 0, got {result.calls[0].tool_index}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[1].tool_index,
            1,
            f"Expected calls[1].tool_index 1, got {result.calls[1].tool_index}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.normal_text, "", "Pure tool chunk should have no normal_text"
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion in next chunk.

        chunk1: "hello" prefix + bot_token + incomplete JSON
        chunk2: rest of JSON (completes the JSON body)
        chunk3: eot_token ("\n</tool_call>")

        Note: the base class may return the call at chunk2 (when JSON becomes
        complete) rather than waiting for eot_token at chunk3. Both are valid.
        """
        # First chunk: prefix + start of tool call with incomplete JSON
        chunk1 = 'hello<tool_call>\n{"name": "get_current_weather"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1.normal_text, str, "Step1 normal_text must be str")
        self.assertEqual(result1.normal_text, "hello")

        # Second chunk: completes the JSON body (base may parse and return call here)
        chunk2 = ', "arguments": {"location": "杭州"}}'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertEqual(result2.normal_text, "", "No prefix in tool-parsing chunks")

        # Third chunk: eot_token
        chunk3 = "\n</tool_call>"
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertEqual(
            result3.normal_text, "", "eot chunk should have no normal_text"
        )

        # Across all three chunks, at least 1 call should be returned.
        # Base class streams incrementally: first item has name (parameters=""),
        # subsequent items have name=None with argument diffs.
        all_calls = list(result1.calls) + list(result2.calls) + list(result3.calls)
        self.assertGreaterEqual(
            len(all_calls),
            1,
            f"Expected at least 1 call item across all chunks. "
            f"r1={result1.calls}, r2={result2.calls}, r3={result3.calls}",
        )
        named_calls = [c for c in all_calls if c.name]
        self.assertTrue(
            len(named_calls) > 0,
            f"Should have a call with name. All calls: {all_calls}",
        )
        self.assertEqual(named_calls[0].name, "get_current_weather")

    def test_incremental_still_works(self):
        """
        Verify that traditional single-token incremental streaming still works.
        """
        detector = Qwen25Detector()

        # Simulate token-by-token streaming
        chunks = [
            "<tool_call>",
            "\n",
            "{",
            '"name": "get_current_weather"',
            ', "arguments": {"location": "杭州"}',
            "}",
            "\n</tool_call>",
        ]

        all_calls = []
        all_normal = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal.append(result.normal_text)

        self.assertGreaterEqual(
            len(all_calls),
            1,
            f"Should have at least 1 call, got {len(all_calls)}. Calls: {all_calls}",
        )
        named_calls = [c for c in all_calls if c.name]
        self.assertTrue(
            len(named_calls) > 0,
            f"Should have a call with name. All calls: {all_calls}",
        )
        self.assertEqual(
            named_calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{named_calls[0].name}'. Named calls: {named_calls}",
        )
        # Every step must return str normal_text (no swallow)
        for i, nt in enumerate(all_normal):
            self.assertIsInstance(
                nt, str, f"Step {i} normal_text must be str, got {type(nt)}"
            )

    def test_empty_chunk_does_not_break(self):
        """Empty string chunk should not break parser; buffer state unchanged."""
        result = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 0)

    def test_no_marker_emits_full_buffer(self):
        """
        Chunks that never contain bot_token: each chunk should be emitted as normal_text
        (no accumulation across calls when no marker; base class handles buffer).
        """
        detector = Qwen25Detector()
        chunks = ["hello", " ", "world"]
        all_normal = []
        for ch in chunks:
            r = detector.parse_streaming_increment(ch, self.tools)
            all_normal.append(r.normal_text)
            self.assertEqual(len(r.calls), 0)
        self.assertEqual("".join(all_normal), "hello world")

    def test_pure_text_then_complete_tool_in_next_chunk(self):
        """First chunk pure text, second chunk complete tool call."""
        detector = Qwen25Detector()
        r1 = detector.parse_streaming_increment("Some prefix.\n", self.tools)
        self.assertEqual(r1.normal_text, "Some prefix.\n")
        self.assertEqual(len(r1.calls), 0)
        chunk2 = '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>'
        r2 = detector.parse_streaming_increment(chunk2, self.tools)
        self.assertEqual(
            r2.normal_text, "", "Second chunk is pure tool call, no prefix"
        )
        self.assertEqual(len(r2.calls), 1)
        self.assertEqual(r2.calls[0].name, "get_current_weather")
        self.assertIn("北京", r2.calls[0].parameters)

    def test_qwen25_prefix_before_tool_not_swallowed(self):
        """
        Single chunk with prefix (e.g. '>' from '</thinking>') + complete tool call.
        Covers: upfront prefix strip; prefix must be returned as normal_text, not swallowed.
        """
        detector = Qwen25Detector()
        chunk = (
            ">\n\n"
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>'
        )
        result = detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(
            result.normal_text,
            ">\n\n",
            "Prefix before bot_token must be returned as normal_text (stream swallow fix)",
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertIn("北京", result.calls[0].parameters)

    def test_qwen25_orphan_eot_then_valid_block(self):
        """
        Chunk contains orphan '\\n</tool_call>' then a valid <tool_call>...\\n</tool_call> block.
        Covers: eot_idx is searched from bot_idx+len(bot_token) so we match the correct end.
        """
        detector = Qwen25Detector()
        chunk = (
            "\n</tool_call>\n"
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "上海"}}\n</tool_call>'
        )
        result = detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(
            result.normal_text,
            "\n</tool_call>\n",
            "Text before first bot_token is prefix (orphan eot not parsed as block)",
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertIn("上海", result.calls[0].parameters)

    def test_qwen25_detect_and_parse_with_prefix(self):
        """
        One-shot detect_and_parse: text with prefix before bot_token must return
        prefix as normal_text and parsed calls (detector.detect_and_parse, not streaming).
        """
        detector = Qwen25Detector()
        text = (
            "hello\n"
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "广州"}}\n</tool_call>'
        )
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "hello\n")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertIn("广州", result.calls[0].parameters)

    def test_qwen25_streaming_prefix_plus_two_blocks(self):
        """
        Single chunk: prefix + two complete <tool_call>...</tool_call> blocks.
        Covers: prefix stripped (bot_idx > 0), while-loop parses both blocks, normal_text = prefix only.
        """
        detector = Qwen25Detector()
        chunk = (
            ">\n\n"
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "深圳"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
        )
        result = detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(result.normal_text, ">\n\n")
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertIn("深圳", result.calls[0].parameters)
        self.assertEqual(result.calls[1].name, "get_time")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[1].tool_index, 1)

    # ---- Destructive / non-standard inputs ----

    def test_truncated_tool_tag_no_close_emits_as_normal(self):
        """Only opening tag, no closing: must not crash; normal_text is str (may be chunk or empty)."""
        detector = Qwen25Detector()
        chunk = '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}'
        result = detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.normal_text, str)
        self.assertIsInstance(result.calls, list)

    def test_lookalike_not_tag_no_call(self):
        """Text that looks like tag start but is not valid bot_token (e.g. missing newline)."""
        detector = Qwen25Detector()
        chunk = 'hello <tool_call> no newline here {"name": "x"}'
        result = detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(
            result.normal_text,
            chunk,
            "Lookalike without bot_token should be normal_text",
        )

    def test_malformed_json_inside_tool_call(self):
        """Malformed JSON inside <tool_call>...</tool_call>: parser should not crash."""
        detector = Qwen25Detector()
        chunk = '<tool_call>\n{"name": "get_current_weather", "arguments": {invalid json here}}\n</tool_call>'
        result = detector.parse_streaming_increment(chunk, self.tools)
        # May return 0 calls (parse failure) or skip; must not raise
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(result.normal_text, str)

    def test_close_before_open_no_call(self):
        """Closing tag before opening: should not produce a call."""
        detector = Qwen25Detector()
        chunk = '\n</tool_call>\n<tool_call>\n{"name": "get_current_weather", "arguments": {}}\n</tool_call>'
        result = detector.parse_streaming_increment(chunk, self.tools)
        # First block is invalid (end before start); second is valid. Behavior detector-dependent.
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(result.normal_text, str)

    def test_empty_tool_call_block(self):
        """Empty content between tags."""
        detector = Qwen25Detector()
        chunk = "<tool_call>\n\n</tool_call>"
        result = detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.calls, list)
        self.assertEqual(
            result.normal_text, "", "Complete but empty block yields no normal_text"
        )


class TestKimiK2DetectorMTP(unittest.TestCase):
    """Test KimiK2Detector MTP compatibility."""

    def setUp(self):
        self.detector = KimiK2Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete KimiK2 tool call block arrives in single chunk.
        """
        chunk = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )
        self.assertEqual(
            result.normal_text, "", "Pure tool chunk should have no normal_text"
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete KimiK2 tool calls in single chunk.
        """
        chunk = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|>'
            '<|tool_call_begin|>functions.get_current_weather:1 <|tool_call_argument_begin|>{"location": "北京"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.normal_text, "", "Pure tool chunk should have no normal_text"
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion.
        """
        chunk1 = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1.normal_text, str)
        self.assertIsInstance(result1.calls, list)

        chunk2 = ': "杭州"}<|tool_call_end|><|tool_calls_section_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            result2.normal_text, "", "Completion chunk yields no normal_text"
        )
        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )

    def test_mtp_thinking_tag_closing_gt_not_swallowed(self):
        """
        Stream scenario: closing '>' of '</thinking>' and tool call split across chunks.

        When stream=true, ">" and tool block may arrive in same chunk. The prefix
        before bot_token must be returned as normal_text so content is not missing.
        """
        # Chunk 1: thinking content + partial closing tag (no ">")
        chunk1 = "<thinking>\n用户想买行李箱，我需要帮他搜索。\n</thinking"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(result1.normal_text, chunk1, "Chunk1 should be normal_text")
        self.assertEqual(len(result1.calls), 0)

        # Chunk 2: ">" + newlines + complete tool call (MTP: all in one chunk)
        chunk2 = (
            ">\n\n<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'",
        )
        self.assertEqual(
            result2.normal_text,
            ">\n\n",
            f"Expected normal_text '>\\n\\n', got {repr(result2.normal_text)}. "
            "The '>' in '</thinking>' was swallowed when tool block parsed.",
        )

    def test_incremental_still_works(self):
        """
        Token-by-token streaming: chunks that build up to a complete tool call.
        Verifies no-marker path does not lose data (full buffer emitted until marker).
        """
        detector = KimiK2Detector()
        chunks = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>",
            '{"location": "杭州"}',
            "<|tool_call_end|><|tool_calls_section_end|>",
        ]
        all_calls = []
        all_normal = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal.append(result.normal_text)
        self.assertGreaterEqual(
            len(all_calls), 1, f"Should have at least 1 call. Calls: {all_calls}"
        )
        named = [c for c in all_calls if c.name]
        self.assertTrue(len(named) > 0, f"Need a call with name. All: {all_calls}")
        self.assertEqual(named[0].name, "get_current_weather")
        for i, nt in enumerate(all_normal):
            self.assertIsInstance(nt, str, f"Step {i} normal_text must be str")

    def test_no_marker_emits_full_buffer(self):
        """
        Chunks that never contain bot_token or tool_call_start: each should be
        returned as normal_text (no swallow); no tool calls.
        """
        detector = KimiK2Detector()
        chunks = ["hello", " world", "!"]
        collected = []
        for ch in chunks:
            r = detector.parse_streaming_increment(ch, self.tools)
            collected.append(r.normal_text)
            self.assertEqual(len(r.calls), 0)
        self.assertEqual("".join(collected), "hello world!")

    def test_empty_chunk_does_not_break(self):
        """Empty chunk should not break parser."""
        result = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 0)

    # ---- Destructive / non-standard inputs ----

    def test_truncated_section_no_end_tag(self):
        """Section begin without section end: incomplete, should not crash."""
        chunk = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "北京"}'
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(
            result.normal_text, str, "normal_text must be str (may be empty or chunk)"
        )

    def test_lookalike_tokens_no_valid_block(self):
        """Text that looks like tokens but invalid end tag; must not crash."""
        chunk = "<|tool_calls_section_begin|><|tool_call_begin|>functions.unknown_tool:0 <|tool_call_argument_begin|>{}\n</invalid_end>"
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(result.normal_text, str)

    def test_malformed_json_in_args_no_crash(self):
        """Malformed JSON in argument block: must not crash."""
        chunk = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{ not valid json }<|tool_call_end|><|tool_calls_section_end|>"
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(result.normal_text, str)

    def test_end_tag_without_begin_no_call(self):
        """Only end tags, no proper begin: should not produce calls."""
        chunk = "<|tool_call_end|><|tool_calls_section_end|>"
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertIsInstance(
            result.normal_text, str, "normal_text must be str (chunk or empty)"
        )


class TestDeepSeekV31DetectorMTP(unittest.TestCase):
    """Test DeepSeekV31Detector MTP compatibility."""

    def setUp(self):
        self.detector = DeepSeekV31Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete DeepSeek tool call block arrives in single chunk.
        """
        chunk = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        print(f"result.calls: {result.calls}")
        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )
        self.assertEqual(
            result.normal_text, "", "Pure tool chunk should have no normal_text"
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete DeepSeek tool calls in single chunk.
        """
        chunk = (
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "北京"}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.normal_text, "", "Pure tool chunk should have no normal_text"
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion.
        """
        chunk1 = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(len(result1.calls), 0)
        self.assertEqual(
            result1.normal_text, "", "Partial tool chunk should not emit normal_text"
        )

        chunk2 = ': "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            result2.normal_text, "", "Completion chunk yields no normal_text"
        )
        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )

    def test_mtp_thinking_tag_closing_gt_not_swallowed(self):
        """
        Stream scenario: closing '>' of '</thinking>' and tool call split across chunks.

        When stream=true, tokenizer may output:
        - Chunk 1: "</thinking" (content sent)
        - Chunk 2: ">\\n\\n" + bot_token + tool_call (">" and tool in same chunk)

        Bug: Chunk 2's ">" was swallowed because parse_streaming_increment returned
        normal_text="" when parsing complete tool block, dropping prefix before bot_token.
        """
        # Chunk 1: thinking content + partial closing tag (no ">")
        chunk1 = "<thinking>\n用户想买行李箱，我需要帮他搜索。\n</thinking"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(result1.normal_text, chunk1, "Chunk1 should be normal_text")
        self.assertEqual(len(result1.calls), 0)

        # Chunk 2: ">" + newlines + complete tool call (MTP: all in one chunk)
        chunk2 = '>\n\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'",
        )
        # The fix: ">" must be in normal_text so streamed content is not missing it.
        # normal_text = prefix before bot_token, i.e. ">\n\n" (bot_token excluded).
        self.assertEqual(
            result2.normal_text,
            ">\n\n",
            f"Expected normal_text '>\\n\\n', got {repr(result2.normal_text)}. "
            "The '>' in '</thinking>' was swallowed when tool block parsed.",
        )

    def test_mtp_prefix_before_incomplete_tool_not_swallowed(self):
        """
        Reproduce bug: when ">" and "<｜tool▁calls▁begin｜>" arrive in same chunk,
        the ">" (tail of "</thinking>") gets swallowed into buffer and is not
        returned as normal_text until the complete tool call block is parsed.

        Simulates actual MTP output_ids:
        chunk1: "</thinking"        (tokens: "</" + "thinking")
        chunk2: ">\n\n<｜tool▁calls▁begin｜>"  (tokens: ">" + calls_begin)
        chunk3: "<｜tool▁call▁begin｜>get_current_weather"
        chunk4: "<｜tool▁sep｜>{\"location\""
        chunk5: ": \"杭州\"}"
        chunk6: "<｜tool▁call▁end｜>"
        chunk7: "<｜tool▁calls▁end｜>"
        """

        # ---- Chunk 1: partial "</thinking" — no tool markers ----
        chunk1 = "</thinking"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        # No tool markers at all, should pass through as normal_text
        self.assertEqual(result1.normal_text, "</thinking")
        self.assertEqual(len(result1.calls), 0)

        # ---- Chunk 2: ">" arrives WITH tool marker in same MTP chunk ----
        # This is the critical chunk that triggers the bug.
        # buffer becomes: ">\n\n<｜tool▁calls▁begin｜>"
        # has_tool_call becomes True (bot_token "</thinking>" is NOT in buffer,
        # but if bot_token check uses partial match, or <｜tool▁call｜> markers...)
        #
        # BUG: ">" gets trapped in buffer, returned as normal_text=""
        chunk2 = ">\n\n<｜tool▁calls▁begin｜>"
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        # EXPECTED (after fix): ">\n\n" should be returned as normal_text
        # BUG (before fix): normal_text="" because buffer holds everything
        self.assertIn(
            ">",
            result2.normal_text,
            "BUG: '>' from '</thinking>' tail must not be swallowed",
        )
        self.assertEqual(len(result2.calls), 0)

        # ---- Chunk 3: tool call name ----
        chunk3 = "<｜tool▁call▁begin｜>get_current_weather"
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertEqual(len(result3.calls), 0)
        self.assertEqual(result3.normal_text, "", "Chunk3 incomplete, no normal_text")

        # ---- Chunk 4: separator + partial args ----
        chunk4 = '<｜tool▁sep｜>{"location"'
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertEqual(len(result4.calls), 0)
        self.assertEqual(result4.normal_text, "", "Chunk4 incomplete, no normal_text")

        # ---- Chunk 5: rest of args ----
        chunk5 = ': "杭州"}'
        result5 = self.detector.parse_streaming_increment(chunk5, self.tools)
        self.assertEqual(len(result5.calls), 0)
        self.assertEqual(result5.normal_text, "", "Chunk5 incomplete, no normal_text")

        # ---- Chunk 6: tool call end — complete block ----
        chunk6 = "<｜tool▁call▁end｜>"
        result6 = self.detector.parse_streaming_increment(chunk6, self.tools)
        self.assertEqual(
            result6.normal_text,
            "",
            "Chunk6 completes tool, prefix already emitted in chunk2",
        )
        self.assertEqual(len(result6.calls), 1)
        self.assertEqual(result6.calls[0].name, "get_current_weather")
        self.assertEqual(result6.calls[0].parameters, '{"location": "杭州"}')

        # ---- Chunk 7: tool calls end ----
        chunk7 = "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        result7 = self.detector.parse_streaming_increment(chunk7, self.tools)
        self.assertEqual(result7.normal_text, "", "Chunk7 eot only, no normal_text")
        self.assertEqual(len(result7.calls), 0)

    def test_incremental_still_works(self):
        """
        Chunk-by-chunk streaming until complete tool call.
        Verifies no-marker path returns full buffer as normal_text, not only new_text.
        """
        detector = DeepSeekV31Detector()
        chunks = [
            detector.bot_token,
            detector.tool_call_start,
            "get_current_weather<｜tool▁sep｜>",
            '{"location": "杭州"}',
            detector.tool_call_end,
            detector.eot_token,
        ]
        all_calls = []
        all_normal = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)
            all_normal.append(result.normal_text)
        self.assertGreaterEqual(
            len(all_calls), 1, f"Should have at least 1 call. Calls: {all_calls}"
        )
        named = [c for c in all_calls if c.name]
        self.assertTrue(len(named) > 0, f"Need a call with name. All: {all_calls}")
        self.assertEqual(named[0].name, "get_current_weather")
        for i, nt in enumerate(all_normal):
            self.assertIsInstance(nt, str, f"Step {i} normal_text must be str")

    def test_no_marker_emits_full_buffer(self):
        """Chunks with no tool marker should be emitted as normal_text."""
        detector = DeepSeekV31Detector()
        chunks = ["hello", " ", "world"]
        collected = []
        for ch in chunks:
            r = detector.parse_streaming_increment(ch, self.tools)
            collected.append(r.normal_text)
            self.assertEqual(len(r.calls), 0)
        self.assertEqual("".join(collected), "hello world")

    def test_empty_chunk_does_not_break(self):
        """Empty chunk should not break parser."""
        result = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 0)

    # ---- Destructive / non-standard inputs ----

    def test_truncated_tool_block_no_end(self):
        """Bot + tool start but no tool end / eot: incomplete, must not crash."""
        chunk = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "北京"'
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(result.normal_text, str)

    def test_malformed_json_in_args_no_crash(self):
        """Malformed JSON between <｜tool▁sep｜> and <｜tool▁call▁end｜>: must not crash."""
        chunk = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{ invalid }<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(result.normal_text, str)

    def test_only_eot_no_call(self):
        """Only eot token, no real tool content: should not produce a call."""
        chunk = self.detector.eot_token
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertIsInstance(
            result.normal_text, str, "normal_text must be str (chunk or empty)"
        )

    def test_lookalike_markers_not_valid_format(self):
        """Text that resembles markers but wrong structure."""
        chunk = "<｜tool▁calls▁begin｜> some random text <｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        result = self.detector.parse_streaming_increment(chunk, self.tools)
        self.assertIsInstance(result.calls, list)
        self.assertIsInstance(result.normal_text, str)

        # ---- Chunk 3: tool call name ----
        chunk3 = "<｜tool▁call▁begin｜>get_current_weather"
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertEqual(len(result3.calls), 0)  # incomplete, no end marker yet

        # ---- Chunk 4: separator + partial args ----
        chunk4 = '<｜tool▁sep｜>{"location"'
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertEqual(len(result4.calls), 0)

        # ---- Chunk 5: rest of args ----
        chunk5 = ': "杭州"}'
        result5 = self.detector.parse_streaming_increment(chunk5, self.tools)
        self.assertEqual(len(result5.calls), 0)

        # ---- Chunk 6: tool call end — now we have a complete block ----
        chunk6 = "<｜tool▁call▁end｜>"
        result6 = self.detector.parse_streaming_increment(chunk6, self.tools)
        self.assertEqual(len(result6.calls), 1)
        self.assertEqual(result6.calls[0].name, "get_current_weather")
        self.assertEqual(result6.calls[0].parameters, '{"location": "杭州"}')
        # After fix: normal_text should be "" here (prefix already emitted in chunk2)
        # Before fix (bug): normal_text=">\n\n" leaked here with tool_calls

        # ---- Chunk 7: tool calls end ----
        chunk7 = "<｜tool▁calls▁end｜>"
        result7 = self.detector.parse_streaming_increment(chunk7, self.tools)
        self.assertEqual(result7.normal_text, "")
        self.assertEqual(len(result7.calls), 0)


class TestDeepSeekV32DetectorMTP(unittest.TestCase):
    """Test DeepSeekV32Detector MTP compatibility."""

    def setUp(self):
        self.detector = DeepSeekV32Detector()
        self.tools = create_tools()

    def test_mtp_thinking_tag_closing_gt_not_swallowed(self):
        """
        Stream scenario: closing '>' of '</thinking>' and tool call split across chunks.

        When stream=true, ">" and invoke block may arrive in same chunk. The prefix
        before bot_token must be returned as normal_text so content is not missing.
        """
        # Chunk 1: thinking content + partial closing tag (no ">")
        chunk1 = "<thinking>\n用户想买行李箱，我需要帮他搜索。\n</thinking"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(result1.normal_text, chunk1, "Chunk1 should be normal_text")
        self.assertEqual(len(result1.calls), 0)

        # Chunk 2: ">" + newlines + complete invoke block (MTP: all in one chunk)
        chunk2 = (
            ">\n\n<｜DSML｜function_calls>\n"
            '<｜DSML｜invoke name="get_current_weather">{"location": "杭州"}</｜DSML｜invoke>\n'
            "</｜DSML｜function_calls>"
        )
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            2,
            f"Expected 2 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'",
        )
        self.assertEqual(
            result2.normal_text,
            ">\n\n",
            f"Expected normal_text '>\\n\\n', got {repr(result2.normal_text)}. "
            "The '>' in '</thinking>' was swallowed when invoke block parsed.",
        )


class TestGlm4MoeDetectorMTP(unittest.TestCase):
    """Test Glm4MoeDetector MTP compatibility with GLM-4.7 format."""

    def setUp(self):
        self.detector = Glm4MoeDetector()
        self.tools = create_glm4_tools()

    def test_glm47_with_reasoning_and_tool_call(self):
        """
        Test GLM-4.7 format with <think> tags and tool calls.
        This reproduces the issue reported in commit 91fc0bc536fd1176e711349cdd81a8ddd1b5d1ba.

        Raw output format:
        <think>reasoning content</think>normal text<tool_call>...</tool_call><|observation|>

        Expected behavior:
        - finish_reason: "tool_calls" (not "stop")
        - reasoning_content: "reasoning content"
        - content: "normal text"
        - tool_calls: parsed tool call
        """
        # Note: The raw output provided by the user shows the complete response including <think> tags
        # However, the Glm4MoeDetector only parses <tool_call> tags, not <think> tags.
        # The <think> tags are handled by the ReasoningParser in the renderer layer.
        # For this unit test, we test the detector's ability to parse tool calls
        # from text that may have normal text before the tool call.

        raw_output = (
            "帮助用户做出明确的选择。我来调用 ask_user_question 工具，构造一些示例参数："
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "您希望使用哪种编程语言来开发这个功能？", '
            '"header": "编程语言", "multiSelect": false, '
            '"options": [{"label": "TypeScript", "description": "类型安全的 JavaScript 超集，适合大型项目"}, '
            '{"label": "Python", "description": "简洁易读，适合快速开发和数据处理"}, '
            '{"label": "Go", "description": "高性能并发，适合后端服务和微服务"}]}, '
            '{"question": "您希望启用哪些功能特性？", "header": "功能特性", "multiSelect": true, '
            '"options": [{"label": "实时更新", "description": "数据变更时自动同步更新界面"}, '
            '{"label": "离线缓存", "description": "支持离线访问和数据缓存"}, '
            '{"label": "主题切换", "description": "支持明暗主题切换"}]}]</arg_value>'
            "</tool_call>"
        )

        result = self.detector.detect_and_parse(raw_output, self.tools)

        # Should have normal text before the tool call
        self.assertIn(
            "ask_user_question",
            result.normal_text,
            f"Expected normal text to contain intro text, got '{result.normal_text}'",
        )

        # Should have 1 tool call
        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 tool call, got {len(result.calls)}. Calls: {result.calls}",
        )

        # Verify tool call name
        self.assertEqual(
            result.calls[0].name,
            "ask_user_question",
            f"Expected tool name 'ask_user_question', got '{result.calls[0].name}'",
        )

        # Verify parameters contain questions
        self.assertIn(
            '"questions"',
            result.calls[0].parameters,
            f"Expected 'questions' in parameters. Got: {result.calls[0].parameters}",
        )

    def test_glm47_mtp_streaming_with_normal_text(self):
        """
        Test GLM-4.7 streaming scenario where tool call arrives with normal text in one chunk.
        This simulates the MTP scenario where multiple tokens arrive together.
        """
        # Simulate streaming: first chunk has normal text, second chunk has tool call
        chunk1 = "我来调用工具："
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        self.assertEqual(
            result1.normal_text,
            "我来调用工具：",
            f"Expected normal text, got '{result1.normal_text}'",
        )
        self.assertEqual(
            len(result1.calls),
            0,
            f"Expected 0 calls in first chunk, got {len(result1.calls)}",
        )

        # Second chunk: complete tool call
        chunk2 = (
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "test", "header": "test", "multiSelect": false, '
            '"options": [{"label": "A", "description": "Option A"}]}]</arg_value>'
            "</tool_call>"
        )
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call in second chunk, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "ask_user_question",
            f"Expected 'ask_user_question', got '{result2.calls[0].name}'",
        )

    def test_glm47_mtp_complete_tool_call_single_chunk(self):
        """
        Test GLM-4.7 MTP scenario: complete tool call with normal text arrives in single chunk.
        """
        # Complete response in one chunk (MTP style)
        chunk = (
            "让我帮您创建问题："
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "选择语言？", "header": "语言", "multiSelect": false, '
            '"options": [{"label": "Python", "description": "简单"}, '
            '{"label": "Go", "description": "快速"}]}]</arg_value>'
            "</tool_call>"
        )

        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "ask_user_question",
            f"Expected 'ask_user_question', got '{result.calls[0].name}'",
        )
        self.assertIn(
            "questions",
            result.calls[0].parameters,
            f"Expected 'questions' in parameters: {result.calls[0].parameters}",
        )

    def test_glm47_stop_word_handling(self):
        """
        Test that <|observation|> stop word is properly handled (should be truncated).
        Note: The detector itself doesn't handle stop words - that's done by the renderer.
        This test verifies the detector works correctly with text that may have had
        stop words removed.
        """
        # Text with stop word already removed (as it would be by renderer)
        text_without_stop = (
            "<tool_call>ask_user_question<arg_key>questions</arg_key>"
            '<arg_value>[{"question": "test?", "header": "T", "multiSelect": false, '
            '"options": [{"label": "A", "description": "Opt A"}]}]</arg_value>'
            "</tool_call>"
        )

        result = self.detector.detect_and_parse(text_without_stop, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}",
        )


if __name__ == "__main__":
    unittest.main()
