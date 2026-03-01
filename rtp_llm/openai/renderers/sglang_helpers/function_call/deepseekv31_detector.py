import json
import logging
import re
from typing import List, Optional

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import Tool
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.ebnf_composer import (
    EBNFComposer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.utils import (
    _is_complete_json,
)

logger = logging.getLogger(__name__)


class DeepSeekV31Detector(BaseFormatDetector):
    """
    Detector for DeepSeek V3 model function call format.
    The DeepSeek V3 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.
    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Examples:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `<｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>`
    - Arguments: JSON code block between `<｜tool▁sep｜>` and `<｜tool▁call▁end｜>`
    - Supports multiple tool calls
    Reference: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"
        self.eot_token = "<｜tool▁calls▁end｜>"
        self.tool_call_start = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = (
            r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>"
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        # Markers for _find_tool_boundary (prefix ends before earliest)
        self._tool_boundary_markers = (self.bot_token, self.tool_call_start)

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.
        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
                func_args = json.loads(func_args)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Streaming incremental parsing tool calls for DeepSeekV3 format.

        MTP-safe: First checks for complete <｜tool▁call▁begin｜>...<｜tool▁call▁end｜> blocks.
        Falls back to regex-based incremental parsing for partial data.

        NOTE: This MTP-safe path is a workaround, not a true incremental parser.
        The root cause is that the base class assumes bot_token at buffer start (for not MTP, yes).
        When prefix content exists (e.g., "</think>\n\n<tool_call>..."), the base
        class fails with MalformedJSON because it tries to parse from index 0.

        Current behavior: Buffer accumulates failed chunks until both bot_token
        and eot_token are present, then parses everything at once (batch parsing).

        A cleaner fix would be to modify base class to:
        1. Find bot_token anywhere in buffer (not just at start)
        2. Emit text before bot_token as normal_text immediately
        3. Continue true incremental parsing from bot_token onward

        This workaround was added to avoid modifying base class behavior that
        affects all detectors. Future refactoring should consider the base class fix.
        Currently, this workaround only applies to Qwen25Detector, DeepSeekV31Detector and KimiK2Detector.
        """
        self._buffer += new_text
        current_text = self._buffer

        # ---- Separate plain text prefix from tool call data ----
        # Find the earliest tool-related marker in buffer.
        # Any text before it is plain content that must be emitted immediately,
        # not trapped in the buffer. This fixes the bug where trailing chars
        # (e.g. ">" from "</thinking>") get swallowed when they arrive in the
        # same MTP chunk as a tool marker.
        tool_boundary = self._find_tool_boundary(current_text)

        if tool_boundary is None:
            # No marker yet: emit full buffer as normal text (don't return only new_text
            # or incremental streaming would lose accumulated content)
            self._buffer = ""
            for e_token in [self.eot_token, self.tool_call_end]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        prefix_text = ""
        if tool_boundary > 0:
            prefix_text = current_text[:tool_boundary]
            current_text = current_text[tool_boundary:]
            self._buffer = current_text

        # ---- MTP-safe path: Parse any complete tool call blocks ----
        # This handles MTP scenarios where multiple tokens arrive in single chunk
        collected_calls: list[ToolCallItem] = []

        while (
            self.tool_call_start in current_text and self.tool_call_end in current_text
        ):
            start_idx = current_text.find(self.tool_call_start)
            # Search for end marker AFTER start to avoid matching
            # a residual end marker that precedes the current begin
            end_idx = current_text.find(
                self.tool_call_end, start_idx + len(self.tool_call_start)
            )

            # No valid complete block
            if end_idx == -1:
                break

            # Extract the complete tool call block
            block_end = end_idx + len(self.tool_call_end)
            complete_block = current_text[start_idx:block_end]

            # Try to parse with the full regex
            match = re.search(self.func_detail_regex, complete_block, re.DOTALL)
            if match:
                func_name = match.group(1).strip()
                func_args_raw = match.group(2).strip()

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                # Create complete tool call item
                collected_calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=func_name,
                        parameters=func_args_raw,
                    )
                )

                # Store tool call info for serving layer
                try:
                    parsed_args = json.loads(func_args_raw) if func_args_raw else {}
                except json.JSONDecodeError:
                    parsed_args = {}
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": parsed_args,
                }
                self.streamed_args_for_tool[self.current_tool_id] = func_args_raw or ""
                self.current_tool_id += 1

                # Extend arrays for next potential tool call
                self.prev_tool_call_arr.append({})
                self.streamed_args_for_tool.append("")

            # Remove processed block from buffer
            current_text = current_text[block_end:]
            self._buffer = current_text

        # If we parsed any complete blocks, return those results
        if collected_calls:
            return StreamingParseResult(normal_text=prefix_text, calls=collected_calls)

        # No complete blocks but have prefix → emit prefix first
        if prefix_text:
            return StreamingParseResult(normal_text=prefix_text)

        # ---- Incremental parsing for incomplete tool call ----
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(1).strip()
                func_args_raw = partial_match.group(2).strip()

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    # Store the tool call info for serving layer completions endpoint
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[
                            self.current_tool_id
                        ] += argument_diff

                    if _is_complete_json(func_args_raw):
                        # Update the stored arguments
                        try:
                            parsed_args = json.loads(func_args_raw)
                            self.prev_tool_call_arr[self.current_tool_id][
                                "arguments"
                            ] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        tool_call_end_pattern = (
                            r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
                        )
                        match = re.search(
                            tool_call_end_pattern, current_text, re.DOTALL
                        )
                        if match:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[match.end() :]
                        else:
                            self._buffer = ""

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def _find_tool_boundary(self, text: str) -> Optional[int]:
        """Find position of the earliest tool-related marker in text.

        Returns None if no marker is found.
        """
        earliest = None
        for marker in self._tool_boundary_markers:
            idx = text.find(marker)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx
        return earliest

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
            end="<｜tool▁call▁end｜>",
            trigger="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
        )

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            sequence_start_token=self.bot_token,
            sequence_end_token=self.eot_token,
            tool_call_separator="",
            call_rule_fmt='"<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{arguments_rule}<｜tool▁call▁end｜>"',
            function_format="json",
        )
