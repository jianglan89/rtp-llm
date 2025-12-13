import asyncio
import contextlib
import functools
import logging
import time
from typing import AsyncGenerator, Dict, Optional

import grpc
import numpy as np
from grpc import StatusCode
from grpc import aio

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import RoleType
from rtp_llm.ops import EPLBConfig, FfnDisAggregateConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ErrorDetailsPB,
    GenerateInputPB,
    GenerateOutputsPB,
    MultimodalInputPB,
    RoleAddrPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.distribute.worker_info import g_parallel_info, g_worker_info
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateConfig,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)
from rtp_llm.utils.grpc_util import trans_option, trans_option_cast, trans_tensor

MAX_GRPC_TIMEOUT_SECONDS = 3600


class StreamState:
    def __init__(self):
        self.cached_logits_dict = {}


def trans_role_type(role_type: RoleType) -> RoleAddrPB.RoleType:
    if role_type == RoleType.PDFUSION:
        return RoleAddrPB.RoleType.PDFUSION
    elif role_type == RoleType.PREFILL:
        return RoleAddrPB.RoleType.PREFILL
    elif role_type == RoleType.DECODE:
        return RoleAddrPB.RoleType.DECODE
    elif role_type == RoleType.VIT:
        return RoleAddrPB.RoleType.VIT
    elif role_type == RoleType.FRONTEND:
        return RoleAddrPB.RoleType.FRONTEND


def trans_input(input_py: GenerateInput):
    input_pb = GenerateInputPB()
    input_pb.request_id = input_py.request_id
    input_pb.token_ids.extend(input_py.token_ids.reshape(-1).tolist())

    trans_multimodal_input(input_py, input_pb, input_py.generate_config)
    # check generate config is valid before enter into engine
    input_py.generate_config.validate()

    generate_config_pb = input_pb.generate_config
    generate_config_pb.max_new_tokens = input_py.generate_config.max_new_tokens
    generate_config_pb.max_thinking_tokens = (
        input_py.generate_config.max_thinking_tokens
    )
    generate_config_pb.end_think_token_ids.extend(
        input_py.generate_config.end_think_token_ids
    )
    generate_config_pb.in_think_mode = input_py.generate_config.in_think_mode
    generate_config_pb.num_beams = input_py.generate_config.num_beams
    generate_config_pb.variable_num_beams.extend(
        input_py.generate_config.variable_num_beams
    )
    generate_config_pb.num_return_sequences = (
        input_py.generate_config.num_return_sequences
    )
    generate_config_pb.min_new_tokens = input_py.generate_config.min_new_tokens
    generate_config_pb.top_k = input_py.generate_config.top_k
    generate_config_pb.top_p = input_py.generate_config.top_p
    generate_config_pb.temperature = input_py.generate_config.temperature
    generate_config_pb.sp_edit = input_py.generate_config.sp_edit
    generate_config_pb.force_disable_sp_run = (
        input_py.generate_config.force_disable_sp_run
    )
    generate_config_pb.force_sp_accept = input_py.generate_config.force_sp_accept
    generate_config_pb.repetition_penalty = input_py.generate_config.repetition_penalty
    generate_config_pb.presence_penalty = input_py.generate_config.presence_penalty
    generate_config_pb.frequency_penalty = input_py.generate_config.frequency_penalty
    generate_config_pb.do_sample = input_py.generate_config.do_sample
    trans_option(generate_config_pb, input_py.generate_config, "no_repeat_ngram_size")
    trans_option(generate_config_pb, input_py.generate_config, "random_seed")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_decay")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_min")
    trans_option(generate_config_pb, input_py.generate_config, "top_p_reset_ids")
    trans_option(generate_config_pb, input_py.generate_config, "adapter_name")
    trans_option_cast(
        generate_config_pb, input_py.generate_config, "task_id", functools.partial(str)
    )

    generate_config_pb.select_tokens_id.extend(
        input_py.generate_config.select_tokens_id
    )
    generate_config_pb.calculate_loss = input_py.generate_config.calculate_loss
    generate_config_pb.return_logits = input_py.generate_config.return_logits
    generate_config_pb.return_incremental = input_py.generate_config.return_incremental
    generate_config_pb.return_hidden_states = (
        input_py.generate_config.return_hidden_states
    )
    generate_config_pb.return_all_hidden_states = (
        input_py.generate_config.return_all_hidden_states
    )
    generate_config_pb.hidden_states_cut_dim = (
        input_py.generate_config.hidden_states_cut_dim
    )
    generate_config_pb.normalized_hidden_states = (
        input_py.generate_config.normalized_hidden_states
    )
    generate_config_pb.is_streaming = input_py.generate_config.is_streaming
    generate_config_pb.timeout_ms = input_py.generate_config.timeout_ms
    if input_py.generate_config.sp_advice_prompt_token_ids:
        generate_config_pb.sp_advice_prompt_token_ids.extend(
            input_py.generate_config.sp_advice_prompt_token_ids
        )
    generate_config_pb.return_cum_log_probs = (
        input_py.generate_config.return_cum_log_probs
    )
    generate_config_pb.return_all_probs = input_py.generate_config.return_all_probs
    generate_config_pb.return_softmax_probs = (
        input_py.generate_config.return_softmax_probs
    )
    generate_config_pb.can_use_pd_separation = (
        input_py.generate_config.can_use_pd_separation
    )
    generate_config_pb.gen_timeline = input_py.generate_config.gen_timeline
    generate_config_pb.profile_step = input_py.generate_config.profile_step
    generate_config_pb.global_request_id = input_py.generate_config.global_request_id
    generate_config_pb.inter_request_id = input_py.generate_config.inter_request_id
    generate_config_pb.ignore_eos = input_py.generate_config.ignore_eos
    generate_config_pb.reuse_cache = input_py.generate_config.reuse_cache
    generate_config_pb.enable_3fs = input_py.generate_config.enable_3fs
    generate_config_pb.enable_memory_block_cache = (
        input_py.generate_config.enable_memory_block_cache
    )

    trans_option_cast(
        generate_config_pb, input_py.generate_config, "trace_id", functools.partial(str)
    )

    for i in range(len(input_py.generate_config.stop_words_list)):
        stop_words = generate_config_pb.stop_words_list.rows.add()
        stop_words.values.extend(input_py.generate_config.stop_words_list[i])

    for role_addr in input_py.generate_config.role_addrs:
        role_addr_pb = RoleAddrPB()
        role_addr_pb.role = trans_role_type(role_addr.role)
        role_addr_pb.ip = role_addr.ip
        role_addr_pb.http_port = role_addr.http_port
        role_addr_pb.grpc_port = role_addr.grpc_port

        generate_config_pb.role_addrs.append(role_addr_pb)

    return input_pb


def trans_multimodal_input(
    input_py: GenerateInput, input_pb: GenerateInputPB, generate_config: GenerateConfig
):
    resized_shape = [-1, -1]
    if generate_config.resized_shape:
        if len(generate_config.resized_shape) != 2:
            logging.info(
                "Resized shape must be a list with 2 positive int, refering width and height"
            )
        else:
            resized_shape = generate_config.resized_shape
    for mm_input in input_py.mm_inputs:
        mm_input_pb = MultimodalInputPB()
        mm_input_pb.multimodal_url = mm_input.url
        mm_input_pb.multimodal_type = mm_input.mm_type
        mm_preprocess_config_pb = mm_input_pb.mm_preprocess_config
        mm_preprocess_config_pb.width = (
            mm_input.config.width if mm_input.config.width != -1 else resized_shape[0]
        )
        mm_preprocess_config_pb.height = (
            mm_input.config.height if mm_input.config.height != -1 else resized_shape[1]
        )
        mm_preprocess_config_pb.min_pixels = mm_input.config.min_pixels
        mm_preprocess_config_pb.max_pixels = mm_input.config.max_pixels
        mm_preprocess_config_pb.fps = mm_input.config.fps
        mm_preprocess_config_pb.min_frames = mm_input.config.min_frames
        mm_preprocess_config_pb.max_frames = mm_input.config.max_frames
        input_pb.multimodal_inputs.append(mm_input_pb)


# 假设 trans_tensor 函数将 Protobuf 的 TensorPB 转换为 numpy array
# from .utils import trans_tensor


def trans_output(
    input_py: GenerateInput, outputs_pb: GenerateOutputsPB, stream_state: StreamState
) -> GenerateOutputs:
    logging.debug("outputs_pb = %s", outputs_pb)
    output_pb = outputs_pb.flatten_output
    num_outputs = len(output_pb.finished)

    if num_outputs == 0:
        return GenerateOutputs()

    logits_index = input_py.generate_config.logits_index
    aux_info_flag = input_py.generate_config.aux_info

    all_output_ids = (
        trans_tensor(output_pb.output_ids)
        if output_pb.HasField("output_ids")
        and (len(output_pb.output_ids.shape) > 0 and output_pb.output_ids.shape[0] > 0)
        else None
    )
    all_hidden_states = (
        trans_tensor(output_pb.hidden_states)
        if output_pb.HasField("hidden_states")
        and len(output_pb.hidden_states.shape) > 0
        and output_pb.hidden_states.shape[0] > 0
        else None
    )
    all_all_hidden_states = (
        trans_tensor(output_pb.all_hidden_states)
        if output_pb.HasField("all_hidden_states")
        and len(output_pb.all_hidden_states.shape) > 0
        and output_pb.all_hidden_states.shape[0] > 0
        else None
    )
    all_loss = (
        trans_tensor(output_pb.loss)
        if output_pb.HasField("loss")
        and len(output_pb.loss.shape) > 0
        and output_pb.loss.shape[0] > 0
        else None
    )
    all_logits = (
        trans_tensor(output_pb.logits)
        if output_pb.HasField("logits")
        and len(output_pb.logits.shape) > 0
        and output_pb.logits.shape[0] > 0
        else None
    )
    all_all_probs = (
        trans_tensor(output_pb.all_probs)
        if output_pb.HasField("all_probs")
        and len(output_pb.all_probs.shape) > 0
        and output_pb.all_probs.shape[0] > 0
        else None
    )

    outputs_py = GenerateOutputs()
    input_token_ids = input_py.token_ids.reshape(1, -1)

    # 遍历每个 beam/output
    for i in range(num_outputs):
        output_py = GenerateOutput()
        output_py.finished = output_pb.finished[i]
        current_aux_info = None
        if aux_info_flag and len(output_pb.aux_info) > i:
            aux_info_pb = output_pb.aux_info[i]
            current_aux_info = AuxInfo(
                cost_time=aux_info_pb.cost_time_us / 1000.0,
                first_token_cost_time=aux_info_pb.first_token_cost_time_us / 1000.0,
                wait_time=aux_info_pb.wait_time_us / 1000.0,
                iter_count=aux_info_pb.iter_count,
                input_len=aux_info_pb.input_len,
                prefix_len=aux_info_pb.prefix_len,
                output_len=aux_info_pb.output_len,
                step_output_len=aux_info_pb.step_output_len,
                fallback_tokens=aux_info_pb.fallback_tokens,
                fallback_times=aux_info_pb.fallback_times,
                pd_sep=aux_info_pb.pd_sep,
                reuse_len=aux_info_pb.total_reuse_len,
                local_reuse_len=aux_info_pb.local_reuse_len,
                remote_reuse_len=aux_info_pb.remote_reuse_len,
                prefill_total_reuse_len=aux_info_pb.prefill_total_reuse_len,
                prefill_local_reuse_len=aux_info_pb.prefill_local_reuse_len,
                prefill_remote_reuse_len=aux_info_pb.prefill_remote_reuse_len,
                decode_total_reuse_len=aux_info_pb.decode_total_reuse_len,
                decode_local_reuse_len=aux_info_pb.decode_local_reuse_len,
                decode_remote_reuse_len=aux_info_pb.decode_remote_reuse_len,
                aux_string=aux_info_pb.aux_string,
                role_addrs=input_py.generate_config.role_addrs,
            )
            if aux_info_pb.HasField("cum_log_probs"):
                current_aux_info.cum_log_probs = trans_tensor(
                    aux_info_pb.cum_log_probs
                ).tolist()
            if aux_info_pb.HasField("softmax_probs"):
                current_aux_info.softmax_probs = trans_tensor(
                    aux_info_pb.softmax_probs
                ).tolist()

            output_py.aux_info = current_aux_info

        if all_output_ids is not None:
            output_py.output_ids = all_output_ids[i]
        output_py.input_ids = input_token_ids

        if all_hidden_states is not None:
            output_py.hidden_states = all_hidden_states[i]

        if all_all_hidden_states is not None:
            output_py.all_hidden_states = all_all_hidden_states[i]

        if all_loss is not None:
            loss_slice = all_loss[i]
            if input_py.generate_config.calculate_loss == 1:
                output_py.loss = (
                    loss_slice[0]
                    if hasattr(loss_slice, "__len__") and len(loss_slice) > 0
                    else loss_slice
                )
            else:
                output_py.loss = loss_slice

        if all_logits is not None:
            output_py.logits = all_logits[i]

        if all_all_probs is not None:
            output_py.all_probs = all_all_probs[i]

        if (
            logits_index is not None
            and all_logits is not None
            and current_aux_info
            and current_aux_info.output_len == logits_index
        ):
            stream_state.cached_logits_dict[i] = output_py.logits

        if output_py.finished and i in stream_state.cached_logits_dict:
            output_py.logits = stream_state.cached_logits_dict[i]

        outputs_py.generate_outputs.append(output_py)

    return outputs_py


class HostChannel:
    __slots__ = ("channel", "last_used")
    def __init__(self, channel: aio.Channel):
        self.channel = channel
        self.last_used = time.time()

class HostChannelPool:
    def __init__(self, options=None, idle_timeout=300, cleanup_interval=60):
        """
        idle_timeout: seconds to keep an unused channel before closing it
        cleanup_interval: how often to scan for idle channels
        """
        self.options = options or []
        self.idle_timeout = idle_timeout
        self.cleanup_interval = cleanup_interval

        self._channels: Dict[str, HostChannel] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stopped = False

        # Start cleanup task only if there's a running event loop
        try:
            # Check if there's a running event loop
            asyncio.get_running_loop()
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No running event loop, defer task creation until later
            pass

    def __del__(self):
        """Clean up resources when the pool is garbage collected"""
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if not self._stopped:
                # Schedule the cleanup in the existing loop
                asyncio.ensure_future(self.stop())
        except RuntimeError:
            # No running event loop, try to clean up synchronously
            # This is a best-effort cleanup since we can't await in __del__
            if not self._stopped:
                self._stopped = True
                if self._cleanup_task:
                    self._cleanup_task.cancel()
                # Can't close channels properly without event loop
                # But at least mark as stopped and clear references
                self._channels.clear()
        except Exception as e:
            # Log but don't raise - __del__ should not throw
            logging.warning(f"Failed to cleanup HostChannelPool in __del__: {e}")

    async def start(self):
        # Deprecated, kept for compatibility
        pass

    async def stop(self):
        self._stopped = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
        # close and drop everything
        await self.close_all()

    # ---------- main API ----------

    async def get(self, target: str) -> aio.Channel:
        """
        Get or create a channel for `target`.
        """
        # Start cleanup task if it hasn't been started yet
        if self._cleanup_task is None and not self._stopped:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
                logging.info(f"Channel cleanup task started (idle_timeout={self.idle_timeout}s, cleanup_interval={self.cleanup_interval}s)")
            except RuntimeError:
                # Still no event loop, which is unusual in an async context
                logging.warning("Failed to start channel cleanup task: no running event loop")

        async with self._lock:
            entry = self._channels.get(target)
            if entry is None:
                ch = aio.insecure_channel(target, options=self.options)
                entry = HostChannel(ch)
                self._channels[target] = entry
            entry.last_used = time.time()
            return entry.channel

    async def recreate(self, target: str) -> aio.Channel:
        """
        Force-close and recreate a channel for `target`.
        Useful after UNAVAILABLE/INTERNAL errors.
        """
        async with self._lock:
            entry = self._channels.pop(target, None)  # remove from map
        if entry is not None:
            # close old channel, ignore errors but ensure it can be GC'd
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(entry.channel.close(), timeout=2.0)

        # create new
        ch = aio.insecure_channel(target, options=self.options)
        async with self._lock:
            self._channels[target] = HostChannel(ch)
        return ch

    async def delete(self, target: str):
        """
        Explicitly remove one host from pool and release all its resources.
        """
        async with self._lock:
            entry = self._channels.pop(target, None)
        if entry is not None:
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(entry.channel.close(), timeout=2.0)
        # no reference left -> GC can collect it

    async def close_all(self):
        """
        Close and drop all channels from the pool.
        """
        async with self._lock:
            entries = list(self._channels.values())
            self._channels.clear()

        # close outside the lock
        tasks = [asyncio.wait_for(e.channel.close(), timeout=2.0) for e in entries]
        with contextlib.suppress(Exception):
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---------- background cleanup ----------

    async def _cleanup_loop(self):
        logging.info(f"Channel cleanup loop started, will run every {self.cleanup_interval}s")
        try:
            while not self._stopped:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle()
        except asyncio.CancelledError:
            logging.info("Channel cleanup loop cancelled")
        finally:
            logging.info("Channel cleanup loop stopped")

    async def _cleanup_idle(self):
        """
        Find idle hosts, remove them from the pool, and close their channels.
        """
        now = time.time()
        to_close: list[tuple[str, HostChannel]] = []  # Keep target for logging

        async with self._lock:
            total_channels = len(self._channels)
            for target, entry in list(self._channels.items()):
                idle_time = now - entry.last_used
                if idle_time > self.idle_timeout:
                    logging.info(f"Channel {target} has been idle for {idle_time:.1f}s (timeout: {self.idle_timeout}s), marking for cleanup")
                    to_close.append((target, entry))
                    del self._channels[target]  # remove reference

            remaining_channels = len(self._channels)
            if to_close:
                logging.info(f"Channel cleanup: closing {len(to_close)} idle channels, {remaining_channels} channels remaining (was {total_channels})")
            elif total_channels > 0:
                logging.debug(f"Channel cleanup: no idle channels found, {total_channels} active channels")

        # Close outside lock
        closed_count = 0
        failed_count = 0
        for target, entry in to_close:
            try:
                await asyncio.wait_for(entry.channel.close(), timeout=2.0)
                closed_count += 1
                logging.debug(f"Successfully closed channel for {target}")
            except asyncio.TimeoutError:
                failed_count += 1
                logging.warning(f"Timeout while closing channel for {target}")
            except Exception as e:
                failed_count += 1
                logging.warning(f"Error closing channel for {target}: {e}")

        if to_close:
            logging.info(f"Channel cleanup completed: {closed_count} channels closed successfully, {failed_count} failed")
                
class ModelRpcClient(object):
    def __init__(
        self,
        addresses: list[str],
        client_config,
        max_rpc_timeout_ms: int = 0,
        decode_entrance: bool = False,
    ):
        """Initialize ModelRpcClient with addresses.
        
        Args:
            addresses: List of RPC addresses for data parallel communication
            max_rpc_timeout_ms: Maximum RPC timeout in milliseconds
            decode_entrance: Whether this is a decode entrance
        """
        self._addresses = addresses
        self._max_rpc_timeout_ms = max_rpc_timeout_ms
        self._decode_entrance = decode_entrance
        self.options = []
        for key, value in client_config.items():
            self.options.append((key, value))
        logging.info(f"client options: {self.options}")

        # Initialize the channel pool
        options = [
            ("grpc.max_metadata_size", 1024 * 1024 * 1024),
        ]
        self._channel_pool = HostChannelPool(
            options=options,
            idle_timeout=300,  # 5 minutes
            cleanup_interval=60  # clean up every minute
        )

    async def close(self):
        """Clean up resources when shutting down the client"""
        await self._channel_pool.stop()

    async def enqueue(
        self, input_py: GenerateInput
    ) -> AsyncGenerator[GenerateOutputs, None]:
        request_timeout_ms = input_py.generate_config.timeout_ms
        rpc_timeout_ms = (
            self._max_rpc_timeout_ms
            if self._max_rpc_timeout_ms > 0
            else MAX_GRPC_TIMEOUT_SECONDS * 1000
        )
        if request_timeout_ms == None or request_timeout_ms <= 0:
            grpc_timeout_seconds = rpc_timeout_ms / 1000
        else:
            grpc_timeout_seconds = request_timeout_ms / 1000
        input_py.generate_config.timeout_ms = (int)(grpc_timeout_seconds * 1000)
        input_pb = trans_input(input_py)
        response_iterator = None
        stream_state = StreamState()

        address_list = self._addresses

        for role_addr in input_py.generate_config.role_addrs:
            if (
                (
                    self._decode_entrance
                    and role_addr.role == RoleType.DECODE
                )
                or role_addr.role == RoleType.PDFUSION
                or (
                    not self._decode_entrance
                    and role_addr.role == RoleType.PREFILL
                )
            ):
                if role_addr.ip != "":
                    address_list = [role_addr.ip + ":" + str(role_addr.grpc_port)]
                    break
        
        if not address_list:
            raise ValueError(f"No address found for request: {input_pb.request_id}")

        try:
            # Select target address
            target_address = address_list[input_py.request_id % len(address_list)]

            # Get channel from pool
            channel = await self._channel_pool.get(target_address)
            stub = RpcServiceStub(channel)

            response_iterator = stub.GenerateStreamCall(
                input_pb, timeout=grpc_timeout_seconds
            )
            # 调用服务器方法并接收流式响应
            count = 0
            async for response in response_iterator.__aiter__():
                count += 1
                yield trans_output(input_py, response, stream_state)
        except grpc.RpcError as e:
            # TODO(xinfei.sxf) 非流式的请求无法取消了
            if response_iterator:
                response_iterator.cancel()
            error_details = ErrorDetailsPB()
            metadata = e.trailing_metadata()
            if "grpc-status-details-bin" in metadata and error_details.ParseFromString(
                metadata["grpc-status-details-bin"]
            ):
                logging.error(
                    f"request: [{input_pb.request_id}] RPC failed: "
                    f"{e.code()}, {e.details()}, detail error code is "
                    f"{ExceptionType.from_value(error_details.error_code)}"
                )
                raise FtRuntimeException(
                    ExceptionType(error_details.error_code), error_details.error_message
                )
            else:
                logging.error(
                    f"request: [{input_pb.request_id}] RPC failed: "
                    f"error code is {e.code()}, detail is {e.details()}"
                )
                if e.code() == StatusCode.DEADLINE_EXCEEDED:
                    raise FtRuntimeException(
                        ExceptionType.GENERATE_TIMEOUT, e.details()
                    )
                elif e.code() == StatusCode.CANCELLED:
                    raise FtRuntimeException(ExceptionType.CANCELLED_ERROR, e.details())
                else:
                    raise FtRuntimeException(ExceptionType.UNKNOWN_ERROR, e.details())
        except Exception as e:
            logging.error(f"rpc unknown error:{str(e)}")
            raise e
        finally:
            if response_iterator:
                response_iterator.cancel()
