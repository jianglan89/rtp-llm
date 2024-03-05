import os
import asyncio
from unittest import TestCase, main
from fastapi import Request as RawRequest

from maga_transformer.server.inference_server import InferenceServer
from maga_transformer.utils.complete_response_async_generator import CompleteResponseAsyncGenerator

from typing import Any
from pydantic import BaseModel, Field

class FakePipelinResponse(BaseModel):
    res: str

class FakeInferenceWorker(object):
    def inference(self, prompt: str, *args: Any, **kwargs: Any):

        response_generator = self._inference(prompt, *args, **kwargs)
        return CompleteResponseAsyncGenerator(response_generator, CompleteResponseAsyncGenerator.get_last_value)

    async def _inference(self, prompt: str, *args: Any, **kwargs: Any):
        yield FakePipelinResponse(res=prompt)

    def is_streaming(self, *args: Any, **kawrgs: Any):
        return False

class FakeRawRequest(object):
    async def is_disconnected(self):
        return False

class InferenceServerTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        os.environ
        self.inference_server = InferenceServer()
        self.inference_server._inference_worker = FakeInferenceWorker()

    async def _async_run(self, *args: Any, **kwargs: Any):
        res = await self.inference_server.inference(*args, **kwargs)
        return res

    def test_simple(self):
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(self._async_run(req={"prompt": "hello"}, raw_request=FakeRawRequest()))
        self.assertEqual(res.body.decode('utf-8'), '{"res":"hello"}', res.body.decode('utf-8'))
        res = loop.run_until_complete(self._async_run(req='{"prompt": "hello"}',raw_request=FakeRawRequest()))
        self.assertEqual(res.body.decode('utf-8'), '{"res":"hello"}', res.body.decode('utf-8'))

main()