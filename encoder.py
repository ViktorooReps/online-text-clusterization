import logging
import pickle
import time
import traceback
from copy import deepcopy
from os import cpu_count, environ
from pathlib import Path
from typing import List, TypeVar, Type, Optional, Dict

import torch
from torch import Tensor, LongTensor
from torch.nn import Module, Parameter
from torch.nn.functional import normalize
from torch.onnx import export
from transformers import AutoTokenizer, AutoModel, BatchEncoding, PreTrainedModel, PreTrainedTokenizer, TensorType
from transformers.convert_graph_to_onnx import quantize
from transformers.onnx import FeaturesManager, OnnxConfig
from transformers.utils import to_numpy

torch.set_num_threads(cpu_count() // 2)

# Constants from the performance optimization available in onnxruntime
# It needs to be done before importing onnxruntime
environ["OMP_NUM_THREADS"] = str(cpu_count() // 2)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

logger = logging.getLogger(__name__)

try:
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions
except:
    logger.warning(f'Could not import ONNX inference tools with exception {traceback.format_exc()}!')


_ModelType = TypeVar('_ModelType', bound='SerializableModel')


class SerializableModel(Module):

    def __init__(self):
        super().__init__()
        self._dummy_param = Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self._dummy_param.device

    def save(self, save_path: Path) -> None:
        previous_device = self.device
        self.cpu()
        with open(save_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        self.to(previous_device)

    @classmethod
    def load(cls: Type[_ModelType], load_path: Path) -> _ModelType:
        with open(load_path, 'rb') as f:
            return pickle.load(f)


class Encoder(SerializableModel):

    def __init__(self, bert_model: str):
        super(Encoder, self).__init__()
        self._bert_encoder: PreTrainedModel = AutoModel.from_pretrained(bert_model)
        self._bert_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.eval()

    @property
    def hidden_size(self) -> int:
        return self._bert_encoder.config.hidden_size

    def prepare_inputs(self, texts: List[str]) -> BatchEncoding:
        return self._bert_tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512,
            add_special_tokens=True
        )

    @torch.no_grad()
    def forward(self, input_ids: LongTensor) -> Tensor:
        return normalize(self._bert_encoder(input_ids)['last_hidden_state'][:, 0])  # select [CLS] token representation

    def optimize(
            self,
            onnx_dir: Path,
            quant: bool = True,
            fuse: bool = True,
            opset_version: int = 13,
            do_constant_folding: bool = True
    ) -> None:
        onnx_model_path = onnx_dir.joinpath('model.onnx')
        onnx_optimized_model_path = onnx_dir.joinpath('model-optimized.onnx')

        # load config
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self._bert_encoder)
        onnx_config: OnnxConfig = model_onnx_config(self._bert_encoder.config)

        model_inputs = onnx_config.generate_dummy_inputs(self._bert_tokenizer, framework=TensorType.PYTORCH)
        dynamic_axes = {0: 'batch', 1: 'sequence'}
        # export to onnx
        export(
            self._bert_encoder,
            ({'input_ids': model_inputs['input_ids']},),
            f=onnx_model_path.as_posix(),
            verbose=False,
            input_names=('input_ids',),
            output_names=('last_hidden_state',),
            dynamic_axes={'input_ids': dynamic_axes, 'last_hidden_state': dynamic_axes},
            do_constant_folding=do_constant_folding,
            opset_version=opset_version,
        )

        if fuse:
            opt_options = FusionOptions('bert')
            opt_options.enable_embed_layer_norm = False

            optimizer.optimize_model(
                str(onnx_model_path),
                'bert',
                num_heads=12,
                hidden_size=768,
                optimization_options=opt_options
            ).save_model_to_file(str(onnx_optimized_model_path))

            onnx_model_path = onnx_optimized_model_path

        if quant:
            onnx_model_path = quantize(onnx_model_path)

        self._bert_encoder = ONNXOptimizedEncoder(onnx_model_path)


class ONNXOptimizedEncoder(Module):

    def __init__(self, onnx_path: Path):
        super().__init__()
        self._onnx_path = onnx_path
        self._session: Optional[InferenceSession] = None

    def __getstate__(self):
        state = deepcopy(self.__dict__)
        state.pop('_session')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._session = None

    def _start_session(self) -> None:
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        self._session = InferenceSession(self._onnx_path.as_posix(), options, providers=['CPUExecutionProvider'])
        self._session.disable_fallback()

    def forward(self, input_ids: LongTensor, **_) -> Dict[str, Tensor]:
        if self._session is None:
            logger.info(f'Starting inference session for {self._onnx_path}.')
            start_time = time.time()
            self._start_session()
            logger.info(f'Inference started in {time.time() - start_time:.4f}s.')

        # Run the model (None = get all the outputs)
        return {
            'last_hidden_state': torch.tensor(self._session.run(
                None,
                {
                    'input_ids': to_numpy(input_ids)
                }
            )[0])
        }
