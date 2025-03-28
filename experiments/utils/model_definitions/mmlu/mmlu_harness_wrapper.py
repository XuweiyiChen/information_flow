from typing import Optional, Union

import torch
from tuned_lens import TunedLens
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM
"""
ADAPTED FROM https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/mamba_lm.py
"""

@register_model("pythia_lens")
class PythiaLens(HFLM):
    VALID_SIZES = ['70m', '160m', '410m', '1b', '1.4b']
    def __init__(
        self,
        model_size='410m',
        evaluation_layer=-1
    ) -> None:
        assert model_size in self.VALID_SIZES

        model_path=f"EleutherAI/pythia-{model_size}-deduped-v0"
        self.is_hf = True
        self.evaluation_layer = evaluation_layer

        super().__init__(
            pretrained=model_path,
            tokenizer=model_path,
            max_length=2048,
        )

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        **kwargs,
    ) -> None:

        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map='cuda:0',
            revision='main'
        )

        print(self.evaluation_layer, self.config.num_hidden_layers)
        self.lens = TunedLens.from_model_and_pretrained(self._model)
        self.lens.to(self._model.device)

        assert self.evaluation_layer <= self.config.num_hidden_layers, \
            f"Evaluation layer={self.evaluation_layer} cannot be larger than the number of layers={self.config.num_hidden_layers}! "

    def _model_call(self, inps, attn_mask=None, labels=None):
        # returns the logits
        assert self.AUTO_MODEL_CLASS == AutoModelForCausalLM

        outputs = self._model(inps, output_hidden_states=True)

        hs = list(outputs.hidden_states)
        logits = self.lens.forward(hs[self.evaluation_layer], self.evaluation_layer )
        return logits
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        return self.lens.generate(
            model=self._model,
            layer=self.evaluation_layer,
            input_ids=context,
            max_new_tokens=max_length,
            **generation_kwargs,
        )
