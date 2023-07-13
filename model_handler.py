from enum import Enum
from typing import Optional
import json
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class ModelHandler:
    class Models(Enum):
        NOUS = "1"
        PYGMALION = "2"

    def __init__(self):
        self._models = {
            ModelHandler.Models.NOUS: {"pipe": None, "tokenizer": None},
            ModelHandler.Models.PYGMALION: {"pipe": None, "tokenizer": None}
        }
        self._load_pipes()

    def _load_pipes(self):
        for model in self._models:
            if model == ModelHandler.Models.NOUS:
                pipe, tokenizer = self._load_nous_model()
            else:
                pipe, tokenizer = self._load_pygmalion_model()
            self._models[model]["pipe"] = pipe
            self._models[model]["tokenizer"] = tokenizer

    def _load_nous_model(self, model_name_or_path: str = "TheBloke/Nous-Hermes-13B-GPTQ", model_basename: str = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"):
        use_triton = False
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        pipe = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=use_triton,
                quantize_config=None)
        return pipe, tokenizer

    def _load_pygmalion_model(self, model_name_or_path="TheBloke/Pygmalion-7B-SuperHOT-8K-GPTQ", model_basename="pygmalion-7b-superhot-8k-GPTQ-4bit-128g.no-act.order"):
        use_triton = False
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        pipe = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device_map='auto',
                use_triton=use_triton,
                quantize_config=None)
        pipe.seqlen = 8192
        return pipe, tokenizer

    def generate(self, model: "ModelHandler.Models", character_name: str, persona: str, prompt: str, chat_history: Optional[str] = None) -> str:
        if model == ModelHandler.Models.NOUS.value:
            return self._generate_nous(character_name, persona, prompt, chat_history)
        elif model == ModelHandler.Models.PYGMALION.value:
            return self._generate_pygmalion(character_name, persona, prompt, chat_history)

    def _generate_nous(self, character_name: str, persona: str, prompt: str, chat_history: Optional[str] = None) -> str:
        if chat_history is not None:
            prompt_template = f'''### Instruction: Continue the conversation as {character_name}. {persona}\n### Input: \n{chat_history}\n### You: {prompt}\n### {character_name}: '''
        else:
            prompt_template = f'''### Instruction: Continue the conversation as {character_name}. {persona}\n### You: {prompt}\n### {character_name}: '''

        tokenizer = self._models[ModelHandler.Models.NOUS]["tokenizer"]
        model = self._models[ModelHandler.Models.NOUS]["pipe"]
        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
        return tokenizer.decode(output[0])

    def _generate_pygmalion(self, character_name: str, persona: str, prompt: str, chat_history: Optional[str] = None) -> str:
        if chat_history is not None:
            prompt_template=f'''{character_name}'s Persona: {persona}
            <START>\n{chat_history}\nYou: {prompt}\n{character_name}: '''
        else:
            prompt_template=f'''{character_name}'s Persona: {persona}
            <START>\nYou: {prompt}\n{character_name}: '''

        tokenizer = self._models[ModelHandler.Models.PYGMALION]["tokenizer"]
        model = self._models[ModelHandler.Models.PYGMALION]["pipe"]
        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512, repetition_penalty = 1.5)
        return tokenizer.decode(output[0])