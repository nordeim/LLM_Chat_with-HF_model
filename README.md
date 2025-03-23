# LLM_Chat_with-HF_model

**Install libraries:**
```bash
pip install gradio transformers torch Pillow
```

![image](https://github.com/user-attachments/assets/9fb57a02-d6b0-4c00-a7fb-22cedb34f0e9)

$ python3 hf_chat_copilot-v2.py
* Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Skipping data after last boundary
config.json: 100%|██████████████████████████████████████████████████| 3.90k/3.90k [00:00<00:00, 27.7MB/s]
model.safetensors: 100%|████████████████████████████████████████████| 513M/513M [00:22<00:00, 22.3MB/s]
generation_config.json: 100%|███████████████████████████████████████| 141/141 [00:00<00:00, 924kB/s]
tokenizer_config.json: 100%|████████████████████████████████████████| 27.4k/27.4k [00:00<00:00, 449kB/s]
vocab.json: 100%|███████████████████████████████████████████████████| 801k/801k [00:00<00:00, 1.04MB/s]
merges.txt: 100%|███████████████████████████████████████████████████| 466k/466k [00:00<00:00, 3.52MB/s]
tokenizer.json: 100%|███████████████████████████████████████████████| 3.55M/3.55M [00:00<00:00, 7.85MB/s]
added_tokens.json: 100%|████████████████████████████████████████████| 3.67k/3.67k [00:00<00:00, 24.3MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████| 1.07k/1.07k [00:00<00:00, 8.09MB/s]
Device set to use cpu
The model 'Idefics3ForConditionalGeneration' is not supported for text-generation. Supported models are ['AriaTextForCausalLM', 'BambaForCausalLM', 'BartForCausalLM', 'BertLMHeadModel', 'BertGenerationDecoder', 'BigBirdForCausalLM', 'BigBirdPegasusForCausalLM', 'BioGptForCausalLM', 'BlenderbotForCausalLM', 'BlenderbotSmallForCausalLM', 'BloomForCausalLM', 'CamembertForCausalLM', 'LlamaForCausalLM', 'CodeGenForCausalLM', 'CohereForCausalLM', 'Cohere2ForCausalLM', 'CpmAntForCausalLM', 'CTRLLMHeadModel', 'Data2VecTextForCausalLM', 'DbrxForCausalLM', 'DiffLlamaForCausalLM', 'ElectraForCausalLM', 'Emu3ForCausalLM', 'ErnieForCausalLM', 'FalconForCausalLM', 'FalconMambaForCausalLM', 'FuyuForCausalLM', 'GemmaForCausalLM', 'Gemma2ForCausalLM', 'GitForCausalLM', 'GlmForCausalLM', 'GotOcr2ForConditionalGeneration', 'GPT2LMHeadModel', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM', 'GPTNeoForCausalLM', 'GPTNeoXForCausalLM', 'GPTNeoXJapaneseForCausalLM', 'GPTJForCausalLM', 'GraniteForCausalLM', 'GraniteMoeForCausalLM', 'GraniteMoeSharedForCausalLM', 'HeliumForCausalLM', 'JambaForCausalLM', 'JetMoeForCausalLM', 'LlamaForCausalLM', 'MambaForCausalLM', 'Mamba2ForCausalLM', 'MarianForCausalLM', 'MBartForCausalLM', 'MegaForCausalLM', 'MegatronBertForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM', 'MllamaForCausalLM', 'MoshiForCausalLM', 'MptForCausalLM', 'MusicgenForCausalLM', 'MusicgenMelodyForCausalLM', 'MvpForCausalLM', 'NemotronForCausalLM', 'OlmoForCausalLM', 'Olmo2ForCausalLM', 'OlmoeForCausalLM', 'OpenLlamaForCausalLM', 'OpenAIGPTLMHeadModel', 'OPTForCausalLM', 'PegasusForCausalLM', 'PersimmonForCausalLM', 'PhiForCausalLM', 'Phi3ForCausalLM', 'PhimoeForCausalLM', 'PLBartForCausalLM', 'ProphetNetForCausalLM', 'QDQBertLMHeadModel', 'Qwen2ForCausalLM', 'Qwen2MoeForCausalLM', 'RecurrentGemmaForCausalLM', 'ReformerModelWithLMHead', 'RemBertForCausalLM', 'RobertaForCausalLM', 'RobertaPreLayerNormForCausalLM', 'RoCBertForCausalLM', 'RoFormerForCausalLM', 'RwkvForCausalLM', 'Speech2Text2ForCausalLM', 'StableLmForCausalLM', 'Starcoder2ForCausalLM', 'TransfoXLLMHeadModel', 'TrOCRForCausalLM', 'WhisperForCausalLM', 'XGLMForCausalLM', 'XLMWithLMHeadModel', 'XLMProphetNetForCausalLM', 'XLMRobertaForCausalLM', 'XLMRobertaXLForCausalLM', 'XLNetLMHeadModel', 'XmodForCausalLM', 'ZambaForCausalLM', 'Zamba2ForCausalLM'].
