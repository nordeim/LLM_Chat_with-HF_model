The short answer is that the error occurs because the pipeline you’re using isn’t constructed correctly for the EXAONE model. That model requires special handling (for example, using `AutoModelForCausalLM` with the correct configuration), and by simply passing the model’s name to the generic text‐generation pipeline (with `trust_remote_code=True`), you end up with an object whose callable (generation) method is either missing or set to `None`. When your code later does

```python
output = gen_pipeline(conversation)[0]["generated_text"]
```

it’s essentially trying to call something that isn’t callable, hence the error:  
`'NoneType' object is not callable`.

**Detailed Explanation:**

1. **Model-specific Requirements:**  
   The snippet you shared at the start shows that for models with “EXAONE” in the name, you would normally select `AutoModelForCausalLM` instead of the generic `AutoModel`. This extra step is needed because certain models load a custom forward/generation method via remote code. Unfortunately, the pipeline instantiation in your cached function (`get_model_pipeline`) doesn’t make this distinction. It always calls:
   ```python
   pipeline("text-generation", model=model_id, max_new_tokens=150, trust_remote_code=True)
   ```
   When `model_id` is something like `"LGAI-EXAONE/EXAONE-Deep-2.4B"`, the custom generation logic (that might correctly bind the callable) isn’t applied, so the resulting pipeline object ends up in a broken state—its intended generation function is either missing or overwritten by `None`.

2. **Caching Complications:**  
   The use of `@lru_cache` means that once the pipeline is instantiated (even in a misconfigured state), it’s reused. So if the model wasn’t loaded with the proper configuration the first time, every subsequent call will try to use that cached (and “non-callable”) pipeline.

3. **Randomness in Responses:**  
   You mentioned that for some prompts you see random responses. When models are loaded via `AutoModelForCausalLM` directly, they often apply custom decoding or prompt handling logic. The generic pipeline may not be invoking these routines properly—which leads to outputs that aren’t tied to the prompt (or even might be “garbage”) because the model isn’t receiving and processing the prompt as expected.

### What to Do Next

- **Use the Correct Model Instantiation:**  
  In your pipeline-loading function, check if the model name contains `"EXAONE"` (like in your original snippet) and then explicitly load the model using `AutoModelForCausalLM` plus the corresponding tokenizer. Then pass the actual model and tokenizer objects into the pipeline call, for example:
  
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
  
  def get_custom_pipeline(model_id: str):
      if "EXAONE" in model_id:
          model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
      else:
          # Fallback option if needed.
          model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
      tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
      return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
  ```
  
- **Adjust Your Caching Strategy:**  
  Ensure that the pipeline is created with these model-specific adjustments before caching it. That way you won’t accidentally cache a pipeline where the generation function isn’t set up correctly.

- **Verify the Model’s Remote Code:**  
  Some models require particular arguments or additional configuration when using `trust_remote_code=True`. Check the model card or documentation on Hugging Face to see if there are extra parameters that need to be passed.

---

In summary, the error message  
`"Error generating response: 'NoneType' object is not callable"`  
appears because the text-generation pipeline isn’t properly set up for the EXAONE model. Fixing this involves instantiating the model using the appropriate class (i.e. `AutoModelForCausalLM`) and passing both the model and tokenizer to the pipeline. This ensures that all necessary generation methods are correctly bound and callable.

If you’re curious, you might also explore how custom pipelines for remote code models differ from the standard ones, and whether the model’s repository provides any usage examples specific to interactive chat scenarios.

---
https://copilot.microsoft.com/shares/EQkkM5yLouWmo7RKkBwyd
