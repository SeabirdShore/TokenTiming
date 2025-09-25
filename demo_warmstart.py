import torch
from transformers import pipeline, AutoTokenizer, GenerationConfig
import time
import gc

def run_generation_and_evaluate(
    pipe,
    tokenizer,
    prompt: str,
    do_sample: bool = True,
    strategy: str = None,
    **kwargs,
):
    """
    Runs a generation and prints performance statistics.
    This function is no longer responsible for loading and unloading the model, but receives a pre-loaded pipeline.

    Args:
        pipe: The pre-loaded text generation pipeline.
        tokenizer: The pre-loaded tokenizer.
        prompt: The input prompt.
        do_sample (bool): Whether to use sampling. Defaults to True.
        strategy (str): The vocab_mapping_strategy to use.
        **kwargs: Additional generation config parameters.
    """
    # Package all configurations into a GenerationConfig object
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=do_sample,
        temperature=0.6,
        pad_token_id=pipe.tokenizer.eos_token_id,
        mapping_strategy=strategy,
        **kwargs
    )

    # Encode the input to calculate the number of newly generated tokens
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(pipe.device)

    # --- Timing and running generation ---
    start_time = time.perf_counter()
    output = pipe(prompt, generation_config=generation_config)
    end_time = time.perf_counter()
    output_ids = tokenizer(output[0]['generated_text'], return_tensors="pt").input_ids

    # --- Calculating and printing performance statistics ---
    duration = end_time - start_time
    num_input_tokens = input_ids.shape[1]
    num_output_tokens = output_ids.shape[1]
    num_new_tokens = num_output_tokens - num_input_tokens
    tokens_per_second = num_new_tokens / duration if duration > 0 else 0

    print(f"\n--- Generation Result ---")
    print(output[0]['generated_text'])
    print("\n--- Performance Metrics ---")
    print(f"Generation Time: {duration:.4f} seconds")
    print(f"Number of new tokens generated: {num_new_tokens}")
    print(f"Generation Speed: {tokens_per_second:.2f} tokens/second")
    print(f"{'='*50}\n")

def main_evaluation():
    """
    The main evaluation function, responsible for loading the model once, performing a warm-up, and iterating through all prompts for evaluation.
    """
    # --- Global Setup ---
    MODEL_NAME = "/data/model/models/Qwen/Qwen3-32B"
    ASSISTANT_MODEL = "/data/model/models/Qwen/Qwen2.5-0.5B-Instruct"
    PROMPTS = [
        "The field of Artificial Intelligence has seen tremendous growth in recent years. One of the most exciting areas of research is",
        "Write a python function to calculate the factorial of a number.",
        "Summarize the plot of Shakespeare's 'Hamlet' in three paragraphs.",
        "I have a 5-gallon jug and a 3-gallon jug, and an unlimited supply of water. How can I measure out exactly 4 gallons?",
        "Convert the following text into a structured JSON object with keys 'name', 'age', and 'city': 'John Doe is 30 years old and lives in New York.'",
    ]

    # --- 1. Load Pipeline and Tokenizer once ---
    print("Loading model and tokenizer...")
    try:
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            assistant_model=ASSISTANT_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    print("Model loaded successfully.")
    print("\nRunning warm-up phase...")
    # --- 2. Warm-up Phase ---
    # Run a few small-scale generations to ensure the model is fully loaded into memory and GPU performance is optimized
    for _ in range(2):
        run_generation_and_evaluate(
            pipe,
            tokenizer,
            prompt="Hello, world!",
            do_sample=True,
            strategy="dtw"
        )
    print("Warm-up phase complete.")

    # --- 3. Iterate through all prompts and evaluate ---
    for prompt in PROMPTS:
        print(f"\n{'#'*30} Testing with a new Prompt: '{prompt}' {'#'*30}\n")
        print(f"\n{'='*20} Evaluation Strategy: dtw {'='*20}")
        run_generation_and_evaluate(
            pipe=pipe,
            tokenizer=tokenizer,
            prompt=prompt,
            strategy="dtw"
        )

    # --- 4. Unload model and clean memory after evaluation is complete ---
    print(f"\n>>> All evaluations complete, unloading model and cleaning memory...")
    del pipe
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(">>> Cleaning complete.")

if __name__ == "__main__":
    main_evaluation()
