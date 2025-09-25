import torch
from transformers import pipeline, AutoTokenizer, GenerationConfig
import time
import gc

def run_speculative_decoding_evaluation(
    prompt: str,
    model_name: str,
    assistant_model_name: str,
    do_sample: bool = True,
    strategy: str = None,
    **kwargs,
):
    """
    Runs a speculative decoding generation and prints performance statistics.
    A new pipeline is initialized for each call to ensure a clean evaluation environment.

    Args:
        prompt: The input prompt for generation.
        model_name: The name or path of the main model.
        assistant_model_name: The name or path of the assistant model.
        do_sample (bool): Whether to use sampling. Defaults to True.
        strategy (str): The vocab_mapping_strategy to use.
        **kwargs: Extra arguments for the generation config.
    """
    for _ in range(3):
        gc.collect()

    # Clear CUDA memory if available
    if torch.cuda.is_available():
        # Explicitly empty CUDA cache
        torch.cuda.empty_cache()

        # Force synchronization of CUDA threads
        torch.cuda.synchronize()

        # Collect inter-process CUDA memory
        torch.cuda.ipc_collect()

        # Print memory stats for debugging (optional)
        for i in range(torch.cuda.device_count()):
            print(
                f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f}GB",
                flush=True,
            )
            print(
                f"GPU {i} memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f}GB",
                flush=True,
            )

    print(
        "Memory cleared: Python memory garbage collected and GPU cache emptied.",
        flush=True,
    )

    print(f"\n{'='*20} Evaluation Strategy: {strategy or 'Default'} {'='*20}")

    # --- 1. Initialize Pipeline and Tokenizer ---
    print("Loading model and tokenizer...")
    try:
        pipe = pipeline(
            "text-generation",
            model=model_name,
            assistant_model=assistant_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    print("Model loaded successfully.")

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

    # --- 2. Timing and running generation ---

    start_time = time.perf_counter()
    print(start_time)
    # Run generation
    output = pipe(prompt, generation_config=generation_config)
    
    end_time = time.perf_counter()
    print(end_time)
    output_ids = tokenizer(output[0]['generated_text'], return_tensors="pt").input_ids 
 

    # --- 3. Calculating and printing performance statistics ---
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
    
    if pipe is not None:
        print(f"\n>>> Unloading model and cleaning memory after evaluation...")
        del pipe
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(">>> Cleaning complete.")

if __name__ == "__main__":
    # --- Global Setup ---
    MODEL_NAME = "/data/model/models/Qwen/Qwen3-32B"
    ASSISTANT_MODEL = "/data/model/models/Qwen/Qwen2.5-0.5B-Instruct"
    PROMPTS = [
        "The field of Artificial Intelligence has seen tremendous growth in recent years. One of the most exciting areas of research is",
        #"请用中文解释一下什么是相对论。",
        "Write a python function to calculate the factorial of a number.",
        #"Présentation de l’écrivain hugo"

        # --- New Prompts ---

        # 1. Creative Writing
        #"Escribe un poema corto sobre la lluvia en la ciudad.", # Spanish: Write a short poem about the city rain.
        #"「夏祭りの夜」をテーマにした俳句を三句作ってください。", # Japanese: Please create three haikus on the theme of "Summer Festival Night."
        #"Write a short story about an astronaut who discovers a new habitable planet.", # English: Write a short story about an astronaut who discovers a new habitable planet.
        #"اكتب قصة قصيرة عن روبوت يكتشف المشاعر الإنسانية.", # Arabic: Write a short story about a robot discovering human emotions.

        # 2. Information Synthesis & Explanation
        #"Was sind die Hauptunterschiede zwischen klassischer und Quantenphysik?", # German: What are the main differences between classical and quantum physics?
        #"김치찌개를 만드는 방법을 단계별로 설명해주세요.", # Korean: Please explain how to make kimchi stew step-by-step.
        "Summarize the plot of Shakespeare's 'Hamlet' in three paragraphs.", # English: Summarize the plot of Shakespeare's play 'Hamlet' in three paragraphs.
        #"Кратко изложите основные события романа 'Война и мир'.", # Russian: Briefly outline the main events of the novel 'War and Peace.'

        # 3. Code & Technical
        #"Explain what this SQL query does: SELECT department, COUNT(*) FROM employees WHERE start_date >= '2023-01-01' GROUP BY department HAVING COUNT(*) > 5;", # English: Explain the purpose of this SQL query.
        #"HTMLで基本的な会社の「お問い合わせ」フォームを作成してください。", # Japanese: Please use HTML to create a basic company 'Contact Us' form.
        #"Génère un exemple de code en Java pour se connecter à une base de données MySQL.", # French: Generate an example Java code to connect to a MySQL database.
        #"如何用 CSS 将一个 div 元素在其父容器中水平和垂直居中？", # Chinese: How to center a div element both horizontally and vertically within its parent container using CSS?

        # 4. Translation & Multi-language Tasks
        "Translate the following English proverb into French, Spanish, and German, and format the result as a JSON object: 'The early bird catches the worm.'", # English: Translate the following English proverb into French, Spanish, and German, and format the result as a JSON object: 'The early bird catches the worm.'

        # 5. Role-playing & Scenarios
        #"Simula una conversación en un restaurante donde pido la carta, ordeno mi comida y pido la cuenta.", # Spanish: Simulate a conversation in a restaurant where I ask for the menu, order my food, and ask for the bill.
        #"Rédige un e-mail professionnel à un client pour l'informer d'un retard de livraison. Sois poli et propose une solution.", # French: Write a professional email to a client to inform them of a delivery delay. Be polite and propose a solution.

        # 6. Logic & Problem Solving
        "I have a 5-gallon jug and a 3-gallon jug, and an unlimited supply of water. How can I measure out exactly 4 gallons?", # English: I have a 5-gallon jug and a 3-gallon jug, and an unlimited supply of water. How can I measure out exactly 4 gallons?

        # 7. Formatting & Data Handling
        #"Erstelle eine Markdown-Tabelle mit drei Spalten (Produkt, Preis, Verfügbarkeit) und vier Zeilen mit Beispieldaten für einen Online-Shop.", # German: Create a Markdown table with three columns (Product, Price, Availability) and four rows of example data for an online shop.
        "Convert the following text into a structured JSON object with keys 'name', 'age', and 'city': 'John Doe is 30 years old and lives in New York.'", # English: Convert the following text into a structured JSON object with keys 'name', 'age', and 'city': 'John Doe is 30 years old and lives in New York.'

        # 8. Factual Q&A
        #"भारत के स्वतंत्रता संग्राम में महात्मा गांधी की क्या भूमिका थी?", # Hindi: What role did Mahatma Gandhi play in India's independence struggle?
        #"Quali sono i principali monumenti da visitare a Roma?", # Italian: What are the main sights to visit in Rome?
        
    ]

    # --- Run all strategy evaluations ---
    for prompt in PROMPTS:
        print(f"\n{'#'*30} Testing with a new Prompt: '{prompt}' {'#'*30}\n")
        """
        run_speculative_decoding_evaluation(
            prompt=prompt, 
            model_name=MODEL_NAME, 
            assistant_model_name=ASSISTANT_MODEL,
            do_sample=True
        )
        """
        run_speculative_decoding_evaluation(
            prompt=prompt, 
            model_name=MODEL_NAME, 
            assistant_model_name=ASSISTANT_MODEL, 
            strategy="dtw"
        ) 
