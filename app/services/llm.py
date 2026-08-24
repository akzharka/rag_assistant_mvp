from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()


def parse_json_response(text: str):

    # Remove possible Markdown formatting
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(
            f"No JSON found in model response: {text}"
        )

    json_text = match.group()

    return json.loads(json_text)


def extract_contract_data(contract_text: str):

    prompt = f"""
            Extract structured information from the contract below.

            Return ONLY valid JSON.

            Use exactly this structure:

            {{
                "contract_type": "",
                "party_a": "",
                "party_b": "",
                "effective_date": "",
                "expiration_date": "",
                "payment_terms": "",
                "termination_terms": "",
                "governing_law": ""
            }}

            Rules:
            - Use only information found in the contract.
            - Do not invent information.
            - If information is missing, use "Not specified".
            - Do not include explanations.
            - Do not include Markdown.
            - Return JSON only.

            CONTRACT:

            {contract_text}
            """

    messages = [
        {
            "role": "system",
            "content": "You are a contract analysis assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer(
        [formatted_prompt],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=500,
            do_sample=False
        )

    # Remove original prompt tokens
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]

    response_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return parse_json_response(response_text)
