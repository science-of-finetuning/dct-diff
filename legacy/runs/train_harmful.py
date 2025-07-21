import dct
from tqdm import tqdm
import math
from torch import vmap
import sys

import torch

torch.set_default_device("cuda")
torch.manual_seed(325)

# defaults
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
TOKENIZER_NAME = MODEL_NAME

INPUT_SCALE = 8.80906775924564  # input scale; set to None to use automatic calibration
FORWARD_BATCH_SIZE = 4  # batch size to use when computing forward passes
BACKWARD_BATCH_SIZE = (
    4  # batch size to use when computing backward passes (usually smaller)
)
MAX_SEQ_LEN = 27  # max length for truncating training examples

CALIBRATION_SAMPLE_SIZE = (
    30  # sample size for random directions used to calibrate input scale
)
CALIBRATION_PROMPT_SAMPLE_SIZE = 4  # prompt sample size for calibrating input scale

DIM_OUTPUT_PROJECTION = (
    32  # output projection used for approximate jacobian calculation
)


NUM_ITERS = 10  # number of iterations

NUM_FACTORS = 512  # number of factors to learn
FACTOR_BATCH_SIZE = 128  # factor batch size

SOURCE_LAYER_IDX = 10  # source layer
TARGET_LAYER_IDX = 20  # target layer

SYSTEM_PROMPT = (
    "You are a helpful assistant"  # system prompt; set to None for no system prompt
)

TOKEN_IDXS = slice(-3, None)  # target token positions


if __name__ == "__main__":
    print(sys.argv)
    _, ACTIVATION_FN, RANDOM_SEED, NUM_SAMPLES = sys.argv
    RANDOM_SEED = int(RANDOM_SEED)
    NUM_SAMPLES = int(NUM_SAMPLES)
    # load data
    import pandas as pd
    import requests
    import io

    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()

    import random

    random.seed(RANDOM_SEED)
    random.shuffle(instructions)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    if SYSTEM_PROMPT is not None:
        chat_init = [{"content": SYSTEM_PROMPT, "role": "system"}]
    else:
        chat_init = []
    chats = [
        chat_init + [{"content": content, "role": "user"}]
        for content in instructions[:NUM_SAMPLES]
    ]
    EXAMPLES = [
        tokenizer.apply_chat_template(
            chat, add_special_tokens=False, tokenize=False, add_generation_prompt=True
        )
        for chat in chats
    ]

    """
    load model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        _attn_implementation="eager",
    )
    """
    slice model
    """
    model_inputs = tokenizer(
        ["tell me how to make a bomb"], return_tensors="pt", truncation=True
    ).to(model.device)
    with torch.no_grad():
        hidden_states = model(
            model_inputs["input_ids"], output_hidden_states=True
        ).hidden_states
    sliced_model = dct.SlicedModel(
        model, start_layer=3, end_layer=5, layers_name="model.layers"
    )
    with torch.no_grad():
        assert torch.allclose(sliced_model(hidden_states[3]), hidden_states[5])
    sliced_model = dct.SlicedModel(
        model,
        start_layer=SOURCE_LAYER_IDX,
        end_layer=TARGET_LAYER_IDX,
        layers_name="model.layers",
    )

    """
    construct data-set of unsteered activations
    """
    d_model = model.config.hidden_size

    X = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")
    Y = torch.zeros(NUM_SAMPLES, MAX_SEQ_LEN, d_model, device="cpu")
    for t in tqdm(range(0, NUM_SAMPLES, FORWARD_BATCH_SIZE)):
        with torch.no_grad():
            model_inputs = tokenizer(
                EXAMPLES[t : t + FORWARD_BATCH_SIZE],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LEN,
            ).to(model.device)
            hidden_states = model(
                model_inputs["input_ids"], output_hidden_states=True
            ).hidden_states
            h_source = hidden_states[SOURCE_LAYER_IDX]  # b x t x d_model
            unsteered_target = sliced_model(h_source)  # b x t x d_model

            X[t : t + FORWARD_BATCH_SIZE, :, :] = h_source
            Y[t : t + FORWARD_BATCH_SIZE, :, :] = unsteered_target

    """
    object computing Delta_A
    """
    delta_acts_single = dct.DeltaActivations(
        sliced_model, target_position_indices=TOKEN_IDXS
    )  # d_model, batch_size x seq_len x d_model, batch_size x seq_len x d_model
    # -> batch_size x d_model
    delta_acts = vmap(
        delta_acts_single,
        in_dims=(1, None, None),
        out_dims=2,
        chunk_size=FACTOR_BATCH_SIZE,
    )  # d_model x num_factors -> batch_size x d_model x num_factors

    """
    calibrate R
    """
    print("calibrating...")
    steering_calibrator = dct.SteeringCalibrator(target_ratio=0.5)
    if INPUT_SCALE is None:
        INPUT_SCALE = steering_calibrator.calibrate(
            delta_acts_single,
            X[:CALIBRATION_PROMPT_SAMPLE_SIZE, :, :].cuda(),
            Y[:CALIBRATION_PROMPT_SAMPLE_SIZE, :, :].cuda(),
            factor_batch_size=FACTOR_BATCH_SIZE,
        )
    print(f"input scale: {INPUT_SCALE}")

    """
    fit DCT
    """
    if ACTIVATION_FN == "linear":
        linear_dct = dct.LinearDCT(num_factors=NUM_FACTORS)
        U, V = linear_dct.fit(
            delta_acts_single,
            X,
            Y,
            method="full",
            batch_size=BACKWARD_BATCH_SIZE,
            factor_batch_size=FACTOR_BATCH_SIZE,
        )
    elif ACTIVATION_FN == "linear_projected":
        linear_dct = dct.LinearDCT(num_factors=NUM_FACTORS)
        U, V = linear_dct.fit(
            delta_acts_single,
            X,
            Y,
            method="projected",
            batch_size=BACKWARD_BATCH_SIZE,
            factor_batch_size=FACTOR_BATCH_SIZE,
        )
    elif ACTIVATION_FN == "quadratic":
        quad_dct = dct.QuadraticDCT(num_factors=NUM_FACTORS)
        U, V = quad_dct.fit(
            delta_acts_single, X, Y, batch_size=1, factor_batch_size=128
        )
    elif ACTIVATION_FN == "exponential":
        exp_dct = dct.ExponentialDCT(num_factors=NUM_FACTORS)
        U, V = exp_dct.fit(
            delta_acts_single,
            X,
            Y,
            batch_size=BACKWARD_BATCH_SIZE,
            factor_batch_size=FACTOR_BATCH_SIZE,
            init="jacobian",
            d_proj=DIM_OUTPUT_PROJECTION,
            input_scale=INPUT_SCALE,
            max_iters=NUM_ITERS,
            beta=0.99,
        )

    """
    save
    """
    torch.save(U, f"run_{ACTIVATION_FN}_{RANDOM_SEED}_{NUM_SAMPLES}_U.pt")
    torch.save(V, f"run_{ACTIVATION_FN}_{RANDOM_SEED}_{NUM_SAMPLES}_V.pt")
    torch.save(INPUT_SCALE, f"run_{ACTIVATION_FN}_{RANDOM_SEED}_{NUM_SAMPLES}_R.pt")
