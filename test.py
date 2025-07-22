import torch as th
from dct_diff import DCT, SoftOrthGradIteration, SteerSingleModel
from nnterp import StandardizedTransformer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-n", "--num-epochs", type=int, default=4)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("-v", "--num-steering-vectors", type=int, default=10)
parser.add_argument("-s", "--source-layer", type=int, default=5)
parser.add_argument("-t", "--target-layer", type=int, default=8)
parser.add_argument("-b", "--batch-size", type=int, default=16)
parser.add_argument("-m", "--model", type=str, default="gpt2")
args = parser.parse_args()
device_map = "auto" if not args.cpu else "cpu"
model = StandardizedTransformer(args.model, device_map=device_map)
user_msgs = ["Tell me a story!", "Tell me a joke"]
convos = [
    [dict(role="user", content=m)] for m in user_msgs
]
if model.tokenizer.chat_template is None:
    context = "System: You are a helpful assistant.\nUser: I keep forgetting to water my plants. Any tips?\nAssistant: Set phone reminders for specific days, or use the \"finger test\" - stick your finger an inch into the soil, and if it's dry, it's time to water. Most houseplants need water every 1-2 weeks.\n"
    prompts = [context + f"User: {m}\nAssistant:" for m in user_msgs]
else:
    prompts = model.tokenizer.apply_chat_template(
        convos, add_generation_prompt=True, tokenize=False
    )
    if model.tokenizer.bos_token is not None:
        clean_prompts = []
        for p in prompts:
            if p.startswith(model.tokenizer.bos_token):
                clean_prompts.append(p[len(model.tokenizer.bos_token) :])
            else:
                clean_prompts.append(p)
        prompts = clean_prompts
print(f"Running on prompts: {prompts}")
with th.no_grad():
    with model.trace(prompts) as tracer:

        out = model.layers_output[args.source_layer].save()
        tracer.stop()
    median_norm = out.norm(dim=-1).median()
print(f"Median norm: {median_norm}")
steering_model = SteerSingleModel(
    model,
    args.source_layer,
    args.target_layer,
    args.batch_size,
    steering_factor=median_norm.item(),
)
trainer = SoftOrthGradIteration(
    num_steering_vectors=args.num_steering_vectors,
    prompts=prompts,
    steering_diffs_extractor=steering_model,
    logger_type="print",
)
trainer.fit(num_epochs=args.num_epochs)
