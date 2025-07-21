import torch as th
from dct_diff import DCT, SoftOrthGradIteration, SteerSingleModel
from nnterp import StandardizedTransformer
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-n", "--num-epochs", type=int, default=4)
args = parser.parse_args()
device_map = "auto"
device_map = "cpu"
model = StandardizedTransformer("gpt2", device_map=device_map)
context = "System: You are a helpful assistant.\nUser: I keep forgetting to water my plants. Any tips?\nAssistant: Set phone reminders for specific days, or use the \"finger test\" - stick your finger an inch into the soil, and if it's dry, it's time to water. Most houseplants need water every 1-2 weeks.\n"
prompts = ["User: Tell me a story!\nAssistant:", "User: Tell me a joke\nAssistant:"]
prompts = [context + p for p in prompts]
with th.no_grad():
    with model.trace(prompts) as tracer:
        out = model.layers_output[5].save()
        tracer.stop()
    median_norm = out.norm(dim=-1).median()
print(f"Median norm: {median_norm}")
steering_model = SteerSingleModel(model, 5, 8, 16, steering_factor=median_norm.item())
trainer = SoftOrthGradIteration(
    num_steering_vectors=10,
    prompts=prompts,
    steering_diffs_extractor=steering_model,
    logger_type="print",
)
trainer.fit(num_epochs=args.num_epochs)
