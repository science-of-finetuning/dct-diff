import json
import time
import sys
from pathlib import Path

from coolname import generate_slug
import torch as th
from nnterp import StandardizedTransformer
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from dct_diff import SoftOrthGradIteration, SteerSingleModel
from dct_diff.utils import cleanup_generations

def tee_file():
    """Decorator that tees stdout and stderr to multiple files."""

    class TeeFile:
        def __init__(self, *files):
            self.files = files

        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    def decorator(func):
        def wrapper(*args, **kwargs):
            hydra_path = HydraConfig.get().runtime.output_dir
            stdout_file = open(Path(hydra_path) / "stdout.log", "w")
            stderr_file = open(Path(hydra_path) / "stderr.log", "w")
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = TeeFile(sys.__stdout__, stdout_file)
                sys.stderr = TeeFile(sys.__stderr__, stderr_file)
                return func(*args, **kwargs)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                stdout_file.close()
                stderr_file.close()

        return wrapper

    return decorator


def user_msgs_to_prompts(
    user_msgs: list[str],
    model: StandardizedTransformer,
    no_template_context: str | None = None,
) -> list[str]:
    """
    Convert user messages to prompts.
    """
    if model.tokenizer.chat_template is None:
        # context = "System: You are a helpful assistant.\nUser: I keep forgetting to water my plants. Any tips?\nAssistant: Set phone reminders for specific days, or use the \"finger test\" - stick your finger an inch into the soil, and if it's dry, it's time to water.\n"
        no_template_context = no_template_context or ""
        prompts = [no_template_context + f"User: {m}\nAssistant:" for m in user_msgs]
    else:
        convos = [[dict(role="user", content=m)] for m in user_msgs]
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
    return prompts


@hydra.main(version_base=None, config_path="configs", config_name="config")
@tee_file()
def main(cfg: DictConfig) -> None:
    device_map = "auto" if not cfg.cpu else "cpu"
    print(f"Running model: {cfg.model_name} on device: {device_map}")

    model = StandardizedTransformer(
        cfg.model_name, device_map=device_map, attn_implementation=None
    )
    prompts = user_msgs_to_prompts(cfg.training.user_messages, model, cfg.context)
    print(f"Running on prompts: {prompts}")
    with th.no_grad():
        with model.trace(prompts) as tracer:
            out = model.layers_output[cfg.source_layer].save()
            tracer.stop()
        median_norm = out.norm(dim=-1).median().item()
    print(f"Median norm: {median_norm}")
    steering_model = SteerSingleModel(
        model,
        cfg.source_layer,
        cfg.target_layer,
        cfg.batch_size,
        steering_factor=median_norm,
    )
    trainer = SoftOrthGradIteration(
        num_steering_vectors=cfg.num_steering_vectors,
        prompts=prompts,
        steering_diffs_extractor=steering_model,
        logger_type="print",
        device=out.device,
    )
    steering_vectors = trainer.fit(num_epochs=cfg.num_epochs)
    test_prompts = user_msgs_to_prompts(
        cfg.generation.user_messages, model, cfg.context
    )
    assert len(test_prompts) == len(
        cfg.generation.user_messages
    ), f"test_prompts length {len(test_prompts)} != test_user_msgs length {len(cfg.generation.user_messages)}"
    hydra_path = Path(HydraConfig.get().runtime.output_dir)
    th.save(steering_vectors, hydra_path / "steering_vectors.pt")
    rel_steering_factors = [0.5, 1, 2, 5, 10]
    steering_factors = th.tensor(rel_steering_factors) * median_norm
    print(
        f"Testing with steering factors: {steering_factors.int().tolist()} and prompts:\n- ",
        end="",
    )
    print("\n- ".join(test_prompts))
    generations_all = steering_model.steered_generations(
        test_prompts,
        steering_vectors,
        steering_factors,
        steer_type="all",
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
    )
    generations_input_only = steering_model.steered_generations(
        test_prompts,
        steering_vectors,
        steering_factors,
        steer_type="input_only",
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
    )
    with model.generate(
        test_prompts, max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample
    ) as tracer:
        out = model.generator.output.save()
    baseline_generations = cleanup_generations(model.tokenizer.batch_decode(out), model.tokenizer)

    assert len(generations_all) == len(
        steering_vectors
    ), f"generations_all length {len(generations_all)} != steering_vectors length {len(steering_vectors)}"
    assert len(generations_input_only) == len(
        steering_vectors
    ), f"generations_input_only length {len(generations_input_only)} != steering_vectors length {len(steering_vectors)}"

    steered_generation_dict = []
    for i in range(len(steering_vectors)):
        vector_data = {}
        for k, prompt in enumerate(cfg.generation.user_messages):
            vector_data[prompt] = {
                steering_factor.item(): {
                    "steer_input": generations_input_only[i][j][k],
                    "steer_all": generations_all[i][j][k],
                }
                for j, steering_factor in enumerate(steering_factors)
            }
        steered_generation_dict.append(vector_data)

    closest_tokens = model.get_topk_closest_tokens(steering_vectors.to(model.device))

    exp_id = f"{int(time.time())}_{generate_slug(2)}"
    save_data = {
        "baseline_generations": {
            prompt: baseline_generations[i]
            for i, prompt in enumerate(cfg.generation.user_messages)
        },
        "model_name": cfg.model_name,
        "steered_generation": steered_generation_dict,
        "steering_factors": steering_factors.tolist(),
        "rel_steering_factors": rel_steering_factors,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "exp_id": exp_id,
        "closest_tokens": closest_tokens,
        "median_norm": median_norm,
    }

    with open(hydra_path / "steering_data.json", "w") as f:
        json.dump(save_data, f, indent=2)

    results_path = (
        Path("results") / "vanilla_dct" / cfg.model_name.split("/")[-1] / exp_id
    )
    results_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {results_path}")

    with open(results_path / "steering_data.json", "w") as f:
        json.dump(save_data, f, indent=2)

    th.save(steering_vectors, results_path / "steering_vectors.pt")


if __name__ == "__main__":
    main()
