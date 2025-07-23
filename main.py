import torch as th
from dct_diff import DCT, SoftOrthGradIteration, SteerSingleModel
from nnterp import StandardizedTransformer
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
import sys


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


def user_msgs_to_prompts(user_msgs: list[str], model: StandardizedTransformer, no_template_context: str | None = None) -> list[str]:
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

@tee_file()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device_map = "auto" if not cfg.cpu else "cpu"
    model = StandardizedTransformer(cfg.model_name, device_map=device_map)
    prompts = user_msgs_to_prompts(cfg.train_user_msgs, model, cfg.train_context)
    print(f"Running on prompts: {prompts}")
    with th.no_grad():
        with model.trace(prompts) as tracer:
            out = model.layers_output[cfg.source_layer].save()
            tracer.stop()
        median_norm = out.norm(dim=-1).median()
    print(f"Median norm: {median_norm}")
    steering_model = SteerSingleModel(
        model,
        cfg.source_layer,
        cfg.target_layer,
        cfg.batch_size,
        steering_factor=median_norm.item(),
    )
    trainer = SoftOrthGradIteration(
        num_steering_vectors=cfg.num_steering_vectors,
        prompts=prompts,
        steering_diffs_extractor=steering_model,
        logger_type="print",
        device=out.device,
    )
    steering_vectors = trainer.fit(num_epochs=cfg.num_epochs)
    hydra_path = HydraConfig.get().runtime.output_dir
    th.save(steering_vectors, Path(hydra_path) / "steering_vectors.pt")
    generations = steering_model.steered_generations(cfg.test_user_msgs, steering_vectors)





if __name__ == "__main__":
    main()
