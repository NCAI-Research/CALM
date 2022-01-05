#!/usr/bin/env python
import time

import torch
import transformers.training_args
import wandb
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from transformers import HfArgumentParser

import utils
from arguments import (CollaborativeArguments, TPUTrainerArguments,
                       TrainingPeerArguments)
from callback import CollaborativeCallback
from lib.modules.linear import SharedMatrix, AdaptedLinear
from lib.training.tpu import TPUManager
from tasks.mlm.task import MLMTrainingTask

use_hivemind_log_handler("in_root_logger")
logger = get_logger()

transformers.training_args.is_torch_tpu_available = lambda: False  # disable builtin TPU support to use custom code


def main():
    parser = HfArgumentParser((TrainingPeerArguments, TPUTrainerArguments, CollaborativeArguments))
    peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses()

    logger.info(f"Found {len(peer_args.initial_peers)} initial peers: {peer_args.initial_peers}")
    if len(peer_args.initial_peers) == 0:
        logger.warning("Please specify at least one network endpoint in initial peers.")

    utils.setup_logging(trainer_args)
    task = MLMTrainingTask(peer_args, trainer_args, collab_args)
    model, optimizer = task.model, task.collaborative_optimizer
    # ^-- note: we know that the optimizer is initialized at this point

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SharedMatrix):
                module.matrix.data = module.matrix.bfloat16()
            if isinstance(module, AdaptedLinear):
                module.adapter_first.data = module.adapter_first.bfloat16()
                module.adapter_second.data = module.adapter_second.bfloat16()

    # BEGIN init TPU
    assert trainer_args.do_train and not trainer_args.do_eval
    tpu_manager = TPUManager(
        model,
        dataset=task.training_dataset,
        collate_fn=task.data_collator,
        grad_accumulation_steps=trainer_args.gradient_accumulation_steps,
        batch_size_per_device=trainer_args.per_device_train_batch_size,
        nprocs=trainer_args.num_tpus,
        post_init=lambda model: model.tie_weights(),
        compress_grads=True,
        start=True,
    )
    assert model is task.model and model is tpu_manager._synchronizer.master_model
    assert any(param.dtype == torch.float16 for param in optimizer.state_averager.main_parameters)
    assert not any(param.dtype == torch.float16 for group in optimizer.state_averager.optimizer.param_groups for param in group["params"])


    # warmup tpus
    logger.info("Waiting for TPUs to warm up, this may take a minute...")
    # for i in range(10):
    #     tpu_manager.step()
    #     logger.info("Warmup step 1 / 3 done.")

    for i in range(10):
        tpu_manager.update_model_parameters(model.parameters())
        tpu_manager.step()
        logger.info("Warmup step 2 / 3 done.")

    for i in range(10):
        # TODO
        tpu_manager.step()
        tpu_manager.get_aggregated_gradients()
        tpu_manager.zero_grad()
        logger.info("Warmup step 3 / 3 done.")

    # END init TPU
    raise NotImplementedError()


    def push_params_onto_tpu():
        logger.info("Pushing new params onto TPU.")
        tpu_manager.update_model_parameters(model.parameters())
        tpu_manager.zero_grad()

    raise NotImplementedError()
    collaborative_optimizer = task.collaborative_optimizer
    collaborative_optimizer.callbacks.on_after_global_step.add(push_params_onto_tpu)
    collaborative_optimizer.callbacks.on_load_state_from_peers(push_params_onto_tpu)

    collaborative_training_callback = CollaborativeCallback(task, peer_args)

    state = transformers.TrainerState()
    control = transformers.TrainerControl()
    collaborative_training_callback.on_train_begin(trainer_args, state, control)
    tpu_manager.update_model_parameters(model.parameters())

    wandb.init(project=trainer_args.wandb_project, name=trainer_args.run_name)

    while True:
        start_time = time.perf_counter()
        loss, num_accumulated = tpu_manager.step()
        time_delta = time.perf_counter() - start_time
        logger.info(f"Accumulated {num_accumulated} gradients at {num_accumulated / time_delta:.3f} samples/second.")
        wandb.log({"train/loss": loss, "train/learning_rate": collaborative_optimizer.scheduler.get_lr()[0]})

        with torch.no_grad():
            for param, grad_from_tpu in zip(model.parameters(), tpu_manager.get_aggregated_gradients()):
                param.grad[...] = grad_from_tpu
            collaborative_optimizer.step()

        state.log_history.append(dict(loss=loss))
        collaborative_training_callback.on_step_end(trainer_args, state, control)


if __name__ == "__main__":
    main()
