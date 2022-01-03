import ctypes
import multiprocessing as mp
import threading
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import zip_longest
from typing import Iterable

import torch
import torch.nn as nn
import torch.utils.data
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from hivemind.utils.logging import get_logger

logger = get_logger(__name__)

GRAD_CHUNK_NUMEL = 20_000_000  # transfer at most this many gradients to host per call to save memory


class TPUManager(mp.Process):
    """Auxiliary class that manages model training over an array of TPU cores"""

    def __init__(
        self,
        model,
        dataset,
        *,
        collate_fn: callable = None,
        nprocs: int = 8,
        prefetch: int = 16,
        batch_size_per_device: int = 1,
        grad_accumulation_steps: int = 1,
        seed_base: int = 42,
        post_init: callable = lambda model: model.tie_weights(),
        start: bool,
    ):
        super().__init__()
        self.lock = mp.Lock()
        self.nprocs, self.prefetch, self.seed_base = nprocs, prefetch, seed_base
        self.batch_size_per_device, self.grad_accumulation_steps = batch_size_per_device, grad_accumulation_steps
        self.collate_fn = collate_fn
        self.step_triggered, self.step_finished = mp.Event(), mp.Event()
        self._synchronizer = TPUSynchronizer(model, post_init)
        self._data_manager = TPUDataManager(dataset, nprocs, prefetch)

        # shared fields for communicating statistics after each step
        self.should_load_parameters = mp.Value(ctypes.c_bool, False)
        self.gradients_accumulated = mp.Value(ctypes.c_long, 0)
        self.loss_accumulated = mp.Value(ctypes.c_double, 0)
        if start:
            self.start()

    def run(self):
        thread = threading.Thread(
            target=partial(xmp.spawn, self.runner, nprocs=self.nprocs, start_method="fork"), daemon=True
        )
        thread.start()
        thread.join()

    def update_model_parameters(self, new_host_parameters):
        """Schedule TPUs to update model parameters during at the beginning of the next step"""
        with self.lock, torch.no_grad():
            self._synchronizer.set_host_parameters(new_host_parameters)
            self.should_load_parameters.value = True

    def get_aggregated_gradients(self):
        """Get current accumulated gradients from the master model"""
        with self.lock, torch.no_grad():
            return self._synchronizer.get_aggregated_gradients()

    def zero_grad(self):
        """Reset master accumulated gradients to zeros"""
        with self.lock, torch.no_grad():
            for param in self._synchronizer.master_model.parameters():
                param.grad.zero_()

    def step(self):
        """run forward/backward step with all TPUs, collect gradients"""
        self.loss_accumulated.value = self.gradients_accumulated.value = 0
        self.step_finished.clear()
        self.step_triggered.set()
        self.step_finished.wait()
        return self.loss_accumulated.value, self.gradients_accumulated.value

    def runner(self, tpu_index):
        """Run training steps from the perspective of a single TPU core"""
        ### compute loss and gradients

        # acquire the (unique) Cloud TPU core corresponding to this process's index
        device = xm.xla_device()
        logger.info(f"Process {tpu_index} is using {xm.xla_real_devices([str(device)])[0]}")

        # set random seed for
        torch.manual_seed(self.seed_base + tpu_index)
        print("MEASURE RAM NOW!")
        import time; time.sleep(60)

        # use staged init to minimize peak RAM usage
        for init_index in range(xm.xrt_world_size()):
            xm.rendezvous(f"init_{init_index}")
            if tpu_index == init_index:
                model = self._synchronizer.get_device_model_replica(device)
                data_loader = self._data_manager.get_device_dataloader(
                    batch_size=self.batch_size_per_device, num_workers=0, collate_fn=self.collate_fn, pin_memory=False
                )
                data_loader_iter = iter(data_loader)
                logger.info(f"Process {tpu_index} initialized.")

        xm.rendezvous("init_finished")
        print("INIT_FINISHED")

        while True:
            self.step_triggered.wait()
            xm.rendezvous("before_step")
            if xm.is_master_ordinal():
                self.step_triggered.clear()

            if bool(self.should_load_parameters.value):
                with self.lock if xm.is_master_ordinal() else nullcontext():
                    print("LOADING_PARAMS")
                    self._synchronizer.send_params_to_device(model)
                    self.should_load_parameters.value = False


            loss = 0.0
            for i in range(self.grad_accumulation_steps):
                print("FWD")
                inputs = next(data_loader_iter)
                outputs = model(**inputs)
                loss_i = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss_i = loss_i / (self.grad_accumulation_steps * self.nprocs)
                print("BWD")
                loss_i.backward()
                loss += loss_i
                del inputs, outputs, loss_i

            xm.rendezvous("after_step")
            print("AFTERSTEP")
            ### aggregate gradients from TPUs
            with self.lock if xm.is_master_ordinal() else nullcontext():
                self._synchronizer.aggregate_grads_on_host(model, add=True)
            # clear aggregated gradients from all devices
            model.zero_grad()

            ### accumulate statistics to host
            loss = xm.all_reduce(xm.REDUCE_SUM, loss, scale=1.0)
            xm.do_on_ordinals(self._mark_step_finished, data=(loss,), ordinals=(0,))

    def _mark_step_finished(self, loss):
        self.gradients_accumulated.value = self.batch_size_per_device * self.nprocs * self.grad_accumulation_steps
        self.loss_accumulated.value = float(loss)
        self.step_finished.set()


class TPUSynchronizer:
    """An auxiliary class for manipulating parameters and gradients without producing a ton of XLA graphs"""

    def __init__(self, model: nn.Module, post_init: callable = lambda model: model.tie_weights()):
        self.master_model = model.share_memory()
        self.post_init = post_init
        for param in self.master_model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad = param.grad.share_memory_()

    def get_device_model_replica(self, device: torch.device):
        with torch.no_grad():
            replica = self.master_model.to(device)
        self.post_init(replica)
        return replica

    def set_host_parameters(self, new_host_parameters):
        return self._assign(source=self.master_model.parameters(), target=new_host_parameters, add=False, strict=True)

    def get_aggregated_gradients(self):
        return [param.grad for param in self.master_model.parameters()]

    def send_params_to_device(self, replica: nn.Module):
        """Copy params from master_model to this device_model replica"""
        with torch.no_grad():
            replica_params = list(replica.parameters())
            master_params = list(self.master_model.parameters())
            master_params = xm.send_cpu_data_to_device(master_params, xm.xla_device())
            self._assign(source=master_params, target=replica_params, add=False)
            xm.rendezvous("params_replicated")

    def aggregate_grads_on_host(self, replica: nn.Module, *, add: bool):
        """Aggregate grads from all tpu devices and move them to host"""
        with torch.no_grad():
            replica_grads = [param.grad for param in replica.parameters()]
            replica_grads = xm.all_reduce(xm.REDUCE_SUM, replica_grads, scale=1.0)
            master_grads = [hp.grad for hp in self.master_model.parameters()]

            ordinal = xm.get_ordinal()
            source_chunk, target_chunk, chunk_size = [], [], 0
            print("TOTAL SIZE", len(replica_grads))
            assert len(master_grads) == len(replica_grads)
            for source, target in zip(replica_grads, master_grads):
                if len(source_chunk) != 0 and chunk_size + source.numel() >= GRAD_CHUNK_NUMEL:
                    print("MOVING CHUNK", len(source_chunk), len(target_chunk), chunk_size)
                    xm.do_on_ordinals(
                        lambda *source_chunk: self._assign(source=source_chunk, target=target_chunk, add=add),
                        data=tuple(source_chunk),
                        ordinals=(0,),
                    )  # ^-- do_on_ordinals already runs rendezvous at the end
                    source_chunk, target_chunk, chunk_size = [], [], 0
                source_chunk.append(source)
                target_chunk.append(target)
                chunk_size += source.numel()
            print("MOVING FINAL CHUNK", len(source_chunk), len(target_chunk), chunk_size)
            xm.do_on_ordinals(
                lambda *source_chunk: self._assign(source=source_chunk, target=target_chunk, add=add),
                data=tuple(source_chunk),
                ordinals=(0,),
            ) # ^-- do_on_ordinals already runs rendezvous at the end

    def _assign(self, source: Iterable[torch.Tensor], target: Iterable[torch.Tensor], add: bool, strict: bool = False):
        for source_tensor, target_tensor in zip_longest(source, target):
            print(end="ASSIGN\n")
            assert source_tensor is not None or target_tensor is not None, "Source and target length must match exactly"
            if strict:
                assert source_tensor.shape == target_tensor.shape
                assert source_tensor.device == target_tensor.device
                assert source_tensor.dtype == target_tensor.dtype
            if add:
                target_tensor.add_(source_tensor)
            else:
                target_tensor.copy_(source_tensor)


class TPUDataManager:
    """An auxiliary class that loads centralized dataset from master into multiple TPU devices"""

    def __init__(self, dataset: torch.utils.data.Dataset, nprocs: int, master_prefetch: int = 4):
        self.dataset, self.nprocs = dataset, nprocs
        self.device_queues = [mp.Queue(master_prefetch) for _ in range(nprocs)]
        self._loader_thread = threading.Thread(target=self._load_data_into_queues)
        self._loader_thread.start()

    def _load_data_into_queues(self):
        try:
            for i, batch in enumerate(self.dataset):
                self.device_queues[i % self.nprocs].put(batch)
        finally:
            logger.warning("Minibatch generator finished.")

    def get_device_dataloader(self, **kwargs):
        data_loader = torch.utils.data.DataLoader(QueueDataset(self.device_queues[xm.get_ordinal()]), **kwargs)
        return pl.ParallelLoader(data_loader, [xm.xla_device()]).per_device_loader(xm.xla_device())


class QueueDataset(torch.utils.data.IterableDataset):
    """A dataset that ceaselessly iterates over a queue"""

    def __init__(self, queue: mp.Queue):
        super().__init__()
        self.queue = queue

    def __iter__(self):
        while True:
            yield self.queue.get()

    def __len__(self):
        return (
            10 ** 12
        )  # TODO deprecate this when the issue is resolved: https://github.com/googlecolab/colabtools/issues/2237
