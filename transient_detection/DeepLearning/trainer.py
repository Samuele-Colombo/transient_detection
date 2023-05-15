"""Mostly copy-pasted from https://github.com/ramyamounir/Template/blob/main/lib/trainers/trainer.py"""

import torch
import sys
import os.path as osp
from glob import glob
import gc
import math
import os, functools, datetime, time
from tqdm import tqdm

from transient_detection.DeepLearning.fileio import checkdir
from transient_detection.DeepLearning.tensorboard import get_writer, TBWriter
from transient_detection.DeepLearning.scheduler import cosine_scheduler
from transient_detection.DeepLearning.distributed import MetricLogger
from transient_detection.DeepLearning.utilities import print_with_rank_index

class Trainer:

    def __init__(self, args, train_loader, validation_loader, model, loss, optimizer):

        self.args = args
        self.train_gen = train_loader
        self.validation_gen = validation_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.fp16_scaler = torch.cuda.amp.GradScaler() if args["Trainer"]["fp16"] else None
        self.print = functools.partial(print_with_rank_index, int(os.environ.get("SLURM_LOCALID")) if os.environ.get("SLURM_LOCALID") else 0)

        # === TB writers === #
        if args["main"]:	
            self.writer = get_writer(args)
            self.lr_sched_writer = TBWriter(self.writer, 'scalar', 'Schedules/Learning Rate')			
            self.loss_writer = TBWriter(self.writer, 'scalar', 'Loss/total')
            self.val_loss_writer = TBWriter(self.writer, 'scalar', 'Validation Loss/total')

            checkdir(osp.join(args["PATHS"]["out"], "weights", self.args["Model"]["model"]), self.args["GENERAL"]["reset"])


    def train_one_epoch(self, epoch, lr_schedule):

        header = 'Epoch: [{}/{}]'.format(epoch, self.args["Trainer"]["epochs"])
        metric_logger = MetricLogger(self.train_gen, 0, header, delimiter="  ") # freq 10

        # skipped = 0
        progress_bar = tqdm(total=len(self.train_gen), desc=f"Epoch {epoch} progress")
        for it, values in enumerate(metric_logger):
            # === Global Iteration === #
            it = len(self.train_gen) * epoch + it

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_schedule[it]

            # === Inputs === #
            input_data = values.x.cuda(non_blocking=True)
            labels = values.y.cuda(non_blocking=True)
            edge_indices = values.edge_index.cuda(non_blocking=True)
            edge_attr = values.edge_attr.cuda(non_blocking=True)

            # #########################
            # # check if input data is too big
            # MB = 1024 * 1024
            # if input_data.element_size() * input_data.nelement() > 90 * MB:
            #     skipped += 1
            #     self.print(f"Skipped since too big. Counting total {skipped} skipped files.")
            #     continue

            # === Forward pass === #
            # GB = 1024 * 1024 * 1024
            # print(torch.cuda.memory_summary())
            with torch.cuda.amp.autocast(self.args["Trainer"]["fp16"]):
                # print("bef-model: ", torch.cuda.memory_allocated() / GB, "GB")
                # print(torch.cuda.memory_summary())
                preds = self.model(input_data, edge_indices, edge_attr)
                # print("aft-model: ", torch.cuda.memory_allocated() / GB, "GB")
                # print(torch.cuda.memory_summary())
                loss, true_positives, true_negatives = self.loss(preds, labels)

            # Sanity Check
            if not math.isfinite(loss.item()):
                self.print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            
            # === Backward pass === #
            self.model.zero_grad()

            # print("bef-backprop: ", torch.cuda.memory_allocated() / GB, "GB")
            # print(torch.cuda.memory_summary())
            if self.args["Trainer"]["fp16"]:
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            # print("aft-backprop: ", torch.cuda.memory_allocated() / GB, "GB")
            # print(torch.cuda.memory_summary())

            # === Logging === #
            # start_time = time.time()
            # self.print("Waiting to syncronize.")
            torch.cuda.synchronize()
            # sync_time = time.time() - start_time
            # self.print("Sync time = {}".format(datetime.timedelta(seconds=sync_time)))

            metric_logger.update(loss=loss.item(), true_positives=true_positives, true_negatives=true_negatives)

            if self.args["main"]:
                self.loss_writer(metric_logger.meters['loss'].value, it)
                self.lr_sched_writer(self.optimizer.param_groups[0]["lr"], it)
            del input_data
            torch.cuda.empty_cache()
            gc.collect()
            progress_bar.update(1)

        # self.print("Waiting for other processes to catch up.")
        # metric_logger.synchronize_between_processes()
        # self.print("Averaged stats:", metric_logger)


    def fit(self):

        # === Resume === #
        self.load_if_available()

        # === Schedules === #
        lr_schedule = cosine_scheduler(
                        base_value = self.args["Optimization"]["lr_start"] * (self.args["Dataset"]["batch_per_gpu"] * self.args["world_size"]) / 256.,
                        final_value = self.args["Optimization"]["lr_end"],
                        epochs = self.args["Trainer"]["epochs"],
                        niter_per_ep = len(self.train_gen),
                        warmup_epochs= self.args["Optimization"]["lr_warmup"],
        )

        # === training loop === #
        main_progress_bar = tqdm(total=self.args["Trainer"]["epochs"], desc="Training process")
        for epoch in range(self.start_epoch, self.args["Trainer"]["epochs"]):

            self.train_gen.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch, lr_schedule)

            main_progress_bar.update(1)

            # === save model === #
            if self.args["main"] and epoch%self.args["Trainer"]["save_every"] == 0:
                self.validate(epoch)
                self.save(epoch)

    def load_if_available(self):

        ckpts = sorted(glob(osp.join(self.args["PATHS"]["out"], "weights", self.args["Model"]["model"], "Epoch_*.pth")))

        if len(ckpts) > 0:
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.args["Trainer"]["fp16"]: self.fp16_scaler.load_state_dict(ckpt['fp16_scaler'])
            self.print("Loaded ckpt: ", ckpts[-1])

        else:
            self.start_epoch = 0
            self.print("Starting from scratch")

    def validate(self, epoch):
        """Runs the validation loop."""
        self.model.eval()
        header = 'Validation Epoch: [{}/{}]'.format(epoch, self.args["Trainer"]["epochs"])
        metric_logger = MetricLogger(self.validation_gen, 10, header, delimiter="  ")

        with torch.no_grad():
            for values in metric_logger:
                # === Inputs === #
                input_data = values.x.cuda(non_blocking=True)
                labels = values.y.cuda(non_blocking=True)
                edge_indices = values.edge_index.cuda(non_blocking=True)
                edge_attr = values.edge_attr.cuda(non_blocking=True)

                # === Forward pass === #
                # preds = self.model(input_data)
                preds = self.model(input_data, edge_indices, edge_attr)
                loss, true_positives, true_negatives = self.loss(preds, labels)

                # === Logging === #
                torch.cuda.synchronize()
                metric_logger.update(loss=loss.item(), true_positives=true_positives, true_negatives=true_negatives)

            # Log validation loss
            if self.args["main"]:
                self.val_loss_writer(metric_logger.meters['loss'].value, epoch)

        metric_logger.synchronize_between_processes()
        self.print("Averaged Validation stats:", metric_logger)
        self.model.train()

    def save(self, epoch):

        if self.args["Trainer"]["fp16"]:
            state = dict(epoch=epoch+1, 
                            model=self.model.state_dict(), 
                            optimizer=self.optimizer.state_dict(), 
                            fp16_scaler = self.fp16_scaler.state_dict(),
                            args = self.args
                        )
        else:
            state = dict(epoch=epoch+1, 
                            model=self.model.state_dict(), 
                            optimizer=self.optimizer.state_dict(),
                            args = self.args
                        )

        torch.save(state, "{}/weights/{}/Epoch_{}.pth".format(self.args["PATHS"]["out"], self.args["Model"]["model"], str(epoch).zfill(3) ))
