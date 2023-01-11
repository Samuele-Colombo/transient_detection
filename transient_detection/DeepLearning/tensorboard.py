"""Mostly copy-pasted from https://github.com/ramyamounir/Template/blob/main/lib/utils/tensorboard.py"""

from torch.utils.tensorboard import SummaryWriter
import pprint, re

from transient_detection.DeepLearning.fileio import checkdir

def get_writer(args):
    path = "{}/logs/{}".format(args["PATHS"]["out"], args["Model"]["model"])
    checkdir(path, args["GENERAL"]["reset"])

    writer = SummaryWriter(path)
    writer.add_text('config', re.sub("\n", "  \n", pprint.pformat(args, width = 1)), 0)
    writer.flush()

    if args["GENERAL"]["tb"]:
        def start_tb():
            import subprocess
            command = ["tensorboard", "--samples_per_plugin", "images=0", "--logdir", path]
            subprocess.call(command)

        import threading
        threading.Thread(target=start_tb).start()

    return writer


class TBWriter(object):

    def __init__(self, writer, data_type, tag, mul = 1, add = 0, fps = 4):

        self.step = 0
        self.mul = mul
        self.add = add
        self.fps = fps

        self.writer = writer
        self.type = data_type
        self.tag = tag

    def __call__(self, data, step = None, flush = False):

        counter = step if step != None else self.step*self.mul+self.add

        if self.type == 'scalar':
            self.writer.add_scalar(self.tag, data, global_step = counter)
        elif self.type == 'scalars':
            self.writer.add_scalars(self.tag, data, global_step = counter)
        elif self.type == 'image':
            self.writer.add_image(self.tag, data, global_step = counter)
        elif self.type == 'video':
            self.writer.add_video(self.tag, data, global_step = counter, fps = self.fps)
        elif self.type == 'figure':
            self.writer.add_figure(self.tag, data, global_step = counter)
        elif self.type == 'text':
            self.writer.add_text(self.tag, data, global_step = counter)

        self.step += 1

        if flush:
            self.writer.flush()
