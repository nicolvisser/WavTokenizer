import os

from pytorch_lightning.cli import ArgsType, LightningCLI

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def cli_main(args: ArgsType = None):
    # breakpoint()
    cli = LightningCLI(args=args)
    # breakpoint()
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
