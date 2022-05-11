import shutil, os, argparse

arg = argparse.ArgumentParser()
arg.add_argument('dir')
arg = arg.parse_args()
shutil.rmtree(os.path.join('tensorboard_log', arg.dir))
shutil.rmtree(os.path.join('save', arg.dir))

