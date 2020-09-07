from setuptools import setup, find_packages

setup(name='pyrepgym',
      version='0.1',
      packages=find_packages(include=['pyrepgym']),
      data_files=[('config', ['pyrepgym/envs/weights.npy'],)],
      install_requires=['gym', 'numpy', 'pyquaternion'])
