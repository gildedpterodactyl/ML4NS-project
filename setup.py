from setuptools import setup, find_packages

setup(
      name='proteinfoundation',
      version='1.0.0',
      description='Diffusion Protein AutoEncoder',
      packages=find_packages(),
      package_dir={
          'proteinfoundation': './proteinfoundation',
          'openfold': './openfold',
          'graphein_utils': './graphein_utils'
      }
)