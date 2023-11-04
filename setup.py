from setuptools import setup, find_packages
from typing import List
Hyphen_E_dot = "e ." 

def get_requires(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as f:
      requirements= f.readlines()
    requirements = [x.replace("\n","") for x in requirements]
    
    if Hyphen_E_dot in requirements:
        requirements.remove(Hyphen_E_dot) 
        
    return requirements  




setup(
    name='Diamond_price_prediction',
    version='0.0.1',
    author='Virender Chauhan',
    author_email="virchauhan657@gmail.com",
    install_requires=get_requires("requirements.txt"),
    packages=find_packages(),
    description='Diamond price prediction',
)
