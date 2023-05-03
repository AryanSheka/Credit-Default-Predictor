from setuptools import find_packages,setup
from typing import List 

HYPHEN_E_DOT ='-e .'
def find_packages(file_path:str) ->List[str]:
    # This function finds returns a list of all packages
    requirements =[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="Project1",
    version='0.0.1',
    author='Aryan Sheka',
    author_email="titanium3602@gmail.com",
    install_requires=find_packages('requirements.txt')
)
