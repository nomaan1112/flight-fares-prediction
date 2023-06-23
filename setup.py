from setuptools import find_packages, setup
from typing import List

REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

##function to get the list of packages that are required and also to remove the "-e ." from that list as during package distribution we used find_package() wjich will find teh source code and also "-e ." is not a package
def get_requirements()->List[str]:                           
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list=[requirement_name.replace("\n","") for requirement_name in requirement_list]
    
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list


##information of the package, useful during package distribution
setup(
    name = "flight_fare_prediction",
    version = "0.0.1",
    author = "Rajesh",
    author_email = "rajesh29049495@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements()
)