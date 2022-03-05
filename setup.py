from setuptools import setup, find_packages

setup(
	name="nn",
	version="0.1.1",
	author="Thomas Mazumder",
	author_email="thomas.mazumder@ucsf.edu",
	packages = find_packages(include=['nn','nn.*']),
	install_requires = ['numpy>=1.14.5',]
)
