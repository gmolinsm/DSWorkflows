from setuptools import setup

VERSION = '0.0.2' 
DESCRIPTION = 'Data Science Workflows'
LONG_DESCRIPTION = 'A Python package for encompasing the most typical workflow when Scikit-Learn and Pandas'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="DSWorkflows", 
        version=VERSION,
        author="Guillermo Molins",
        author_email="<guimolins@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=['DSWorkflows'],
        install_requires=['matplotlib', 'numpy', 'pandas', 'IPython', 'imblearn', 'scikit-learn'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows",
        ]
)