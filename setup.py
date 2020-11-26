from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

def requirement():
    return [
        'pandas',
        'scikit-learn'
    ]

setup(
    name='skl-utils',
    packages=find_packages(),
    version='0.0.1',
    description='Scikit-Learn Ultilities',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='LEANG Sotheara',
    author_email='leangsotheara@gmail.com',
    url='https://github.com/sotheara-leang/skl-utils.git',
    keywords=['scikit-learn', 'pandas', 'utilities'],
    install_requires=requirement(),
    python_requires=">=3.5",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License'
    ]
)
