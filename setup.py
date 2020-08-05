import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='nested_coref_resolver',
    version='0.0.2',
    author='Griffin Adams',
    author_email='griffin.adams@columbia.edu',
    description='light-weight python script to transform documents by replacing all mentions of co-referent clusters '
                'with first non-pronominal mention. Importantly, it explicitly handles nested coreferent mentions.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/griff4692/resolve_coref_clusters',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=2.6',
)
