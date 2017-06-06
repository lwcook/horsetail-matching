from distutils.core import setup

def readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(name='horsetailmatching',
        version='0.3',
        description='A method for optimization under uncertainty',
        url='https://www-edc.eng.cam.ac.uk/aerotools/horsetailmatching',
        download_url='https://github.com/lwcook/horsetail-matching/archive/0.3.targ.gz',
        classifiers=[
            'License :: MIT License',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering :: Mathematics'],
        author='Laurence W. Cook',
        author_email='lwc24@cam.ac.uk',
        install_requires=['numpy', 'scipy', 'matplotlib'],
        license='MIT',
        packages=['horsetailmatching'],
        test_suite='nose.collector',
        tests_require=['nose'])
