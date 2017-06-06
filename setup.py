from setuptools import setup

def readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(name='horsetailmatching',
        version='0.4',
        long_description=readme(),
        url='https://www-edc.eng.cam.ac.uk/aerotools/horsetailmatching',
        download_url='https://www-edc.eng.cam.ac.uk/aerotools/horsetailmatching',
        author='Laurence W. Cook',
        author_email='lwc24@cam.ac.uk',
        install_requires=['numpy', 'scipy >= 0.15.0', 'matplotlib'],
        license='MIT',
        packages=['horsetailmatching'],
        test_suite='nose.collector',
        tests_require=['nose'],
        include_package_data=True,
        zip_safe=False)
