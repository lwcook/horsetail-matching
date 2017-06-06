from setuptools import setup

setup(name='horsetailmatching',
        version='0.5',
        url='https://www-edc.eng.cam.ac.uk/aerotools/horsetailmatching',
        download_url='https://www-edc.eng.cam.ac.uk/aerotools/horsetailmatching',
        author='Laurence W. Cook',
        author_email='lwc24@cam.ac.uk',
        install_requires=['numpy >= 1.12.1'],
        license='MIT',
        packages=['horsetailmatching'],
        test_suite='nose.collector',
        tests_require=['nose'],
        include_package_data=True,
        zip_safe=False)
