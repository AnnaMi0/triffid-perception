from setuptools import setup
import os
from glob import glob

package_name = 'triffid_uav_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='triffid',
    maintainer_email='amichailidou@hua.gr',
    description='TRIFFID UAV perception pipeline',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'uav_node = triffid_uav_perception.uav_node:main',
        ],
    },
)
