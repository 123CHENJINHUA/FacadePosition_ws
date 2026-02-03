from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'qwen_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'openai'],
    zip_safe=True,
    maintainer='cjh',
    maintainer_email='jchenjb@connect.ust.hk',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'qwen_service = qwen_pkg.qwen_service:main',
            'qwen_client = qwen_pkg.qwen_client:main',
            'image_publisher = qwen_pkg.image_publisher:main',
            'image_show = qwen_pkg.image_show:main',
            'qwen_service_sim = qwen_pkg.qwen_service_sim:main',
        ],
    },
)
