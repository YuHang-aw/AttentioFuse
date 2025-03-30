# setup.py

import setuptools
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Multi-omics fusion model using attention mechanisms (AttentionFusion)." # Fallback description


requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()
else:
    # Fallback if requirements.txt is missing (not recommended)
    install_requires = [
        'pandas', 'numpy', 'scikit-learn', 'torch',
        'matplotlib', 'seaborn', 'gmt-python', 'imbalanced-learn'
    ]
    print("Warning: requirements.txt not found. Using fallback dependencies.")


setuptools.setup(
    name="attentiofuse", 
    version="0.1.0",       
    description="Multi-omics data fusion using attention mechanisms for cancer stage prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown", 
    classifiers=[ 
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    # --- 核心：指定包含的包 ---
    # find_packages 会自动查找包含 __init__.py 的目录
    packages=setuptools.find_packages(
        where='.', # 在当前目录下查找
        include=['dataprocess*', 'model*'], # 明确包含哪些包
        exclude=[] 
        ),
    python_requires=">=3.8", 
    install_requires=install_requires,
)

print("\n--- setup.py finished ---")
print("To build the package: python setup.py sdist bdist_wheel")
print("To install locally (editable): pip install -e .")
print("To install from built wheel: pip install dist/attentionfusion-*.whl")