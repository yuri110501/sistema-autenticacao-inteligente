from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="sistema_autenticacao_inteligente",
    version="0.1.0",
    author="Seu Nome",
    author_email="seu.email@exemplo.com",
    description="Sistema de autenticação inteligente que utiliza diferentes métodos de autenticação para diferentes setores",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yuri110501/sistema-autenticacao-inteligente",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "autenticacao-api=run_api:main",
            "autenticacao-app=run_app:main",
        ],
    },
) 