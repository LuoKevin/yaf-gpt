from pathlib import Path

from setuptools import setup

PROJECT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    setup(
        name="yaf-gpt",
        version="0.0.3",
        description="Yet another FastAPI-backed GPT experimentation playground.",
        author="Kevin Luo",
        author_email="luokev1@gmail.com",
        url="https://github.com/luokevin/yaf-gpt",
        project_urls={
            "Source": "https://github.com/luokevin/yaf-gpt",
        },
        license="MIT",
        packages=["yaf_gpt"],
        package_dir={"": "src"},
        include_package_data=True,
        python_requires=">=3.10",
    )
