# preprocess.py
from .indexing.pipeline import BuildPipeline

if __name__ == "__main__":

    build = BuildPipeline()

    civil_docs = build.run(
        base_dir="最高法院民事",
        case_type="civil",
    )

    family_docs = build.run(
        base_dir="最高法院家事",
        case_type="civil",
    )

    criminal_docs = build.run(
        base_dir="最高法院刑事",
        case_type="criminal",
    )