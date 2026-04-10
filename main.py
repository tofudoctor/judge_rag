# main.py
from .indexing.pipeline import BuildPipeline
from .searching.pipeline import SearchPipeline

if __name__ == "__main__":

    build = BuildPipeline()

    # civil_docs = build.run(
    #     base_dir="最高法院民事",
    #     case_type="civil",
    #     n_years=2
    # )

    # family_docs = build.run(
    #     base_dir="最高法院家事",
    #     case_type="civil",
    #     n_years=2
    # )

    # criminal_docs = build.run(
    #     base_dir="最高法院刑事",
    #     case_type="criminal",
    #     n_years=2
    # )

    pipeline = SearchPipeline(case_type="criminal")

    answer = pipeline.run("竊盜罪的構成要件是什麼？")

    print(answer)