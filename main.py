# main.py
from .searching.pipeline import QuickSearchPipeline

if __name__ == "__main__":

    pipeline = QuickSearchPipeline(case_type="criminal")

    answer = pipeline.run("詐欺犯罪是甚麼?")

    print(answer)