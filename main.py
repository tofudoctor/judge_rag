# main.py
from .searching.pipeline import QuickSearchPipeline, FullSearchPipeline

if __name__ == "__main__":

    pipeline = FullSearchPipeline(case_type="civil")

    answer = pipeline.run("法人之名譽或信用受侵害，可否依民法第195條第1項規定請求賠償？")

    print(answer)