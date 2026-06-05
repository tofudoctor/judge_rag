# main.py
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import sys

from .searching.pipeline import QuickSearchPipeline, FullSearchPipeline


OUTPUT_PATH = Path(__file__).with_name("benchmark_results.txt")
QUICK_OUTPUT_PATH = Path(__file__).with_name("quicksearch_results.txt")
FULL_OUTPUT_PATH = Path(__file__).with_name("fullsearch_results.txt")

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "case_type": "civil",
        "query": "非原住民為購買取得原住民保留地，與原住民成立借名登記契約，再由該出名人向土地所有權人購買取得土地。其等間之借名登記契約是否有效？",
    },
    {
        "id": 2,
        "case_type": "civil",
        "query": "法人之名譽或信用受侵害，可否依民法第195條第1項規定請求賠償？",
    },
    {
        "id": 3,
        "case_type": "civil",
        "query": "河川浮覆地原所有權人依民法第767條第1項規定行使物上請求權時，有無消滅時效規定之適用？",
    },
    {
        "id": 4,
        "case_type": "criminal",
        "query": "詐欺犯罪的行為人如果並未實際取得個人所得，其犯罪所得如何認定？是否得依詐欺犯罪危害防制條例第47條前段減輕其刑？",
    },
    {
        "id": 5,
        "case_type": "criminal",
        "query": "民意代表在議場外，對行政機關為關說、請託或施壓，是否該當於貪污治罪條例第5條第1項第3款公務員職務受賄罪之職務上之行為？",
    },
    {
        "id": 6,
        "case_type": "criminal",
        "query": "裁定准予交付審判之法官，是否需要迴避本案審判？",
    },
    {
        "id": 7,
        "case_type": None,
        "query": "怎麼打籃球？",
    },
]


def fmt(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else value


def get_timing(record, key):
    return record.get("timing", {}).get(key, "X")


def sum_numeric(values):
    return round(sum(v for v in values if isinstance(v, (int, float))), 2)


def avg(records, key):
    values = [get_timing(r, key) for r in records]
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else "X"


def avg_total(records, columns):
    column_averages = [avg(records, col) for col in columns]
    nums = [v for v in column_averages if isinstance(v, (int, float))]
    return round(sum(nums), 2) if nums else "X"


def format_jids(record):
    refs = record.get("ref_details", [])[:20]
    if not refs:
        return "無"
    return "<br>".join(
        f"{ref['JID']} (score: {ref['score']})"
        for ref in refs
    )


def print_quick_table(records):
    columns = ["retrieve", "rerank", "generate"]
    print("\n| 次數 | Retrieve | Rerank | Generate | Total |")
    print("| --- | ---- | ---- | ----- | ----- |")

    for record in records:
        retrieve = get_timing(record, "retrieve")
        rerank = get_timing(record, "rerank")
        generate = get_timing(record, "generate")
        total = sum_numeric([retrieve, rerank, generate])
        print(
            f"| {record['query_id']} | "
            f"{fmt(retrieve)} | {fmt(rerank)} | {fmt(generate)} | {fmt(total)} |"
        )

    print(
        f"| avg | {fmt(avg(records, 'retrieve'))} | "
        f"{fmt(avg(records, 'rerank'))} | "
        f"{fmt(avg(records, 'generate'))} | "
        f"{fmt(avg_total(records, columns))} |"
    )

def md_cell(value):
    if value is None:
        return ""
    return str(value).replace("\n", "<br>").replace("|", "／")

def print_generation_history(record):
    history = record.get("generation_history") or []
    if not history:
        return

    print("對話式生成紀錄：")
    for index, item in enumerate(history, start=1):
        role = item.get("role", "unknown")
        content = item.get("log_content") or item.get("content") or ""
        role_label = "User" if role == "user" else "Assistant" if role == "assistant" else role
        print(f"--- [{index}] {role_label} ---")
        print(content)

def print_full_table(records):
    columns = [
        "rewrite",
        "retrieve",
        "rerank",
        "doc_grade",
        "generate1",
        "hallucination1",
        "generate2",
        "hallucination2",
        "generate3",
    ]
    print("\n| 次數 | Rewrite | Retrieve | Rerank | doc grade | Generate1 | 幻覺grade1 | Generate2 | 幻覺grade2 | Generate3 | Total |")
    print("| ---- | --- | -------- | ------ | --- | -------- | --- | --- | --- | --- | --- |")

    for record in records:
        values = {column: get_timing(record, column) for column in columns}
        total = sum_numeric(values.values())
        print(
            f"| {record['query_id']} | {fmt(values['rewrite'])} | "
            f"{fmt(values['retrieve'])} | {fmt(values['rerank'])} | "
            f"{fmt(values['doc_grade'])} | {fmt(values['generate1'])} | "
            f"{fmt(values['hallucination1'])} | {fmt(values['generate2'])} | "
            f"{fmt(values['hallucination2'])} | {fmt(values['generate3'])} | "
            f"{fmt(total)} |"
        )

    print(
        f"| avg | {fmt(avg(records, 'rewrite'))} | "
        f"{fmt(avg(records, 'retrieve'))} | "
        f"{fmt(avg(records, 'rerank'))} | "
        f"{fmt(avg(records, 'doc_grade'))} | "
        f"{fmt(avg(records, 'generate1'))} | "
        f"{fmt(avg(records, 'hallucination1'))} | "
        f"{fmt(avg(records, 'generate2'))} | "
        f"{fmt(avg(records, 'hallucination2'))} | "
        f"{fmt(avg(records, 'generate3'))} | "
        f"{fmt(avg_total(records, columns))} |"
    )


def run_benchmark(title, pipeline_cls, model):
    print(f"\n\n{title} 使用{model}")
    pipeline = pipeline_cls(model=model)
    records = []

    for benchmark_query in BENCHMARK_QUERIES:
        query_id = benchmark_query["id"]
        query = benchmark_query["query"]
        case_type = benchmark_query["case_type"]
        case_type_label = case_type or "all"

        print(f"\n===== 題目 {query_id} / {title} / {model} / case_type={case_type_label} =====")
        print(f"問題：{query}")
        result = pipeline.run(query=query, case_type=case_type)
        result["query_id"] = query_id
        result["case_type_label"] = case_type_label
        records.append(result)

        refs = result.get("ref_details", [])[:20]
        refs_text = ", ".join(
            f"{ref['JID']} (score: {ref['score']})"
            for ref in refs
        ) or "無"
        print(f"引用判決 Top 20：{refs_text}")
        if "doc_grade_reason" in result:
            print(f"DocGrader reason：{result.get('doc_grade_reason') or '無'}")
        if "hallucination_reason" in result:
            print(f"HallucinationGrader reason：{result.get('hallucination_reason') or '無'}")
        print_generation_history(result)
        print("回答摘要：")
        print(result.get("answer", "無"))

    return records

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()



def parse_args():
    parser = argparse.ArgumentParser(description="Run judge_rag benchmark tests.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--quick", action="store_true", help="Run QuickSearch only.")
    mode_group.add_argument("--full", action="store_true", help="Run FullSearch only.")
    return parser.parse_args()


def resolve_output_path(args):
    if args.quick:
        return QUICK_OUTPUT_PATH
    return FULL_OUTPUT_PATH


def select_benchmarks(benchmarks, args):
    title = "QuickSearch" if args.quick else "FullSearch"
    return [item for item in benchmarks if item[0] == title]


def main(args=None, output_path=OUTPUT_PATH):
    if args is None:
        args = parse_args()
    benchmarks = [
        ("QuickSearch", QuickSearchPipeline, "gpt-oss:latest", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "gpt-oss:120b", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "gemma4:latest", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "gemma4:26b", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "gemma4:31b", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "mistral-small3.2:latest", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "nemotron-3-nano:4b", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "nemotron3:33b", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "nemotron-3-super:latest", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "granite4.1:8b", print_quick_table),
        ("QuickSearch", QuickSearchPipeline, "granite4.1:30b", print_quick_table),

        ("FullSearch", FullSearchPipeline, "gpt-oss:latest", print_full_table),
        ("FullSearch", FullSearchPipeline, "gpt-oss:120b", print_full_table),
        ("FullSearch", FullSearchPipeline, "gemma4:latest", print_full_table),
        ("FullSearch", FullSearchPipeline, "gemma4:26b", print_full_table),
        ("FullSearch", FullSearchPipeline, "gemma4:31b", print_full_table),
        ("FullSearch", FullSearchPipeline, "mistral-small3.2:latest", print_full_table),
        ("FullSearch", FullSearchPipeline, "nemotron-3-nano:4b", print_full_table),
        ("FullSearch", FullSearchPipeline, "nemotron3:33b", print_full_table),
        ("FullSearch", FullSearchPipeline, "nemotron-3-super:latest", print_full_table),
        ("FullSearch", FullSearchPipeline, "granite4.1:8b", print_full_table),
        ("FullSearch", FullSearchPipeline, "granite4.1:30b", print_full_table),
    ]

    benchmarks = select_benchmarks(benchmarks, args)

    print(f"Benchmark output: {output_path}")
    print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Only pipeline: {'QuickSearch' if args.quick else 'FullSearch'}")

    for title, pipeline_cls, model, printer in benchmarks:
        records = run_benchmark(title, pipeline_cls, model)
        printer(records)


if __name__ == "__main__":
    args = parse_args()
    output_path = resolve_output_path(args)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write("\n" + "=" * 80 + "\n")
        with redirect_stdout(Tee(sys.stdout, output_file)), redirect_stderr(Tee(sys.stderr, output_file)):
            main(args=args, output_path=output_path)
