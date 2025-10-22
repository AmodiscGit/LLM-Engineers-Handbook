from llm_engineering.application.rag.retriever import ContextRetriever


def print_results(query: str, results):
    print('\n' + '=' * 80)
    print(f"Query: {query}\n")
    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, start=1):
        # Robustly extract provenance and snippet from the returned EmbeddedChunk
        cls_name = r.__class__.__name__
        _id = getattr(r, 'id', None)

        # possible fields on article/repository chunks
        link = getattr(r, 'link', None)
        name = getattr(r, 'name', None)

        # content lives on 'content'
        content = getattr(r, 'content', None)

        # fallback to model_dump if available
        if content is None:
            try:
                dump = r.model_dump() if hasattr(r, 'model_dump') else None
                content = dump.get('content') if dump else None
                if content is None:
                    # try common keys
                    for k in ('text', 'chunk_text', 'body'):
                        if dump and k in dump:
                            content = dump[k]
                            break
            except Exception:
                content = None

        source = link or name or 'unknown'

        print(f"{i}. class={cls_name} id={_id} source={source}\n   snippet={str(content)[:300]}\n")


def main():
    retriever = ContextRetriever()

    queries = [
        "methods to fix a fracture",
        "non-accidental injury in infants",
        "open source repository replication guide",
    ]

    for q in queries:
        results = retriever.search(q, k=3, expand_to_n_queries=3)
        print_results(q, results)


if __name__ == "__main__":
    main()
